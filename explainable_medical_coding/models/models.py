from typing import Callable, Optional

import torch
import torch.utils.checkpoint
from pydantic import BaseModel
from torch import nn
from transformers import AutoConfig, AutoModel
#from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

from explainable_medical_coding.explainability.helper_functions import (
    create_baseline_input,
)
from explainable_medical_coding.models.modules.attention import (
    InputMasker,
    LabelAttention,
    LabelCrossAttention,
)


class ModuleNames(BaseModel):
    ln_1: str
    ln_2: str
    dense_values: str
    dense_heads: str
    model_layer_name: str


class RobertaModuleNames(ModuleNames):
    ln_1: str = "attention.output.LayerNorm"
    ln_2: str = "output.LayerNorm"
    dense_values: str = "attention.self.value"
    dense_heads: str = "attention.output.dense"
    model_layer_name: str = "roberta_encoder.encoder.layer"

class ModernBertModuleNames(ModuleNames):
    ln_1: str = "attn_norm"
    ln_2: str = "mlp_norm"
    dense_values: str = "attn.Wqkv"
    dense_heads: str = "attn.Wo"
    model_layer_name: str = "modernbert.layers"


class PLMICD(nn.Module):
    def __init__(
        self,
        num_classes: int,
        model_path: str,
        chunk_size: int,
        pad_token_id: int,
        cross_attention: bool = True,
        scale: float = 1.0,
        mask_input: bool = False,
        use_peft: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.pad_token_id = pad_token_id
        
        # Set module names based on model type
        if "modernbert" in model_path.lower():
            self.module_names = ModernBertModuleNames()
        else:
            self.module_names = RobertaModuleNames()
        
        self.gradient = None

        self.config = AutoConfig.from_pretrained(
            model_path, num_labels=num_classes, finetuning_task=None
        )

        print(f"[DEBUG INIT] Loading model from: {model_path}")
        print(f"[DEBUG INIT] Config vocab_size: {self.config.vocab_size}")
        print(f"[DEBUG INIT] Config pad_token_id: {getattr(self.config, 'pad_token_id', 'not set')}")
        print(f"[DEBUG INIT] Config max_position_embeddings: {getattr(self.config, 'max_position_embeddings', 'not set')}")
        
        # Use the working approach from the successful code
        base_model = AutoModel.from_pretrained(
            model_path, config=self.config
        )
        self.roberta_encoder = base_model
        
        print(f"[DEBUG INIT] Model type: {type(self.roberta_encoder)}")
        print(f"[DEBUG INIT] Model initialization complete")

        if cross_attention:
            self.label_wise_attention = LabelCrossAttention(
                input_size=self.config.hidden_size, num_classes=num_classes, scale=scale
            )
        else:
            self.label_wise_attention = LabelAttention(
                input_size=self.config.hidden_size,
                projection_size=self.config.hidden_size,
                num_classes=num_classes,
            )
        self.mask_input = mask_input
        if self.mask_input:
            self.input_masker = InputMasker(
                input_size=self.config.hidden_size, scale=scale
            )

    @torch.no_grad()
    def predict(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.forward(input_ids=input_ids, attention_masks=attention_mask)
        )

    def split_input_into_chunks(
        self, input_sequence: torch.Tensor, pad_index: int
    ) -> torch.Tensor:
        """Split input into chunks of chunk_size.

        Args:
            input_sequence (torch.Tensor): input sequence to split (batch_size, seq_len)
            pad_index (int): padding index

        Returns:
            torch.Tensor: reshaped input (batch_size, num_chunks, chunk_size)
        """
        batch_size = input_sequence.size(0)
        # pad input to be divisible by chunk_size
        input_sequence = nn.functional.pad(
            input_sequence,
            (0, self.chunk_size - input_sequence.size(1) % self.chunk_size),
            value=pad_index,
        )
        return input_sequence.view(batch_size, -1, self.chunk_size)

    def roberta_encode_embedding_input(self, embedding, attention_masks):
        # Generic approach that works for both RoBERTa and ModernBERT
        if hasattr(self.roberta_encoder, 'encoder'):
            # RoBERTa style
            input_shape = embedding.size()[:-1]
            extended_attention_mask = self.roberta_encoder.get_extended_attention_mask(
                attention_masks, input_shape
            )
            head_mask = self.roberta_encoder.get_head_mask(
                None, self.roberta_encoder.config.num_hidden_layers
            )
            encoder_outputs = self.roberta_encoder.encoder(
                embedding,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                return_dict=False,
            )
            sequence_output = encoder_outputs[0]
        else:
            # ModernBERT style - direct forward pass
            # Handle chunked inputs properly
            batch_chunks, seq_len, hidden_dim = embedding.shape
            batch_chunks_mask, seq_len_mask = attention_masks.shape
            
            # Ensure shapes match
            assert batch_chunks == batch_chunks_mask and seq_len == seq_len_mask, \
                f"Shape mismatch: embedding {embedding.shape}, attention_mask {attention_masks.shape}"
            
            sequence_output = self.roberta_encoder(
                inputs_embeds=embedding,
                attention_mask=attention_masks,
                return_dict=False
            )[0]
        
        return sequence_output

    def get_chunked_attention_masks(
        self, attention_masks: torch.Tensor
    ) -> torch.Tensor:
        return self.split_input_into_chunks(attention_masks, 0)

    def get_input_embeddings(self):
        return self.roberta_encoder.embeddings

    def get_chunked_embedding(self, input_ids):
        input_ids = self.split_input_into_chunks(input_ids, self.pad_token_id)
        chunk_size = input_ids.size(-1)
        
        # Handle different embedding layer names
        if hasattr(self.roberta_encoder.embeddings, 'tok_embeddings'):
            # ModernBERT style
            embedding = self.roberta_encoder.embeddings.tok_embeddings(input_ids.view(-1, chunk_size))
        else:
            # RoBERTa style
            embedding = self.roberta_encoder.embeddings(input_ids.view(-1, chunk_size))
        return embedding

    def get_token_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings. RoBERTa has 512 token limit, ModernBERT supports up to 8192.

        Args:
            input_ids (torch.Tensor): input ids

        Returns:
            torch.Tensor: token embeddings
        """
        sequence_length = input_ids.size(1)
        batch_size = input_ids.size(0)
        
        # Handle different embedding layer names and size limits
        if hasattr(self.roberta_encoder.embeddings, 'tok_embeddings'):
            # ModernBERT - likely doesn't have 512 token limit
            embedding_layer = self.roberta_encoder.embeddings.tok_embeddings
            max_chunk_size = 8192  # ModernBERT's max_position_embeddings
        else:
            # RoBERTa - has 512 token limit
            embedding_layer = self.roberta_encoder.embeddings
            max_chunk_size = 512
            
        if sequence_length <= max_chunk_size:
            return embedding_layer(input_ids)
            
        input_ids = self.split_input_into_chunks(input_ids, self.pad_token_id)
        chunk_size = input_ids.size(-1)
        chunked_embeddings = embedding_layer(input_ids.view(-1, chunk_size))
        embeddings = chunked_embeddings.view(
            batch_size, -1, chunked_embeddings.size(-1)
        )
        return embeddings[:, :sequence_length]

    def _call_encoder_safely(self, input_ids, attention_mask):
        """Wrapper to call roberta_encoder with proper parameters for both RoBERTa and ModernBERT"""
        
        # Fix for ModernBERT: Handle problematic sequences that cause NaN
        if attention_mask is not None and not hasattr(self.roberta_encoder, 'encoder'):
            # Only apply fixes for ModernBERT, not RoBERTa
            attention_mask = attention_mask.clone()
            input_ids = input_ids.clone()
            
            valid_tokens_per_seq = torch.sum(attention_mask, dim=1)
            zero_token_seqs = (valid_tokens_per_seq == 0)
            
            # Fix 1: Handle all-padding sequences
            if zero_token_seqs.any():
                num_empty = zero_token_seqs.sum().item()
                print(f"[FIX] Found {num_empty} all-padding sequences, fixing for ModernBERT...")
                
                # For all-padding sequences, set the first token to attend (prevents NaN)
                attention_mask[zero_token_seqs, 0] = 1
                
                # Use appropriate CLS token based on model type
                if "modernbert" in str(type(self.roberta_encoder)).lower():
                    cls_token_id = 50281  # ModernBERT CLS token
                else:
                    cls_token_id = getattr(self.roberta_encoder.config, 'cls_token_id', 101)  # RoBERTa CLS token
                input_ids[zero_token_seqs, 0] = cls_token_id
            
            # Fix 2: Handle very short sequences (1-2 tokens) that might be unstable
            very_short_seqs = (valid_tokens_per_seq > 0) & (valid_tokens_per_seq <= 2)
            if very_short_seqs.any():
                num_short = very_short_seqs.sum().item()
                print(f"[FIX] Found {num_short} very short sequences, padding for stability...")
                
                # Ensure at least 3 tokens are attended to
                for seq_idx in torch.where(very_short_seqs)[0]:
                    current_length = valid_tokens_per_seq[seq_idx].item()
                    if current_length < 3:
                        # Add attention to the next few tokens
                        attention_mask[seq_idx, current_length:3] = 1
                        # Set these tokens to something safe (EOS token based on model type)
                        if "modernbert" in str(type(self.roberta_encoder)).lower():
                            eos_token_id = 50282  # ModernBERT EOS token
                        else:
                            eos_token_id = getattr(self.roberta_encoder.config, 'eos_token_id', 2)  # RoBERTa EOS token
                        input_ids[seq_idx, current_length:3] = eos_token_id
        
        if hasattr(self.roberta_encoder, 'encoder'):
            # RoBERTa style - can use direct call
            print(f"[DEBUG ENCODER] Using RoBERTa-style call")
            outputs = self.roberta_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )
        else:
            # ModernBERT style call
            outputs = self.roberta_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )
        
        # Check for NaN in outputs
        if torch.isnan(outputs[0]).any():
            print(f"[DEBUG] ModernBERT still producing NaN - investigating...")
            
            # Check if specific sequences are causing issues
            nan_mask = torch.isnan(outputs[0]).any(dim=-1).any(dim=-1)  # [batch_size]
            num_nan_seqs = nan_mask.sum().item()
            print(f"[DEBUG] {num_nan_seqs} out of {outputs[0].shape[0]} sequences have NaN")
            
            if num_nan_seqs < outputs[0].shape[0]:  # Not all sequences have NaN
                # Check if it's related to sequence length or content
                nan_seq_indices = torch.where(nan_mask)[0]
                valid_seq_indices = torch.where(~nan_mask)[0]
                
                if len(nan_seq_indices) > 0 and len(valid_seq_indices) > 0:
                    print(f"[DEBUG] NaN sequence attention sums: {torch.sum(attention_mask[nan_seq_indices], dim=1)}")
                    print(f"[DEBUG] Valid sequence attention sums: {torch.sum(attention_mask[valid_seq_indices], dim=1)}")
                    print(f"[DEBUG] NaN sequence input ranges: {torch.min(input_ids[nan_seq_indices]).item()}-{torch.max(input_ids[nan_seq_indices]).item()}")
                    print(f"[DEBUG] Valid sequence input ranges: {torch.min(input_ids[valid_seq_indices]).item()}-{torch.max(input_ids[valid_seq_indices]).item()}")
            
            # Try a minimal working example to isolate the issue
            print(f"[DEBUG] Testing minimal sequence...")
            try:
                with torch.cuda.amp.autocast(enabled=False):
                    test_ids = torch.tensor([[0, 1, 2]], device=input_ids.device, dtype=input_ids.dtype)
                    test_mask = torch.tensor([[1, 1, 1]], device=attention_mask.device, dtype=attention_mask.dtype)
                    test_out = self.roberta_encoder(input_ids=test_ids, attention_mask=test_mask, return_dict=False)
                    print(f"[DEBUG] Minimal test: has_nan={torch.isnan(test_out[0]).any()}")
            except Exception as e:
                print(f"[DEBUG] Minimal test failed: {e}")
            
        return outputs

    def encoder(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_ids = self.split_input_into_chunks(input_ids, self.pad_token_id)
        
        if attention_masks is not None:
            attention_masks = self.get_chunked_attention_masks(attention_masks)
            
        batch_size, num_chunks, chunk_size = input_ids.size()
        
        # Use the working pattern from successful code
        outputs = self.roberta_encoder(
            input_ids.view(-1, chunk_size),
            attention_mask=attention_masks.view(-1, chunk_size)
            if attention_masks is not None
            else None,
            return_dict=False,
        )
        
        final_output = outputs[0].view(batch_size, num_chunks * chunk_size, -1)
        return final_output

    def forward_with_input_masking(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_mask: bool = False,
        baseline_token_id: int = 500001,
    ):
        baseline = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=baseline_token_id,
            eos_token_id=baseline_token_id,
        )
        baseline = baseline.requires_grad_(False)
        baseline_embeddings = self.get_chunked_embedding(baseline).detach()
        chunked_input_embeddings = self.get_chunked_embedding(input_ids)
        with torch.no_grad():
            token_representations = (
                self.get_token_representations_from_chunked_embeddings(
                    chunked_input_embeddings.detach(), attention_masks
                )
            )
        input_mask = self.input_masker(
            token_representations, attention_masks=attention_masks
        )

        input_mask_sigmoid = torch.sigmoid(input_mask)
        input_mask_sigmoid = input_mask_sigmoid.view(-1, self.chunk_size, 1)
        masked_chunked_input_embeddings = (
            chunked_input_embeddings * input_mask_sigmoid
            + baseline_embeddings * (1 - input_mask_sigmoid)
        )
        masked_token_representations = (
            self.get_token_representations_from_chunked_embeddings(
                masked_chunked_input_embeddings, attention_masks
            )
        )
        if output_mask:
            return self.label_wise_attention(
                masked_token_representations,
                attention_masks=attention_masks,
                output_attention=output_attentions,
            ), input_mask

        return self.label_wise_attention(
            masked_token_representations,
            attention_masks=attention_masks,
            output_attention=output_attentions,
        )

    def get_token_representations_from_chunked_embeddings(
        self,
        chunked_embedding: torch.Tensor,
        attention_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Get token representations from chunked embeddings.

        Args:
            chunked_embedding (torch.Tensor): Chunked embedding of shape [batch_size*num_chunks, chunk_size, embedding_size]
            attention_masks (torch.Tensor): Attention masks of shape [batch_size, sequence_length]

        Returns:
            torch.Tensor: Token representations of shape [batch_size, sequence_length, hidden_size]
        """
        (
            num_chunks_times_batch_size,
            chunk_size,
            embedding_size,
        ) = chunked_embedding.size()
        batch_size = attention_masks.size(0)
        num_chunks = num_chunks_times_batch_size // batch_size
        chunked_attention_masks = self.get_chunked_attention_masks(attention_masks)
        hidden_outputs = self.roberta_encode_embedding_input(
            embedding=chunked_embedding.view(-1, chunk_size, embedding_size),
            attention_masks=chunked_attention_masks.view(-1, chunk_size),
        )
        return hidden_outputs.view(batch_size, num_chunks * chunk_size, -1)

    def forward_embedding_input(
        self,
        chunked_embedding: torch.Tensor,
        attention_masks: torch.Tensor,
        output_attention: bool = False,
    ) -> torch.Tensor:
        """Forward pass for the model with chunked embedding input.

        Args:
            chunked_embedding (torch.Tensor): Chunked embedding of shape [batch_size*num_chunks, chunk_size, embedding_size]
            attention_masks (torch.Tensor): Attention masks of shape [batch_size, num_chunks, chunk_size]

        Returns:
            torch.Tensor:
        """
        token_representations = self.get_token_representations_from_chunked_embeddings(
            chunked_embedding, attention_masks
        )
        return self.label_wise_attention(
            token_representations,
            attention_masks=attention_masks,
            output_attention=output_attention,
        )

    @torch.no_grad()
    def de_chunk_attention(
        self,
        attentions_chunked: torch.Tensor,
        batch_size: int,
        num_layers: int,
        num_chunks: int,
        chunk_size: int,
    ) -> torch.Tensor:
        """De-chunk attention.

        Args:
            attentions_chunked (torch.Tensor): Attention matrix of shape [batch_size, num_chunks, num_layers, chunk_size, chunk_size]
            batch_size (int): Batch size
            num_layers (int): Number of layers
            num_chunks (int): Number of chunks
            chunk_size (int): Chunk size

        Returns:
            torch.Tensor: Attention matrix of shape [batch_size, num_layers, num_chunks*chunk_size, num_chunks*chunk_size]
        """
        attentions = torch.zeros(
            batch_size,
            num_layers,
            num_chunks * chunk_size,
            num_chunks * chunk_size,
            device=attentions_chunked.device,
            dtype=torch.float16,
        )
        for chunk_idx in range(num_chunks):
            attentions[
                :,
                :,
                chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size,
                chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size,
            ] = attentions_chunked[:, chunk_idx]

        return attentions

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        attn_grad_hook_fn: Optional[Callable] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.mask_input:
            return self.forward_with_input_masking(
                input_ids, attention_masks, output_attentions, False
            )
        
        hidden_output = self.encoder(input_ids, attention_masks)
        
        final_output = self.label_wise_attention(
            hidden_output,
            attention_masks=attention_masks,
            output_attention=output_attentions,
            attn_grad_hook_fn=attn_grad_hook_fn,
        )
        
        return final_output

    @torch.no_grad()
    def get_encoder_attention_and_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = self.split_input_into_chunks(input_ids, self.pad_token_id)
        if attention_masks is not None:
            attention_masks_chunks = self.split_input_into_chunks(attention_masks, 0)
        batch_size, num_chunks, chunk_size = input_ids.size()
        outputs = self.roberta_encoder(
            input_ids=input_ids.view(-1, chunk_size),
            attention_mask=attention_masks_chunks.view(-1, chunk_size)
            if attention_masks_chunks is not None
            else None,
            return_dict=False,
            output_attentions=True,
            output_hidden_states=True,
        )

        hidden_output = outputs[0].view(batch_size, num_chunks * chunk_size, -1)
        _, label_wise_attention = self.label_wise_attention(
            hidden_output, attention_masks, True
        )
        return outputs[2], outputs[3], label_wise_attention

    @torch.no_grad()
    def attention_rollout(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        (
            _,
            attentions,
            label_wise_attention,
        ) = self.get_encoder_attention_and_hidden_states(input_ids, attention_masks)

        label_wise_attention = torch.softmax(label_wise_attention, dim=1)
        attentions = (
            torch.stack(attentions).to(torch.float16).transpose(1, 0)
        )  # [batch_size*num_chunks, num_layers, num_heads, chunk_size, chunk_size]

        batch_size = input_ids.size(0)
        num_chunks = attentions.size(0) // batch_size
        num_layers = attentions.size(1)
        num_heads = attentions.size(2)
        chunk_size = attentions.size(3)

        attentions = attentions.view(
            batch_size, num_chunks, num_layers, num_heads, chunk_size, chunk_size
        )
        attentions = torch.mean(
            attentions, dim=3
        )  # [batch_size, num_chunks, num_layers, chunk_size, chunk_size]
        attentions = self.de_chunk_attention(
            attentions, batch_size, num_layers, num_chunks, chunk_size
        )  # [batch_size, num_layers, num_chunks*chunk_size, num_chunks*chunk_size]

        attentions = (
            attentions
            + torch.eye(chunk_size * num_chunks, device=attentions.device)
            .unsqueeze(0)
            .unsqueeze(0)
            / 2
        )  # add skip connection

        attention_rollout = attentions[:, 0]
        for hidden_layer_idx in range(1, num_layers):
            attention_rollout = attentions[:, hidden_layer_idx] @ attention_rollout

        return label_wise_attention @ attention_rollout
