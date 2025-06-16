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
        print(f"[DEBUG INIT] Config hidden_size: {self.config.hidden_size}")
        print(f"[DEBUG INIT] Config num_hidden_layers: {self.config.num_hidden_layers}")
        print(f"[DEBUG INIT] Config model_type: {getattr(self.config, 'model_type', 'unknown')}")
        print(f"[DEBUG INIT] Config pad_token_id: {getattr(self.config, 'pad_token_id', 'not set')}")
        print(f"[DEBUG INIT] Config bos_token_id: {getattr(self.config, 'bos_token_id', 'not set')}")
        print(f"[DEBUG INIT] Config eos_token_id: {getattr(self.config, 'eos_token_id', 'not set')}")
        print(f"[DEBUG INIT] Config max_position_embeddings: {getattr(self.config, 'max_position_embeddings', 'not set')}")
        print(f"[DEBUG INIT] Config all attributes:")
        for attr in sorted(dir(self.config)):
            if not attr.startswith('_') and not callable(getattr(self.config, attr)):
                try:
                    value = getattr(self.config, attr)
                    print(f"[DEBUG INIT]   {attr}: {value}")
                except:
                    print(f"[DEBUG INIT]   {attr}: <unable to access>")
        
        base_model = AutoModel.from_pretrained(
            model_path, config=self.config
        )
        
        print(f"[DEBUG INIT] Model type: {type(base_model)}")
        print(f"[DEBUG INIT] Model config type: {type(base_model.config)}")
        
        # Check if model loaded correctly
        try:
            sample_param = next(base_model.parameters())
            print(f"[DEBUG INIT] Sample parameter shape: {sample_param.shape}")
            print(f"[DEBUG INIT] Sample parameter dtype: {sample_param.dtype}")
            print(f"[DEBUG INIT] Sample parameter device: {sample_param.device}")
            print(f"[DEBUG INIT] Sample parameter has NaN: {torch.isnan(sample_param).any()}")
            print(f"[DEBUG INIT] Sample parameter has Inf: {torch.isinf(sample_param).any()}")
        except Exception as e:
            print(f"[DEBUG INIT] Error checking model parameters: {e}")

        #if use_peft:
            # LoRA PEFT config for now
        #    peft_config = LoraConfig(
        #        task_type=TaskType.SEQ_CLS,
        #        inference_mode=False,
        #        r=8,
        #        lora_alpha=32,
        #        lora_dropout=0.1,
        #    )
        #    self.roberta_encoder = get_peft_model(base_model, peft_config)
        #else:
        self.roberta_encoder = base_model
        
        # Final check after assignment
        print(f"[DEBUG INIT] Final roberta_encoder type: {type(self.roberta_encoder)}")
        print(f"[DEBUG INIT] Has .encoder attribute: {hasattr(self.roberta_encoder, 'encoder')}")
        print(f"[DEBUG INIT] Has .embeddings attribute: {hasattr(self.roberta_encoder, 'embeddings')}")
        
        if hasattr(self.roberta_encoder, 'embeddings'):
            emb = self.roberta_encoder.embeddings.tok_embeddings.weight
            print(f"[DEBUG INIT] Embeddings shape: {emb.shape}")
            print(f"[DEBUG INIT] Embeddings range: [{torch.min(emb):.6f}, {torch.max(emb):.6f}]")
            print(f"[DEBUG INIT] Embeddings std: {torch.std(emb):.6f}")
            print(f"[DEBUG INIT] Embeddings has NaN: {torch.isnan(emb).any()}")
            print(f"[DEBUG INIT] Embeddings has Inf: {torch.isinf(emb).any()}")
        
        print(f"[DEBUG INIT] Model initialization complete")
        
        # ── DEBUG: Test ModernBERT with minimal input
        print(f"[DEBUG INIT] Testing ModernBERT with minimal input...")
        try:
            with torch.no_grad():
                test_input_ids = torch.tensor([[1, 2, 3, 4, 5]], device='cuda:3')
                test_attention_mask = torch.tensor([[1, 1, 1, 1, 1]], device='cuda:3')
                
                print(f"[DEBUG INIT] Test input shape: {test_input_ids.shape}")
                print(f"[DEBUG INIT] Test input values: {test_input_ids}")
                
                test_outputs = self.roberta_encoder(
                    input_ids=test_input_ids,
                    attention_mask=test_attention_mask,
                    return_dict=False
                )
                
                print(f"[DEBUG INIT] Test output shape: {test_outputs[0].shape}")
                print(f"[DEBUG INIT] Test output min/max: {torch.min(test_outputs[0]):.6f}/{torch.max(test_outputs[0]):.6f}")
                print(f"[DEBUG INIT] Test output has NaN: {torch.isnan(test_outputs[0]).any()}")
                print(f"[DEBUG INIT] Test output has Inf: {torch.isinf(test_outputs[0]).any()}")
                print(f"[DEBUG INIT] ✓ ModernBERT minimal test PASSED")
                
        except Exception as e:
            print(f"[DEBUG INIT] ✗ ModernBERT minimal test FAILED: {e}")
            import traceback
            traceback.print_exc()

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
        embedding = self.roberta_encoder.embeddings(input_ids.view(-1, chunk_size))
        return embedding

    def get_token_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings. Huggingface Roberta model can't return more than 512 token embeddings at once.

        Args:
            input_ids (torch.Tensor): input ids

        Returns:
            torch.Tensor: token embeddings
        """
        sequence_length = input_ids.size(1)
        batch_size = input_ids.size(0)
        if sequence_length <= 512:
            return self.roberta_encoder.embeddings(input_ids)
        input_ids = self.split_input_into_chunks(input_ids, self.pad_token_id)
        chunk_size = input_ids.size(-1)
        chunked_embeddings = self.roberta_encoder.embeddings(
            input_ids.view(-1, chunk_size)
        )
        embeddings = chunked_embeddings.view(
            batch_size, -1, chunked_embeddings.size(-1)
        )
        return embeddings[:, :sequence_length]

    def _call_encoder_safely(self, input_ids, attention_mask):
        """Wrapper to call roberta_encoder with proper parameters for both RoBERTa and ModernBERT"""
        # ── DEBUG: Check inputs to encoder
        print(f"[DEBUG ENCODER] input_ids shape: {input_ids.shape}")
        print(f"[DEBUG ENCODER] input_ids dtype: {input_ids.dtype}")
        print(f"[DEBUG ENCODER] input_ids device: {input_ids.device}")
        print(f"[DEBUG ENCODER] input_ids min/max: {torch.min(input_ids)}/{torch.max(input_ids)}")
        print(f"[DEBUG ENCODER] input_ids sample values: {input_ids[0][:10] if input_ids.shape[1] > 10 else input_ids[0]}")
        print(f"[DEBUG ENCODER] vocab_size: {self.roberta_encoder.config.vocab_size}")
        if attention_mask is not None:
            print(f"[DEBUG ENCODER] attention_mask shape: {attention_mask.shape}")
            print(f"[DEBUG ENCODER] attention_mask dtype: {attention_mask.dtype}")
            print(f"[DEBUG ENCODER] attention_mask device: {attention_mask.device}")
            print(f"[DEBUG ENCODER] attention_mask sum: {torch.sum(attention_mask)}")
            print(f"[DEBUG ENCODER] attention_mask unique values: {torch.unique(attention_mask)}")
            print(f"[DEBUG ENCODER] attention_mask sample: {attention_mask[0][:10] if attention_mask.shape[1] > 10 else attention_mask[0]}")
        else:
            print(f"[DEBUG ENCODER] attention_mask is None")
        
        if torch.isnan(input_ids).any():
            print(f"[DEBUG ENCODER] input_ids contains NaN!")
        if attention_mask is not None and torch.isnan(attention_mask).any():
            print(f"[DEBUG ENCODER] attention_mask contains NaN!")
            
        # Check if input_ids are within vocab range
        if torch.max(input_ids) >= self.roberta_encoder.config.vocab_size:
            print(f"[DEBUG ENCODER] ERROR: input_ids max: {torch.max(input_ids)}, vocab_size: {self.roberta_encoder.config.vocab_size}")
            raise ValueError(f"input_ids contains tokens >= vocab_size")
        
        # Check for negative token IDs
        if torch.min(input_ids) < 0:
            print(f"[DEBUG ENCODER] ERROR: input_ids contains negative values: min={torch.min(input_ids)}")
            raise ValueError(f"input_ids contains negative token IDs")
        
        # Check sequence lengths - ModernBERT might have different limits
        max_seq_len = getattr(self.roberta_encoder.config, 'max_position_embeddings', None)
        if max_seq_len and input_ids.shape[1] > max_seq_len:
            print(f"[DEBUG ENCODER] WARNING: sequence length {input_ids.shape[1]} exceeds max_position_embeddings {max_seq_len}")
        
        # Check for all-padding sequences that might cause issues
        if attention_mask is not None:
            valid_tokens_per_seq = torch.sum(attention_mask, dim=1)
            zero_token_seqs = (valid_tokens_per_seq == 0).sum()
            if zero_token_seqs > 0:
                print(f"[DEBUG ENCODER] WARNING: {zero_token_seqs} sequences have no valid tokens (all padding)")
            
            # Check for sequences that are too short
            min_tokens = torch.min(valid_tokens_per_seq)
            max_tokens = torch.max(valid_tokens_per_seq)
            print(f"[DEBUG ENCODER] Valid tokens per sequence - min: {min_tokens}, max: {max_tokens}")
        
        # Check if we're using the expected token types for ModernBERT
        pad_token_from_config = getattr(self.roberta_encoder.config, 'pad_token_id', None)
        if pad_token_from_config is not None and pad_token_from_config != self.pad_token_id:
            print(f"[DEBUG ENCODER] WARNING: Config pad_token_id ({pad_token_from_config}) != model pad_token_id ({self.pad_token_id})")
        
        # Check for special tokens in input
        special_tokens = torch.unique(input_ids[input_ids < 10])  # Common special token range
        print(f"[DEBUG ENCODER] Special tokens detected in input: {special_tokens.tolist()}")
        
        # Validate attention mask format
        if attention_mask is not None:
            if not torch.all((attention_mask == 0) | (attention_mask == 1)):
                print(f"[DEBUG ENCODER] ERROR: attention_mask contains values other than 0/1")
                print(f"[DEBUG ENCODER] attention_mask unique values: {torch.unique(attention_mask)}")
                raise ValueError("attention_mask must contain only 0s and 1s")
        
        if hasattr(self.roberta_encoder, 'encoder'):
            # RoBERTa style - can use direct call
            print(f"[DEBUG ENCODER] Using RoBERTa-style call")
            outputs = self.roberta_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )
        else:
            # ModernBERT style - needs special handling
            print(f"[DEBUG ENCODER] Using ModernBERT-style call")
            
            # ── DEBUG: Check ModernBERT model state
            print(f"[DEBUG MODERNBERT] Model device: {next(self.roberta_encoder.parameters()).device}")
            print(f"[DEBUG MODERNBERT] Model dtype: {next(self.roberta_encoder.parameters()).dtype}")
            print(f"[DEBUG MODERNBERT] Model training mode: {self.roberta_encoder.training}")
            
            # Check embeddings weights
            emb_weights = self.roberta_encoder.embeddings.tok_embeddings.weight
            print(f"[DEBUG MODERNBERT] Embedding weights shape: {emb_weights.shape}")
            print(f"[DEBUG MODERNBERT] Embedding weights dtype: {emb_weights.dtype}")
            print(f"[DEBUG MODERNBERT] Embedding weights device: {emb_weights.device}")
            print(f"[DEBUG MODERNBERT] Embedding weights min/max: {torch.min(emb_weights):.6f}/{torch.max(emb_weights):.6f}")
            print(f"[DEBUG MODERNBERT] Embedding weights has NaN: {torch.isnan(emb_weights).any()}")
            print(f"[DEBUG MODERNBERT] Embedding weights has Inf: {torch.isinf(emb_weights).any()}")
            
            # Check input dtype and device compatibility
            print(f"[DEBUG MODERNBERT] input_ids dtype: {input_ids.dtype}, device: {input_ids.device}")
            print(f"[DEBUG MODERNBERT] attention_mask dtype: {attention_mask.dtype}, device: {attention_mask.device}")
            
            # Check for extreme values in input that could cause numerical issues
            print(f"[DEBUG MODERNBERT] input_ids unique values count: {torch.unique(input_ids).shape[0]}")
            print(f"[DEBUG MODERNBERT] attention_mask unique values: {torch.unique(attention_mask)}")
            
            # Check model's current precision settings
            print(f"[DEBUG MODERNBERT] Model in training mode: {self.roberta_encoder.training}")
            print(f"[DEBUG MODERNBERT] Mixed precision enabled: {torch.is_autocast_enabled()}")
            
            # Check for any parameters with extreme values
            extreme_params = []
            for name, param in self.roberta_encoder.named_parameters():
                if torch.isinf(param).any() or torch.isnan(param).any():
                    extreme_params.append(name)
                elif param.abs().max() > 1e6 or (param.abs().min() < 1e-6 and param.abs().min() > 0):
                    extreme_params.append(f"{name} (extreme values)")
            
            if extreme_params:
                print(f"[DEBUG MODERNBERT] Parameters with extreme/invalid values: {extreme_params[:5]}")
            else:
                print(f"[DEBUG MODERNBERT] All parameters have reasonable values")
            
            # Test embedding lookup manually first
            print(f"[DEBUG MODERNBERT] Testing embedding lookup...")
            try:
                test_embeddings = self.roberta_encoder.embeddings.tok_embeddings(input_ids)
                print(f"[DEBUG MODERNBERT] Embedding lookup shape: {test_embeddings.shape}")
                print(f"[DEBUG MODERNBERT] Embedding lookup min/max: {torch.min(test_embeddings):.6f}/{torch.max(test_embeddings):.6f}")
                print(f"[DEBUG MODERNBERT] Embedding lookup has NaN: {torch.isnan(test_embeddings).any()}")
                print(f"[DEBUG MODERNBERT] Embedding lookup has Inf: {torch.isinf(test_embeddings).any()}")
            except Exception as e:
                print(f"[DEBUG MODERNBERT] Embedding lookup FAILED: {e}")
            
            # Test with a simple, non-chunked input to see if chunking is the issue
            print(f"[DEBUG MODERNBERT] Testing with simple non-chunked input...")
            try:
                simple_input = torch.tensor([[1, 2, 3, 4, 5]], device='cuda:3', dtype=input_ids.dtype)
                simple_mask = torch.tensor([[1, 1, 1, 1, 1]], device='cuda:3', dtype=attention_mask.dtype)
                
                with torch.cuda.amp.autocast(enabled=False):
                    simple_output = self.roberta_encoder(
                        input_ids=simple_input,
                        attention_mask=simple_mask,
                        return_dict=False,
                    )
                    print(f"[DEBUG MODERNBERT] Simple input test - output shape: {simple_output[0].shape}")
                    print(f"[DEBUG MODERNBERT] Simple input test - has NaN: {torch.isnan(simple_output[0]).any()}")
                    print(f"[DEBUG MODERNBERT] Simple input test - min/max: {torch.min(simple_output[0]):.6f}/{torch.max(simple_output[0]):.6f}")
            except Exception as e:
                print(f"[DEBUG MODERNBERT] Simple input test FAILED: {e}")
            
            # Add hooks to trace NaN propagation through ModernBERT layers
            def create_nan_hook(layer_name):
                def hook_fn(module, input, output):
                    if isinstance(output, torch.Tensor):
                        has_nan = torch.isnan(output).any()
                        print(f"[DEBUG HOOK] {layer_name}: output has NaN = {has_nan}")
                        if has_nan:
                            print(f"[DEBUG HOOK] {layer_name}: output shape = {output.shape}")
                            print(f"[DEBUG HOOK] {layer_name}: output min/max = {torch.min(output)}/{torch.max(output)}")
                    elif isinstance(output, tuple):
                        for i, out in enumerate(output):
                            if isinstance(out, torch.Tensor):
                                has_nan = torch.isnan(out).any()
                                print(f"[DEBUG HOOK] {layer_name}[{i}]: output has NaN = {has_nan}")
                                if has_nan:
                                    print(f"[DEBUG HOOK] {layer_name}[{i}]: output shape = {out.shape}")
                                    print(f"[DEBUG HOOK] {layer_name}[{i}]: output min/max = {torch.min(out)}/{torch.max(out)}")
                return hook_fn
            
            # Register hooks on key ModernBERT layers
            hooks = []
            if hasattr(self.roberta_encoder, 'embeddings'):
                hooks.append(self.roberta_encoder.embeddings.register_forward_hook(create_nan_hook("embeddings")))
            
            if hasattr(self.roberta_encoder, 'layers'):
                for i, layer in enumerate(self.roberta_encoder.layers):
                    hooks.append(layer.register_forward_hook(create_nan_hook(f"layer_{i}")))
                    
                    # Add hooks to attention and MLP components
                    if hasattr(layer, 'attn'):
                        hooks.append(layer.attn.register_forward_hook(create_nan_hook(f"layer_{i}.attn")))
                    if hasattr(layer, 'mlp'):
                        hooks.append(layer.mlp.register_forward_hook(create_nan_hook(f"layer_{i}.mlp")))
                    if hasattr(layer, 'attn_norm'):
                        hooks.append(layer.attn_norm.register_forward_hook(create_nan_hook(f"layer_{i}.attn_norm")))
                    if hasattr(layer, 'mlp_norm'):
                        hooks.append(layer.mlp_norm.register_forward_hook(create_nan_hook(f"layer_{i}.mlp_norm")))
            
            try:
                # Test if mixed precision is causing issues with ModernBERT
                with torch.cuda.amp.autocast(enabled=False):
                    print(f"[DEBUG MODERNBERT] Calling ModernBERT with autocast disabled...")
                    try:
                        outputs = self.roberta_encoder(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=False,
                        )
                        print(f"[DEBUG MODERNBERT] ModernBERT call succeeded")
                    except Exception as e:
                        print(f"[DEBUG MODERNBERT] ModernBERT call FAILED: {e}")
                        raise e
            finally:
                # Remove hooks after the call
                for hook in hooks:
                    hook.remove()
        
        # ── DEBUG: Check encoder outputs
        if torch.isnan(outputs[0]).any():
            print(f"[DEBUG ENCODER] ERROR: encoder outputs contain NaN!")
            print(f"[DEBUG ENCODER] outputs[0] shape: {outputs[0].shape}")
            print(f"[DEBUG ENCODER] outputs[0] min/max: {torch.min(outputs[0])}/{torch.max(outputs[0])}")
        else:
            print(f"[DEBUG ENCODER] encoder outputs OK - no NaN")
            
        return outputs

    def encoder(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        print(f"[DEBUG CHUNKING] Original input_ids shape: {input_ids.shape}")
        print(f"[DEBUG CHUNKING] pad_token_id: {self.pad_token_id}")
        print(f"[DEBUG CHUNKING] chunk_size: {self.chunk_size}")
        
        input_ids = self.split_input_into_chunks(input_ids, self.pad_token_id)
        print(f"[DEBUG CHUNKING] After chunking input_ids shape: {input_ids.shape}")
        
        if attention_masks is not None:
            print(f"[DEBUG CHUNKING] Original attention_masks shape: {attention_masks.shape}")
            attention_masks = self.get_chunked_attention_masks(attention_masks)
            print(f"[DEBUG CHUNKING] After chunking attention_masks shape: {attention_masks.shape}")
            
        batch_size, num_chunks, chunk_size = input_ids.size()
        print(f"[DEBUG CHUNKING] batch_size: {batch_size}, num_chunks: {num_chunks}, chunk_size: {chunk_size}")
        
        # Check for issues in chunked data
        reshaped_input_ids = input_ids.view(-1, chunk_size)
        reshaped_attention_masks = attention_masks.view(-1, chunk_size) if attention_masks is not None else None
        
        print(f"[DEBUG CHUNKING] Reshaped input_ids shape: {reshaped_input_ids.shape}")
        if reshaped_attention_masks is not None:
            print(f"[DEBUG CHUNKING] Reshaped attention_masks shape: {reshaped_attention_masks.shape}")
        
        # Use the safe encoder wrapper
        outputs = self._call_encoder_safely(
            input_ids=reshaped_input_ids,
            attention_mask=reshaped_attention_masks,
        )
        
        final_output = outputs[0].view(batch_size, num_chunks * chunk_size, -1)
        print(f"[DEBUG CHUNKING] Final output shape: {final_output.shape}")
        
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
        
        # ── DEBUG: Check input tensors
        if torch.isnan(input_ids).any():
            print(f"[DEBUG MODEL] input_ids contains NaN")
        if attention_masks is not None and torch.isnan(attention_masks).any():
            print(f"[DEBUG MODEL] attention_masks contains NaN")
        
        hidden_output = self.encoder(input_ids, attention_masks)
        
        # ── DEBUG: Check encoder output
        if torch.isnan(hidden_output).any():
            print(f"[DEBUG MODEL] encoder output contains NaN")
        
        final_output = self.label_wise_attention(
            hidden_output,
            attention_masks=attention_masks,
            output_attention=output_attentions,
            attn_grad_hook_fn=attn_grad_hook_fn,
        )
        
        # ── DEBUG: Check final output
        if isinstance(final_output, tuple):
            if torch.isnan(final_output[0]).any():
                print(f"[DEBUG MODEL] final_output[0] contains NaN")
        else:
            if torch.isnan(final_output).any():
                print(f"[DEBUG MODEL] final_output contains NaN")
        
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
