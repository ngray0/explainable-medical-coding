from typing import Optional, Callable

import torch
import torch.nn as nn


class LabelAttention(nn.Module):
    def __init__(self, input_size: int, projection_size: int, num_classes: int):
        super().__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)
        self.second_linear = nn.Linear(projection_size, num_classes, bias=False)
        self.third_linear = nn.Linear(input_size, num_classes)
        self._init_weights(mean=0.0, std=0.03)

    def forward(
        self,
        x: torch.Tensor,
        output_attention: bool = False,
        attention_masks: Optional[torch.Tensor] = None,
        attn_grad_hook_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """LAAT attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        weights = torch.tanh(self.first_linear(x))  # [batch_size, seq_len, proj_size]
        att_weights = self.second_linear(weights)  # [batch_size, seq_len, num_classes]
        att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1, 2)
        weighted_output = att_weights @ x  # [batch_size, num_classes, input_size]
        output = (
            self.third_linear.weight.mul(weighted_output)
            .sum(dim=2)
            .add(self.third_linear.bias)
        )  # [batch_size, num_classes]
        if output_attention:
            return output, att_weights
        return output

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        torch.nn.init.normal_(self.first_linear.weight, mean, std)
        torch.nn.init.normal_(self.second_linear.weight, mean, std)
        torch.nn.init.normal_(self.third_linear.weight, mean, std)


class LabelCrossAttention(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        num_classes: int, 
        scale: float = 1.0,
        init_with_descriptions: bool = False,
        model_path: str = None,
        target_tokenizer = None,
        icd_version: int = 10,
        freeze_label_embeddings: bool = False
    ):
        super().__init__()
        self.weights_k = nn.Linear(input_size, input_size, bias=False)
        self.label_representations = torch.nn.Parameter(
            torch.rand(num_classes, input_size), requires_grad=True
        )
        self.weights_v = nn.Linear(input_size, input_size)
        self.output_linear = nn.Linear(input_size, 1)
        self.layernorm = nn.LayerNorm(input_size)
        self.num_classes = num_classes
        self.scale = scale
        
        # Initialize weights
        if init_with_descriptions and model_path and target_tokenizer is not None:
            self._init_weights_description_embeddings(model_path, target_tokenizer, icd_version)
        else:
            self._init_weights(mean=0.0, std=0.03)
        
        # Freeze label embeddings if requested
        if freeze_label_embeddings:
            self.freeze_label_embeddings()

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        output_attention: bool = False,
        attn_grad_hook_fn: Optional[Callable] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Label Cross Attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """

        V = self.weights_v(x)
        K = self.weights_k(x)
        Q = self.label_representations

        att_weights = Q.matmul(K.transpose(1, 2))

        # replace nan with max value of float16
        # att_weights = torch.where(
        #     torch.isnan(att_weights), torch.tensor(30000), att_weights
        # )
        if attention_masks is not None:
            # pad attention masks with 0 such that it has the same sequence lenght as x
            attention_masks = torch.nn.functional.pad(
                attention_masks, (0, x.size(1) - attention_masks.size(1)), value=0
            )
            attention_masks = attention_masks.to(torch.bool)
            # repeat attention masks for each class
            attention_masks = attention_masks.unsqueeze(1).repeat(
                1, self.num_classes, 1
            )
            attention_masks = attention_masks.masked_fill_(
                attention_masks.logical_not(), float("-inf")
            )
            att_weights += attention_masks

        attention = torch.softmax(
            att_weights / self.scale, dim=2
        )  # [batch_size, num_classes, seq_len]
        if attn_grad_hook_fn is not None:
            attention.register_hook(attn_grad_hook_fn)

        y = attention @ V  # [batch_size, num_classes, input_size]

        y = self.layernorm(y)

        output = (
            self.output_linear.weight.mul(y)
            .sum(dim=2)
            .add(self.output_linear.bias)  # [batch_size, num_classes]
        )

        if output_attention:
            return output, att_weights

        return output

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        self.weights_k.weight = torch.nn.init.normal_(self.weights_k.weight, mean, std)
        self.weights_v.weight = torch.nn.init.normal_(self.weights_v.weight, mean, std)
        self.label_representations = torch.nn.init.normal_(
            self.label_representations, mean, std
        )
        self.output_linear.weight = torch.nn.init.normal_(
            self.output_linear.weight, mean, std
        )

    def _init_weights_description_embeddings(
        self, 
        model_path: str, 
        target_tokenizer,
        icd_version: int = 10,
        batch_size: int = 64
    ) -> None:
        """Initialize label representations with ICD code description embeddings.

        Args:
            model_path (str): Path to the transformer model for generating embeddings
            target_tokenizer: TargetTokenizer that maps indices to ICD codes
            icd_version (int): ICD version (9 or 10). Defaults to 10.
            batch_size (int): Batch size for processing descriptions. Defaults to 64.
        """
        from transformers import AutoModel, AutoTokenizer
        from rich.progress import track
        from explainable_medical_coding.utils.data_helper_functions import get_code2description_mimiciv_combined
        
        # Load ICD descriptions
        code2desc = get_code2description_mimiciv_combined(icd_version)
        
        # Load embedding model
        embed_model = AutoModel.from_pretrained(model_path)
        embed_tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Collect all descriptions in target_tokenizer order
        descriptions = []
        for i in range(len(target_tokenizer)):
            icd_code = target_tokenizer.id2target[i]
            descriptions.append(code2desc[icd_code])
        
        # Process descriptions in batches with progress bar
        all_embeddings = []
        batch_ranges = range(0, len(descriptions), batch_size)
        
        for i in track(batch_ranges, description="Generating description embeddings..."):
            batch_descriptions = descriptions[i:i+batch_size]
            
            # Batch tokenization
            tokens = embed_tokenizer(
                batch_descriptions,
                return_tensors="pt", 
                truncation=True, 
                max_length=128,
                padding=True
            )
            
            # Batch forward pass
            with torch.no_grad():
                outputs = embed_model(**tokens)
                # Mean pooling with attention mask to handle padding
                attention_mask = tokens['attention_mask'].unsqueeze(-1)
                masked_embeddings = outputs.last_hidden_state * attention_mask
                embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
                all_embeddings.append(embeddings)
        
        # Combine all embeddings and replace parameter data
        final_embeddings = torch.cat(all_embeddings, dim=0)
        self.label_representations.data.copy_(final_embeddings)
        
        # Clean up the temporary model to free memory
        del embed_model
        del embed_tokenizer

    def freeze_label_embeddings(self) -> None:
        """Freeze the label representation parameters."""
        self.label_representations.requires_grad = False
        
    def unfreeze_label_embeddings(self) -> None:
        """Unfreeze the label representation parameters."""
        self.label_representations.requires_grad = True


class InputMasker(nn.Module):
    def __init__(self, input_size: int, scale: float = 1.0):
        super().__init__()
        self.weights_k = nn.Linear(input_size, input_size)
        self.weights_q = nn.Linear(input_size, input_size)
        self.weights_v = nn.Linear(input_size, input_size)
        self.output_linear = nn.Linear(input_size, 1)
        self.layernorm = nn.LayerNorm(input_size)
        self.scale = scale
        self._init_weights(mean=0.0, std=0.03)

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        output_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Label Cross Attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """

        V = self.weights_v(x)
        K = self.weights_k(x)
        Q = self.weights_q(x)

        att_weights = Q.matmul(K.transpose(1, 2))

        # replace nan with max value of float16
        # att_weights = torch.where(
        #     torch.isnan(att_weights), torch.tensor(30000), att_weights
        # )
        if attention_masks is not None:
            # pad attention masks with 0 such that it has the same sequence lenght as x
            attention_masks = torch.nn.functional.pad(
                attention_masks, (0, x.size(1) - attention_masks.size(1)), value=0
            )
            attention_masks = attention_masks.to(torch.bool)

            # repeat attention masks for each token
            attention_masks = attention_masks.unsqueeze(1).repeat(1, x.size(1), 1)

            attention_masks = attention_masks.masked_fill_(
                attention_masks.logical_not(), float("-inf")
            )
            att_weights += attention_masks

        attention = torch.softmax(
            att_weights / self.scale, dim=2
        )  # [batch_size, num_classes, seq_len]

        y = attention @ V  # [batch_size, num_classes, input_size]

        y = self.layernorm(y)

        output = (
            self.output_linear.weight.mul(y)
            .sum(dim=2)
            .add(self.output_linear.bias)  # [batch_size, num_classes]
        )

        if output_attention:
            return output, att_weights

        return output

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        self.weights_k.weight = torch.nn.init.normal_(self.weights_k.weight, mean, std)
        self.weights_v.weight = torch.nn.init.normal_(self.weights_v.weight, mean, std)
        self.weights_q.weight = torch.nn.init.normal_(self.weights_q.weight, mean, std)
        self.output_linear.weight = torch.nn.init.normal_(
            self.output_linear.weight, mean, std
        )
