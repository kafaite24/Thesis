from baseexplainer import BaseExplainer
from typing import Optional, Tuple, Union
import torch
from captum.attr import DeepLift
from explanation import Explanation
import torch.nn as nn


class DeepLiftExplainer(BaseExplainer):
    NAME = "DeepLift"

    def __init__(
        self,
        model,
        tokenizer,
        model_helper: Optional[str] = None,
        multiply_by_inputs: bool = False,
        device="cpu",
        **kwargs,
    ):
        super().__init__(model, tokenizer, model_helper, **kwargs)
        self.multiply_by_inputs = multiply_by_inputs
        if self.multiply_by_inputs:
            self.NAME += " (x Input)"

    class ForwardWrapper(nn.Module):
        """
        Wraps a custom forward function into an nn.Module.
        """

        def __init__(self, model, attention_mask):
            super().__init__()
            self.model = model
            self.attention_mask = attention_mask

        def forward(self, inputs_embeds):
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=self.attention_mask)
            return outputs.logits
        
    def _generate_baselines(self, input_len):
        """
        Generate baseline embeddings for the input.
        The baseline represents a neutral reference input, such as [CLS], [PAD], [SEP].
        """
        ids = (
            [self.helper.tokenizer.cls_token_id]
            + [self.helper.tokenizer.pad_token_id] * (input_len - 2)
            + [self.helper.tokenizer.sep_token_id]
        )
        embeddings = self.helper._get_input_embeds_from_ids(
            torch.tensor(ids, device=self.device)
        )
        print(f"embeddings {embeddings}")
        return embeddings.unsqueeze(0)

    
    def compute_feature_importance(
        self,
        text: Union[str, Tuple[str, str]],
        target: Union[int, str] = 1,
        target_token: Optional[Union[int, str]] = None,
        show_progress: bool = False,
        **kwargs,
    ):
        """
        Compute feature importance using DeepLift.

        Args:
            text (Union[str, Tuple[str, str]]): Input text for which to compute importance.
            target (Union[int, str]): Target class or label.
            target_token (Optional[Union[int, str]]): Target token index for token-level tasks.
            show_progress (bool): Whether to show progress.
            **kwargs: Additional arguments passed to DeepLift.

        Returns:
            Explanation: An object containing tokens, scores, and metadata.
        """
        print(f"computing feature importance")
        # Sanity checks
        target_pos_idx = self.helper._check_target(target)
        target_token_pos_idx = self.helper._check_target_token(text, target_token)
        text = self.helper._check_sample(text)
        # Tokenize the input and get embeddings
        item = self._tokenize(text)
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        input_len = item["attention_mask"].sum().item()

        # Get input embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # Wrap the forward function into a PyTorch nn.Module
        forward_model = self.ForwardWrapper(self.model, attention_mask)

         # Initialize DeepLift with the custom forward function
        dl = DeepLift(forward_model,multiply_by_inputs=self.multiply_by_inputs)
        
        # dl = DeepLift(self.model, multiply_by_inputs=self.multiply_by_inputs)
        inputs = self.get_input_embeds(text)
        baselines = self._generate_baselines(input_len)

        # Forward pass for the actual input
        def forward_pass(model, inputs_embeds, attention_mask):
            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            return outputs.logits


        # Forward pass for actual input and baseline
        actual_logits = forward_pass(self.model, inputs, attention_mask)
        baseline_logits = forward_pass(self.model, baselines, attention_mask)

        print(f"actual logits {actual_logits}")
        print(f"baseline logits {baseline_logits}")
        # Compute attributions
        attr = dl.attribute(inputs, baselines=baselines, target=target_pos_idx, **kwargs)

        attr = attr[0, :input_len, :].detach().cpu()

        # Pool over hidden size to get a single score per token
        attr = attr.sum(-1).numpy()

        # Create Explanation object
        output = Explanation(
            text=text,
            tokens=self.get_tokens(text),
            scores=attr,
            explainer=self.NAME,
            helper_type=self.helper.HELPER_TYPE,
            target_pos_idx=target_pos_idx,
            target_token_pos_idx=target_token_pos_idx,
            target=self.helper.model.config.id2label[target_pos_idx],
            target_token=self.helper.tokenizer.decode(
                item["input_ids"][0, target_token_pos_idx].item()
            )
            if self.helper.HELPER_TYPE == "token-classification"
            else None,
        )
        return output
