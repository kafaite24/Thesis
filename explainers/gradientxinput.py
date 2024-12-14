# import torch
# from captum.attr import InputXGradient
# from typing import Optional, Tuple, Union
# from .baseexplainer import BaseExplainer
# from .explanation import Explanation

# class InputXGradientExplainer(BaseExplainer):
#     NAME = "GradientxInput"

#     def __init__(
#         self,
#         model,
#         tokenizer,
#         model_helper: Optional[str] = None,
#         multiply_by_inputs: bool = True,
#         device: str = "cpu",
#     ):
#         """
#         Initializes the GradientExplainer with the specified model, tokenizer, and optional model helper.
#         """
#         super().__init__(model, tokenizer, model_helper)
#         self.device = device
#         self.model.to(self.device)
#         self.model.eval()  # Set model to evaluation mode
#         self.multiply_by_inputs = multiply_by_inputs  # Store this value for later use

#     def compute_feature_importance(
#         self,
#         text: Union[str, Tuple[str, str]],
#         target: Union[int, str] = 1,
#         target_token: Optional[Union[int, str]] = None,
#         **kwargs,
#     ):
#         """
#         Computes feature importance using Gradient x Input for the given text input.

#         Parameters:
#             text (str or Tuple): The input text for which feature importance is computed.
#             target (int or str): The target class for which the gradient is calculated.
#             target_token (int or str, optional): Specific token within the text for token-level classification.
#             **kwargs: Additional arguments for attribute method.

#         Returns:
#             Explanation: Object containing tokens, scores, and metadata.
#         """
#         # Define the forward function to calculate logits with respect to input embeddings
#         def func(input_embeds):
#             outputs = self.model(inputs_embeds=input_embeds, attention_mask=item["attention_mask"])
#             logits = outputs.logits
#             return logits

#         # Check and process target class and token
#         target_pos_idx = self._check_target(target)
#         target_token_pos_idx = self._check_target_token(text, target_token)

#         # Tokenize the input text and move to the device
#         item = self._tokenize(text)
#         item = {k: v.to(self.device) for k, v in item.items()}
#         input_len = item["attention_mask"].sum().item()

#         # Initialize Gradient x Input as the attribution method
#         dl = InputXGradient(func)

#         # Retrieve input embeddings for the text
#         inputs = self.get_input_embeds(text)

#         # Compute Gradient x Input attributions
#         attr = dl.attribute(inputs, target=target_pos_idx, **kwargs)
#         attr = attr[0, :input_len, :].detach().cpu()

#         # Sum over the embedding dimensions to get a single score per token
#         attr = attr.sum(-1).numpy()

#         # Create and return an Explanation object with tokens, scores, and additional metadata
#         output = Explanation(
#             text=text,
#             tokens=self.get_tokens(text),
#             scores=attr,
#             explainer=self.NAME,
#             target_pos_idx=target_pos_idx,
#             target_token_pos_idx=target_token_pos_idx,
#             target=self.model.config.id2label[target_pos_idx],
#             target_token=self.tokenizer.decode(
#                 item["input_ids"][0, target_token_pos_idx].item()
#             )
#             if target_token_pos_idx is not None
#             else None,
#         )
#         return output

#     def _tokenize(self, text):
#         """
#         Tokenizes the input text and returns token IDs and attention masks.
#         """
#         encoding = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#         return encoding

#     def get_input_embeds(self, text):
#         """
#         Retrieves input embeddings for the tokenized input text.
#         """
#         encoding = self._tokenize(text)
#         input_ids = encoding["input_ids"].to(self.device)
        
#         # Access embeddings through the `bert` attribute in the model
#         inputs_embeds = self.model.bert.embeddings(input_ids)
#         return inputs_embeds
    

#     def get_tokens(self, text):
#         """
#         Converts token IDs to tokens for easy interpretation.
#         """
#         encoding = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#         tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
#         return tokens
    
   

import torch
from captum.attr import InputXGradient, Saliency
from typing import Optional, Tuple, Union
from .baseexplainer import BaseExplainer
from .explanation import Explanation

class InputXGradientExplainer(BaseExplainer):
    NAME = "GradientxInput"

    def __init__(
        self,
        model,
        tokenizer,
        model_helper: Optional[str] = None,
        multiply_by_inputs: bool = True,
        **kwargs,
    ):
        super().__init__(model, tokenizer, model_helper, **kwargs)

        self.multiply_by_inputs = multiply_by_inputs
        if self.multiply_by_inputs:
            self.NAME += " (x Input)"

    def compute_feature_importance(
        self,
        text: Union[str, Tuple[str, str]],
        target: Union[int, str] = 1,
        target_token: Optional[Union[int, str]] = None,
        **kwargs,
    ):
        def func(input_embeds):
            outputs = self.helper.model(
                inputs_embeds=input_embeds, attention_mask=item["attention_mask"]
            )
            logits = self.helper._postprocess_logits(
                outputs.logits, target_token_pos_idx=target_token_pos_idx
            )
            return logits

        # Sanity checks
        # TODO these checks have already been conducted if used within the benchmark class. Remove them here if possible.
        target_pos_idx = self.helper._check_target(target)
        target_token_pos_idx = self.helper._check_target_token(text, target_token)
        text = self.helper._check_sample(text)

        item = self._tokenize(text)
        item = {k: v.to(self.device) for k, v in item.items()}
        input_len = item["attention_mask"].sum().item()
        dl = (
            InputXGradient(func, **self.init_args)
            if self.multiply_by_inputs
            else Saliency(func, **self.init_args)
        )

        inputs = self.get_input_embeds(text)

        attr = dl.attribute(inputs, target=target_pos_idx, **kwargs)
        attr = attr[0, :input_len, :].detach().cpu()

        # pool over hidden size
        attr = attr.sum(-1).numpy()

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
