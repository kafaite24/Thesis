import torch
from captum.attr import IntegratedGradients
from typing import Optional, Tuple, Union
from .baseexplainer import BaseExplainer
from .explanation import Explanation

class IntegratedGradientsExplainer(BaseExplainer):
    NAME = "Integrated Gradients"

    def __init__(
        self,
        model,
        tokenizer,
        model_helper: Optional[str] = None,
        multiply_by_inputs: bool = True,
        device: str = "cpu",
    ):
        """
        Initializes the IntegratedGradientsExplainer with the specified model, tokenizer, and optional model helper.
        """
        super().__init__(model, tokenizer, model_helper)
        self.device = device
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode
        self.multiply_by_inputs = multiply_by_inputs  # Store this value for later use

    def _generate_baselines(self, input_len):
        """
        Generates baselines using [CLS], [SEP], and [PAD] tokens.
        The baseline is typically the reference input that we compare the actual input against.
        """
        # Generate a sequence of token IDs for the baseline: [CLS] + [PAD] tokens + [SEP]
        ids = (
            [self.tokenizer.cls_token_id]
            + [self.tokenizer.pad_token_id] * (input_len - 2)
            + [self.tokenizer.sep_token_id]
        )
        
        # Convert the IDs into input embeddings
        # input_ids_tensor = torch.tensor(ids, device=self.device).unsqueeze(0)  # Add batch dimension
        # embeddings = self.model.bert.embeddings(input_ids_tensor)

        # Convert the IDs into input embeddings
        input_ids_tensor = torch.tensor(ids, device=self.device).unsqueeze(0)  # Add batch dimension

        # Dynamically access input embeddings for the model
        embedding_layer = self.model.get_input_embeddings()
        embeddings = embedding_layer(input_ids_tensor)
        
        return embeddings

    def compute_feature_importance(
        self,
        text: Union[str, Tuple[str, str]],
        target: Union[int, str] = 1,
        target_token: Optional[Union[int, str]] = None,
        **kwargs,
    ):
        """
        Computes feature importance using Integrated Gradients for the given text input.

        Parameters:
            text (str or Tuple): The input text for which feature importance is computed.
            target (int or str): The target class for which the gradient is calculated.
            target_token (int or str, optional): Specific token within the text for token-level classification.
            **kwargs: Additional arguments for the attribute method.

        Returns:
            Explanation: Object containing tokens, scores, and metadata.
        """
        # Define the forward function to calculate logits with respect to input embeddings
        def func(input_embeds):
            outputs = self.model(inputs_embeds=input_embeds, attention_mask=item["attention_mask"])
            logits = outputs.logits
            return logits

        # Check and process target class and token
        target_pos_idx = self._check_target(target)
        target_token_pos_idx = self._check_target_token(text, target_token)

        # Tokenize the input text and move to the device
        item = self._tokenize(text)
        # item = {k: v.to(self.device) for k, v in item.items()}
        input_len = item["attention_mask"].sum().item()

        # Initialize Integrated Gradients as the attribution method
        ig = IntegratedGradients(func, multiply_by_inputs=self.multiply_by_inputs)

        # Retrieve input embeddings for the text
        inputs = self.get_input_embeds(text)

        # Generate baseline embeddings
        baselines = self._generate_baselines(input_len)
        print(f"target pos idx {target_pos_idx}")
        # Compute Integrated Gradients attributions
        attr = ig.attribute(inputs, baselines=baselines, target=target_pos_idx, **kwargs)
        attr = attr[0, :input_len, :].detach().cpu()

        # Sum over the embedding dimensions to get a single score per token
        attr = attr.sum(-1).numpy()

        # Create and return an Explanation object with tokens, scores, and additional metadata
        output = Explanation(
            text=text,
            tokens=self.get_tokens(text),
            scores=attr,
            explainer=self.NAME,
            target_pos_idx=target_pos_idx,
            target_token_pos_idx=target_token_pos_idx,
            target=target_pos_idx  # self.model.config.id2label[target_pos_idx]
        )
        print(output)
        return output

    def _tokenize(self, text):
        """
        Tokenizes the input text and returns token IDs and attention masks.
        """
        encoding = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        return encoding

    def get_input_embeds(self, text):
        """
        Retrieves input embeddings for the tokenized input text.
        """
        encoding = self._tokenize(text)
        input_ids = encoding["input_ids"].to(self.device)
        
        # Access embeddings through the `bert` attribute in the model
        # inputs_embeds = self.model.bert.embeddings(input_ids)
        embedding_layer = self.model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids)
        return inputs_embeds

    def get_tokens(self, text):
        """
        Converts token IDs to tokens for easy interpretation.
        """
        encoding = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        return tokens
