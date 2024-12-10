import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Optional, Union, Tuple

class BaseExplainer:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, model_helper: Optional[str] = None):
        """
        Initializes the BaseExplainer with a model, tokenizer, and an optional model helper.

        Parameters:
            model (PreTrainedModel): The NLP model to be explained.
            tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
            model_helper (str, optional): Additional helper information or settings for the model.
        """
        self.model = model  # The model to explain
        self.tokenizer = tokenizer  # Tokenizer for text processing
        self.model_helper = model_helper  # Optional helper (e.g., special task settings)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode for consistent gradients

    def _tokenize(self, text: Union[str, Tuple[str, str]]) -> dict:
        """
        Tokenizes the input text and returns a dictionary containing token IDs and attention masks.
        
        Parameters:
            text (str or Tuple): The input text to be tokenized.
            
        Returns:
            dict: A dictionary with token IDs and attention masks.
        """
        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)

    def get_input_embeds(self, text: str) -> torch.Tensor:
        """
        Retrieves input embeddings for the tokenized input text.
        
        Parameters:
            text (str): The input text.
            
        Returns:
            torch.Tensor: The embeddings of the input tokens.
        """
        tokens = self._tokenize(text)
        input_ids = tokens["input_ids"].to(self.device)
        return self.model.embeddings(input_ids)  # Retrieve embeddings from model

    def _check_target(self, target: Union[int, str]) -> int:
        """
        Checks and retrieves the index for the specified target class.

        Parameters:
            target (int or str): Target class as an index or label.

        Returns:
            int: The index of the target class.
        """
        if isinstance(target, int):
            return target
        elif isinstance(target, str):
            return self.model.config.label2id.get(target, 0)  # Default to 0 if not found
        else:
            raise ValueError("Target should be an integer index or a string label.")

    def _check_target_token(self, text: str, target_token: Optional[Union[int, str]]) -> Optional[int]:
        """
        Checks and retrieves the index for a specific target token within the input text, if applicable.

        Parameters:
            text (str): The input text.
            target_token (int or str, optional): The specific token to target.

        Returns:
            int or None: The index of the target token, if specified.
        """
        if target_token is None:
            return None
        tokens = self.tokenizer.tokenize(text)
        if isinstance(target_token, int) and target_token < len(tokens):
            return target_token
        elif isinstance(target_token, str) and target_token in tokens:
            return tokens.index(target_token)
        else:
            raise ValueError("Target token must be a valid token index or token string in the input text.")
