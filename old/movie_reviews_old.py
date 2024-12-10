
# datasets/movie_reviews.py

from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import List, Dict, Tuple, Any
import numpy as np

NONE_RATIONALE = []  

class MovieReviews:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        Initialize the dataset and tokenizer.
        """
        self.dataset = load_dataset("movie_rationales")
        self.tokenizer = tokenizer
        self.classes = [0, 1]
        
        # Set up train, validation, and test splits
        self.train_data = self.dataset["train"]
        self.validation_data = self.dataset["validation"]
        self.test_data = self.dataset["test"]
    
    def get_data_length(self, split_type: str = "test") -> int:
            """
            Returns the number of instances in the specified split.
            """
            data_split = getattr(self, f"{split_type}_data", None)
            if data_split is None:
                raise ValueError(f"{split_type} not a valid dataset split.")
            return len(data_split)
    
    def get_review(self, idx: int, split_type: str = "test", combine_rationales: bool = True) -> Dict[str, Any]:
        """
        Retrieves a single review instance with tokens, rationale, and label.
        """
        item = self._get_item(idx, split_type)
        review_text = item["review"].replace("\n", " ")
        tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(review_text) + [self.tokenizer.sep_token]
        
        # Create rationale mask and retrieve label
        rationale_mask = self._create_rationale_mask(review_text, item.get("evidences", []), combine_rationales)
        label = item["label"]
        
        return {
            "text": review_text,
            "tokens": tokens,
            "rationale": rationale_mask,
            "label": label,
        }
    
    def _get_item(self, idx: int, split_type: str = "test") -> Dict:
        """
        Retrieve an item by index from the specified dataset split.
        """
        data_split = getattr(self, f"{split_type}_data", None)
        if data_split is None:
            raise ValueError(f"{split_type} not a valid dataset split.")
        return data_split[idx]
    
    def _create_rationale_mask(self, text: str, text_rationales: List[str], combine_rationales: bool) -> List[int]:
        """
        Converts rationale text spans to a one-hot mask for tokens.
        """
        encoded = self.tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=True)
        offsets = encoded["offset_mapping"]
        mask = np.zeros(len(offsets), dtype=int)
        
        rationale_offsets = self._get_rationale_offsets(text, text_rationales)
        if combine_rationales:
            combined_offsets = [offset for spans in rationale_offsets for offset in spans]
            mask = self._mark_rationale_tokens(offsets, combined_offsets)
        else:
            for span in rationale_offsets:
                mask = self._mark_rationale_tokens(offsets, span)
                
        return mask.tolist()
    
    def _get_rationale_offsets(self, text: str, text_rationales: List[str]) -> List[List[Tuple[int, int]]]:
        """
        Finds the start and end offsets of rationale spans within the text.
        """
        rationale_offsets = []
        for rationale_text in text_rationales:
            start_idx = text.find(rationale_text)
            if start_idx != -1:
                end_idx = start_idx + len(rationale_text)
                encoded_rationale = self.tokenizer.encode_plus(
                    text[start_idx:end_idx], return_offsets_mapping=True, add_special_tokens=False
                )
                offsets = [(start + start_idx, end + start_idx) for start, end in encoded_rationale["offset_mapping"]]
                rationale_offsets.append(offsets)
            else:
                print(f"Warning: Rationale '{rationale_text}' not found in text.")
        return rationale_offsets
    
    def _mark_rationale_tokens(self, token_offsets: List[Tuple[int, int]], rationale_offsets: List[Tuple[int, int]]) -> np.ndarray:
        """
        Marks tokens within the rationale spans as 1 in the mask.
        """
        mask = np.zeros(len(token_offsets), dtype=int)
        for token_idx, (token_start, token_end) in enumerate(token_offsets):
            for rationale_start, rationale_end in rationale_offsets:
                if token_start >= rationale_start and token_end <= rationale_end:
                    mask[token_idx] = 1
                    break
        return mask