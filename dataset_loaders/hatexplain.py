from typing import List
import numpy as np

NONE_RATIONALE = []

class HateXplain():

    NAME = "HateXplain"
    avg_rationale_size = 7
    # np.mean([sum(self._get_rationale(i, split_type="train")[self._get_ground_truth(i, split_type="train")]) for i in range(self.len("train"))])

    def __init__(self, tokenizer):
        from datasets import load_dataset

        dataset = load_dataset("hatexplain")
        self.train_dataset = dataset["train"]
        self.validation_dataset = dataset["validation"]
        self.test_dataset = dataset["test"]
        self.tokenizer = tokenizer
        self.top_k_hard_rationale = 7
        self.classes = [0, 1, 2]

    def __len__(self):
        # We use the TEST_SET as default
        return self.len()

    def len(self, split_type: str = 'test'):
        if split_type == 'train':
            return len(self.train_dataset)
        elif split_type == 'validation':
            return len(self.validation_dataset)
        elif split_type == 'test':
            return len(self.test_dataset)
        else:
            raise ValueError(
                f"{split_type} not supported as split_type. Specify one among: train, validation or test."
            )

    def _get_item(self, idx: int, split_type: str = 'test'):
        if isinstance(idx, int):
            if split_type == 'train':
                item_idx = self.train_dataset[idx]
            elif split_type == 'validation':
                item_idx = self.validation_dataset[idx]
            elif split_type == 'test':
                item_idx = self.test_dataset[idx]
            else:
                raise ValueError(
                    f"{split_type} not supported as split_type. Specify one among:  train, validation or test."
                )
            return item_idx
        elif isinstance(idx, dict):
            return idx
        else:
            raise ValueError()

    def __getitem__(self, idx):
        # We use the TEST_SET as default
        return self.get_instance(idx)

    def get_instance(self, idx, split_type: str = 'test', rationale_union=True):
        item_idx = self._get_item(idx, split_type)
        text = self._get_text(item_idx)
        tokens = (
            [self.tokenizer.cls_token]
            + self.tokenizer.tokenize(text)
            + [self.tokenizer.sep_token]
        )
        rationale = self._get_rationale(item_idx, split_type, rationale_union)
        true_label = self._get_ground_truth(item_idx, split_type)
        return {
            "text": text,
            "tokens": tokens,
            "rationale": rationale,
            "label": true_label,
        }

    def _get_text(self, idx, split_type: str = 'test'):
        item_idx = self._get_item(idx, split_type)
        post_tokens = item_idx["post_tokens"]
        text = " ".join(post_tokens)
        return text

    def _get_rationale(self, idx, split_type: str = 'test', rationale_union=True):
        item_idx = self._get_item(idx, split_type)
        word_based_tokens = item_idx["post_tokens"]

        # All hatexplain rationales are defined for the label, only for hatespeech or offensive classes
        rationale_label = self._get_ground_truth(idx, split_type)        # Initialize rationale_by_label with placeholders
        rationale_by_label = [NONE_RATIONALE for _ in self.classes]

        if "rationales" in item_idx:
            rationales = item_idx["rationales"]

            # If rationales are a list of lists
            if len(rationales) > 0 and isinstance(rationales[0], list):
                if rationale_union:
                    # If rationale_union is True, combine all rationales into a single 1D array
                    rationale = [any(each) for each in zip(*rationales)]  # Perform the union
                    rationale = [int(each) for each in rationale]  # Convert True/False to 1/0
                else:
                    # If rationale_union is False, we return all the individual rationales in a list (deprecated)
                    rationale_by_label[rationale_label] = [
                        self.get_true_rationale_from_words_to_tokens(word_based_tokens, rationale)
                        for rationale in rationales
                    ]
                    return rationale_by_label
            else:
                # If rationales are just a single list (not a list of lists), directly use it
                rationale = rationales

            # Get the final rationale (converted from words to tokens)
            rationale_by_label[rationale_label] = self.get_true_rationale_from_words_to_tokens(word_based_tokens, rationale)
            print(f"rationale label {rationale_by_label}")
        # Here we ensure the output is a single 1D array
        if rationale_union:
            # If rationale_union is True, return the single unified rationale as a 1D array
            # Filter out empty lists before using zip
            non_empty_rationale_by_label = [r for r in rationale_by_label if r]  # Remove empty lists
            if non_empty_rationale_by_label:
                final_rationale = [int(any(each)) for each in zip(*non_empty_rationale_by_label)]  # Union of all rationales
            else:
                final_rationale = []  # If no valid rationale exists, return an empty list
            print(f"final rationale {final_rationale}")
            return final_rationale
        else:
            # Otherwise, return the rationale for the specific label (may be a list of lists if rationale_union is False)
            return rationale_by_label
    
    def _get_ground_truth(self, idx, split_type: str = 'test'):
        item_idx = self._get_item(idx, split_type)
        labels = item_idx["annotators"]["label"]
        # Label by majority voting
        return max(set(labels), key=labels.count)

    def get_true_rationale_from_words_to_tokens(
        self, word_based_tokens: List[str], words_based_rationales: List[int]
    ) -> List[int]:
        token_rationale = []
        for t, rationale_t in zip(word_based_tokens, words_based_rationales):
            converted_token = self.tokenizer.encode(t)[1:-1]

            for token_i in converted_token:
                token_rationale.append(rationale_t)
        # token_rationale = [0] + token_rationale + [0]  # Add rationale for [CLS] and [SEP] (set to 0)
    
        return token_rationale