import numpy as np
import torch

class FaithfulnessCorrelationEvaluator:
    def __init__(self, model, tokenizer, baseline_token="[PAD]"):
        """
        Initializes the FaithfulnessEvaluator.

        Args:
            model: The trained classifier (e.g., BERT).
            tokenizer: Tokenizer corresponding to the model.
            baseline_token: The token to use as a baseline when masking features.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.baseline_token = baseline_token

    def mask_token(self, tokens, index):
        """
        Masks a specific token in the tokens list by replacing it with the baseline token.

        Args:
            tokens (list): List of tokens.
            index (int): Index of the token to mask.

        Returns:
            list: List of tokens with the specified token masked.
        """
        masked_tokens = tokens.copy()
        masked_tokens[index] = self.baseline_token
        return masked_tokens

    def evaluate_instance(self, tokens, saliency_scores):
        """
        Evaluates the faithfulness for a single instance.

        Args:
            tokens (list): List of tokens for the instance.
            saliency_scores (list): Saliency scores corresponding to each token.

        Returns:
            tuple: Saliency scores and observed effects (change in predicted probabilities).
        """
        # Tokenize the original input and predict probabilities
        original_text = self.tokenizer.convert_tokens_to_string(tokens)
        inputs = self.tokenizer(original_text, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}

        with torch.no_grad():
            original_logits = self.model(**inputs).logits
            original_probs = torch.softmax(original_logits, dim=1).cpu().numpy()

        # Predicted class
        pred_class = np.argmax(original_probs)

        # Sort tokens by saliency importance
        sorted_indices = np.argsort(-np.array(saliency_scores))  # Descending order of importance
        pred_probs = []

        for idx in sorted_indices:
            # Mask the token at the current index
            masked_tokens = self.mask_token(tokens, idx)

            # Tokenize the masked input and predict probabilities
            masked_text = self.tokenizer.convert_tokens_to_string(masked_tokens)
            inputs_masked = self.tokenizer(masked_text, return_tensors="pt", truncation=True, padding=True)
            inputs_masked = {key: val.to(self.model.device) for key, val in inputs_masked.items()}

            with torch.no_grad():
                masked_logits = self.model(**inputs_masked).logits
                masked_probs = torch.softmax(masked_logits, dim=1).cpu().numpy()

            # Store the probability of the predicted class
            pred_probs.append(masked_probs[0][pred_class])

        return saliency_scores, pred_probs

    def compute_faithfulness_score(self, precomputed_scores):
        """
        Computes the faithfulness score for the dataset.

        Args:
            precomputed_scores (list): Precomputed saliency scores for each instance.

        Returns:
            float: The faithfulness correlation score for the dataset.
        """
        all_saliency_importances = []
        all_observed_effects = []

        for data in precomputed_scores:
            tokens = data["tokens"]
            saliency_scores = data["saliency_scores"]

            saliency_importances, observed_effects = self.evaluate_instance(tokens, saliency_scores)
            all_saliency_importances.extend(saliency_importances)
            all_observed_effects.extend(observed_effects)

        # Calculate the correlation coefficient
        correlation = np.corrcoef(all_saliency_importances, all_observed_effects)[0, 1]
        return correlation
