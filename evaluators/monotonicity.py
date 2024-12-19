import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class MonotonicityEvaluator:
    def __init__(self, model, tokenizer, baseline_token="[MASK]", batch_size=16):
        """
        Initializes the MonotonicityEvaluator.

        Args:
            model: Pre-trained classification model (e.g., BERT for sentiment analysis).
            tokenizer: Tokenizer corresponding to the model.
            baseline_token: Token used to mask input tokens.
            batch_size: Number of texts to process in one batch.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.baseline_token = baseline_token
        self.batch_size = batch_size

    def mask_all_tokens(self, tokens):
        """
        Replace all tokens with the baseline token.

        Args:
            tokens (list): List of tokens in the input text.

        Returns:
            list: List of tokens with all tokens replaced by the baseline token.
        """
        return [self.baseline_token] * len(tokens)

    def compute_confidences(self, texts):
        """
        Compute model confidences for a batch of texts.

        Args:
            texts (list of str): List of input texts.

        Returns:
            numpy.ndarray: Array of confidences for the predicted class.
        """
        all_confidences = []
        for i in range(0, len(texts), self.batch_size):
          
            batch_texts = texts[i:i + self.batch_size]
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512
            )
            inputs = {key: val.to(self.model.device) for key, val in inputs.items()}
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            pred_classes = np.argmax(probs, axis=1)
            confidences = probs[np.arange(len(pred_classes)), pred_classes]
            all_confidences.extend(confidences)

        return np.array(all_confidences)

    def monotonicity_metric(self, tokens, saliency_scores, label):
        """
        Computes the monotonicity metric for a single instance.

        Args:
            tokens (list): Tokenized input text.
            saliency_scores (list): Saliency scores for each token.
            label (int): Ground truth label for the text.

        Returns:
            float: Monotonicity metric (fraction of monotonic increases).
            bool: True if all confidence differences are positive, False otherwise.
        """
        saliency_scores = np.abs(saliency_scores[1:-1])  # Remove [CLS] and [SEP], take abs value
        
        # Sort indices by increasing saliency
        sorted_indices = np.argsort(saliency_scores)

        # Mask all tokens
        masked_tokens = [self.baseline_token] * len(tokens)

        # Generate incrementally unmasked texts
        incremental_texts = []
        for idx in sorted_indices:
            masked_tokens[idx] = tokens[idx]
            incremental_texts.append(self.tokenizer.convert_tokens_to_string(masked_tokens))

        # Compute confidences for all incremental texts in batches
        incremental_confidences = self.compute_confidences(incremental_texts)

        # Compute monotonicity metric
        diff_confidences = np.diff(incremental_confidences)
        monotonic_increases = np.sum(diff_confidences >= 0)
        monotonicity_score = monotonic_increases / len(diff_confidences)

        # Check if fully monotonic
        is_fully_monotonic = np.all(diff_confidences >= 0)

        return monotonicity_score, is_fully_monotonic

    def evaluate_dataset(self, saliency_scores_dict):
        """
        Evaluate monotonicity across the entire dataset.

        Args:
            saliency_scores_dict (list): List of instances, each with:
                                         - tokens
                                         - saliency_scores
                                         - label

        Returns:
            dict: Average monotonicity score and percentage of fully monotonic instances.
        """
        total_score = 0.0
        fully_monotonic_count = 0
        num_instances = len(saliency_scores_dict)
        i=0
        for data in saliency_scores_dict[:20]:
            print(i)
            i+=1
            tokens = data["tokens"]
            saliency_scores = data["saliency_scores"]
            label = data["label"]

            # Compute monotonicity for the instance
            score, is_fully_monotonic = self.monotonicity_metric(tokens, saliency_scores, label)
            total_score += score
            if is_fully_monotonic:
                fully_monotonic_count += 1

        # Average monotonicity score
        average_score = total_score / num_instances
        fully_monotonic_percentage = (fully_monotonic_count / num_instances) * 100

        return {
            "average_monotonicity": average_score,
            "fully_monotonic_percentage": fully_monotonic_percentage,
        }


def compute_all_monotonicity(model, tokenizer, dataset, saliency_scores, batch_size=16):
    """
    Compute monotonicity score across the dataset.

    Args:
        model: Pre-trained classification model.
        tokenizer: Tokenizer corresponding to the model.
        dataset: List of instances with tokens, saliency scores, and labels.
        batch_size: Number of texts to process in one batch.

    Returns:
        dict: Average monotonicity score and percentage of fully monotonic instances.
    """
    monotonicity_evaluator = MonotonicityEvaluator(model, tokenizer, batch_size=batch_size)
    metrics = monotonicity_evaluator.evaluate_dataset(saliency_scores)
    return metrics
