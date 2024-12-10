import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
class MonotonicityEvaluator:
    def __init__(self, model, tokenizer, baseline_token="[MASK]"):
        """
        Initializes the MonotonicityEvaluator.

        Args:
            model: Pre-trained classification model (e.g., BERT for sentiment analysis).
            tokenizer: Tokenizer corresponding to the model.
            baseline_token: Token used to mask input tokens.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.baseline_token = baseline_token

    def mask_all_tokens(self, tokens):
        """
        Replace all tokens with the baseline token.

        Args:
            tokens (list): List of tokens in the input text.

        Returns:
            list: List of tokens with all tokens replaced by the baseline token.
        """
        return [self.baseline_token] * len(tokens)

    def monotonicity_metric(self, tokens, saliency_scores, label):
        """
        Computes the monotonicity metric for a single instance.

        Args:
            tokens (list): Tokenized input text.
            saliency_scores (list): Saliency scores for each token.
            label (int): Ground truth label for the text.

        Returns:
            float: Monotonicity metric (fraction of monotonic increases).
        """
        # print(f"\nEvaluating monotonicity for tokens: {tokens}")
        # print(f"Saliency scores: {saliency_scores}")
        print(f"True label: {label}")

        # Convert tokens to text
        original_text = self.tokenizer.convert_tokens_to_string(tokens)
        # print(f"Original text: {original_text}")

        # Tokenize original text and compute model's confidence
        inputs = self.tokenizer(original_text, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        pred_class = np.argmax(probs)
        pred_confidence = probs[0][pred_class]

        print(f"Predicted class: {pred_class}, Original confidence: {pred_confidence:.4f}")

        # Mask all tokens
        masked_tokens = self.mask_all_tokens(tokens)
        masked_text = self.tokenizer.convert_tokens_to_string(masked_tokens)
        # print(f"Masked text: {masked_text}")

        inputs_masked = self.tokenizer(masked_text, return_tensors="pt", truncation=True, padding=True)
        inputs_masked = {key: val.to(self.model.device) for key, val in inputs_masked.items()}

        with torch.no_grad():
            masked_logits = self.model(**inputs_masked).logits
            masked_probs = torch.softmax(masked_logits, dim=1).cpu().numpy()

        confidences = [masked_probs[0][pred_class]]
        print(f"Initial confidence with all tokens masked: {confidences[0]:.4f}")

        # Incrementally add tokens in order of saliency
        sorted_indices = np.argsort(saliency_scores)  # Increasing order of saliency
        print('printing len of sorted indices', len(sorted_indices))
        i=0
        for idx in sorted_indices:
            print('printing idx',i)
            i+=1
            masked_tokens[idx] = tokens[idx]
            incremental_text = self.tokenizer.convert_tokens_to_string(masked_tokens)
            # print(f"Adding token '{tokens[idx]}': {incremental_text}")

            inputs_incremental = self.tokenizer(incremental_text, return_tensors="pt", truncation=True, padding=True)
            inputs_incremental = {key: val.to(self.model.device) for key, val in inputs_incremental.items()}

            with torch.no_grad():
                incremental_logits = self.model(**inputs_incremental).logits
                incremental_probs = torch.softmax(incremental_logits, dim=1).cpu().numpy()

            confidence = incremental_probs[0][pred_class]
            confidences.append(confidence)
            # print(f"Confidence after adding '{tokens[idx]}': {confidence:.4f}")

        # Compute monotonicity metric
        diff_confidences = np.diff(confidences)
        monotonic_increases = np.sum(diff_confidences >= 0)
        monotonicity_score = monotonic_increases / len(diff_confidences)

        # print(f"Confidence sequence: {confidences}")
        print(f"Monotonicity metric for this instance: {monotonicity_score:.4f}\n")

        return monotonicity_score

    def evaluate_instance(self, tokens, saliency_scores, label):
        """
        Evaluate a single instance using the evaluator.
        
        Args:
            tokens (list): List of tokens in the input.
            saliency_scores (list): Saliency scores corresponding to the tokens.
            label (int): Ground truth label.
        
        Returns:
            float: Computed metric (e.g., monotonicity score).
        """
        return self.monotonicity_metric(tokens, saliency_scores, label)

    def evaluate_dataset(self, dataset):
        """
        Evaluate the entire dataset.
        
        Args:
            dataset (list): List of dataset instances. Each instance should have:
                            - tokens
                            - saliency_scores
                            - label
        
        Returns:
            float: Average score across the dataset.
        """
      
        total_score = 0.0
    
        for i, data in enumerate(dataset[:5]):
            print(f"Evaluating instance {i + 1}/{len(dataset)}")
            tokens = data["tokens"]
            saliency_scores = data["saliency_scores"]
            label = data["label"]
            
            # Call the evaluation function for the instance
            score = self.evaluate_instance(tokens, saliency_scores, label)
            print(f"Score for instance {i + 1}: {score:.4f}")
            total_score += score
        
        # Compute average score
        average_score = total_score / len(dataset)
        print(f"\nAverage Score Across Dataset: {average_score:.4f}")
        return average_score
    


def compute_all_monotonicity(model, tokenizer, saliency_scores, device="cpu"):
    monotonicity_evaluator = MonotonicityEvaluator(model, tokenizer)
    average_monotonicity = monotonicity_evaluator.evaluate_dataset(saliency_scores)
    
    return average_monotonicity
