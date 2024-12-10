import numpy as np
import torch
from sklearn.metrics import auc
import matplotlib.pyplot as plt


class FaithfulnessEvaluator:
    def __init__(self, model, tokenizer):
        """
        Initializes the FaithfulnessEvaluator.

        Args:
            model: The pre-trained model (e.g., BERT) to evaluate.
            tokenizer: Tokenizer corresponding to the model.
        """
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def mask_tokens(tokens, importance_scores, threshold):
        """
        Masks the top `threshold%` tokens based on importance scores.

        Args:
            tokens (list): List of tokens.
            importance_scores (list): List of importance scores for each token.
            threshold (float): Percentage of tokens to mask.

        Returns:
            list: Masked tokens.
        """
        n_tokens_to_mask = int(len(tokens) * (threshold / 100))
        indices_to_mask = np.argsort(-np.array(importance_scores))[:n_tokens_to_mask]
        return [
            "[MASK]" if i in indices_to_mask else token
            for i, token in enumerate(tokens)
        ]

    def evaluate_performance(self, precomputed_scores, thresholds):
        """
        Evaluates model performance after masking tokens at various thresholds.

        Args:
            precomputed_scores (list): Precomputed saliency scores for each sample.
            thresholds (list): List of thresholds (percentages of tokens to mask).

        Returns:
            list: Performance scores (accuracy) at each threshold.
        """
        performance_scores = []

        i=0
        for threshold in thresholds:
            print(i)
            i+=1
            masked_texts = []
            labels = []

            for data in precomputed_scores:
                tokens = data["tokens"]
                importance_scores = data["saliency_scores"]
                label = data["label"]

                # Mask tokens based on saliency scores and threshold
                masked_tokens = self.mask_tokens(tokens, importance_scores, threshold)
                masked_text = self.tokenizer.convert_tokens_to_string(masked_tokens)

                masked_texts.append(masked_text)
                labels.append(label)

            # Tokenize and predict
            inputs = self.tokenizer(masked_texts, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            # Compute accuracy or other metrics
            accuracy = np.mean(np.array(predictions) == np.array(labels))
            performance_scores.append(accuracy)

        return performance_scores

    def compute_auc_tp(self, thresholds, performance_scores):
        """
        Computes the Area Under the Threshold-Performance Curve (AUC-TP).

        Args:
            thresholds (list): List of thresholds (percentages of tokens masked).
            performance_scores (list): Performance scores at each threshold.

        Returns:
            float: AUC-TP value.
        """
        auc_tp = auc(thresholds, performance_scores)
        return auc_tp

    def plot_performance(self, thresholds, performance_scores):
        """
        Plots the Threshold-Performance Curve.

        Args:
            thresholds (list): List of thresholds (percentages of tokens masked).
            performance_scores (list): Performance scores at each threshold.
        """
        plt.plot(thresholds, performance_scores, marker="o")
        plt.xlabel("Threshold (% of tokens masked)")
        plt.ylabel("Performance (Accuracy)")
        plt.title("Threshold-Performance Curve")
        plt.show()
