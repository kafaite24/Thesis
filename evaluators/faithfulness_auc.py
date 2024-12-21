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

        
        for threshold in thresholds:
            print(f"threshold---------- {threshold}")
            
            masked_texts = []
            labels = []
            i=0
            for data in precomputed_scores[:400]:
                # print(i)
                i+=1
                tokens = data["tokens"]
                importance_scores = data["saliency_scores"]
                label = data["label"]

                # Mask tokens based on saliency scores and threshold
                masked_tokens = self.mask_tokens(tokens, importance_scores, threshold)
                masked_text = self.tokenizer.convert_tokens_to_string(masked_tokens)

                masked_texts.append(masked_text)
                # print(f"masked texts {masked_text}")
                if isinstance(label, str):
                    label = self.model.config.label2id[label]

                labels.append(label)

            # Tokenize and predict
            inputs = self.tokenizer(masked_texts, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            # print(f"prediction {predictions} labels {labels}")
            # Compute accuracy or other metrics
            accuracy = np.mean(np.array(predictions) == np.array(labels))
            performance_scores.append(accuracy)

        return performance_scores

    def compute_auc_tp(self, thresholds, performance_scores, total_features):
        """
        Computes the Area Under the Threshold-Performance Curve (AUC-TP).

        Args:
            thresholds (list): List of thresholds (percentages of tokens masked).
            performance_scores (list): Performance scores at each threshold.

        Returns:
            float: AUC-TP value.
        """
        auc_tp = auc(thresholds, performance_scores)
        standardized_auc_tp = auc_tp / total_features
        return standardized_auc_tp

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

def compute_faithfulness_auc(model, tokenizer, precomputed_scores, thresholds):
    """
    Computes the faithfulness AUC for the given model, tokenizer, and precomputed scores.

    Args:
        model: Pre-trained model to evaluate.
        tokenizer: Tokenizer corresponding to the model.
        precomputed_scores: Precomputed saliency scores for each sample.
        thresholds (list): List of thresholds (percentages of tokens to mask).

    Returns:
        dict: A dictionary containing performance scores and AUC-TP.
    """
    evaluator = FaithfulnessEvaluator(model, tokenizer)

    # Evaluate performance across thresholds
    performance_scores = evaluator.evaluate_performance(precomputed_scores, thresholds)

     # Compute total features (tokens) in the dataset
    total_features = sum(len(data["tokens"]) for data in precomputed_scores)

    # Compute AUC-TP
    auc_tp = evaluator.compute_auc_tp(thresholds, performance_scores,total_features)
    evaluator.plot_performance(thresholds,performance_scores)
    # Return results
    return {
        "performance_scores": performance_scores,
        "auc_tp": auc_tp,
    }