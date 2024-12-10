import numpy as np
from sklearn.metrics import auc

class AUPRCEvaluator:
    """
    Class to evaluate saliency explanations using AUPRC (Area Under the Precision-Recall Curve).
    """

    def __init__(self, dataset, saliency_scores):
        """
        Initialize the evaluator.

        Args:
            dataset: Dataset object (e.g., MovieReviews) with ground truth rationales.
            saliency_scores: Precomputed saliency scores as a list of dictionaries.
        """
        self.dataset = dataset
        self.saliency_scores = saliency_scores

    def calculate_precision_recall(self, saliency_scores, ground_truth_rationale):
        """
        Calculate precision and recall at multiple thresholds for a single sample.

        Args:
            saliency_scores (list of float): Continuous saliency scores for tokens.
            ground_truth_rationale (list of int): Binary ground-truth rationale mask.

        Returns:
            tuple: Arrays of precision and recall values.
        """
        # Ensure saliency scores and rationale masks have the same length
        print(f"saliency scores length {len(saliency_scores)}, ground trurh length {len(ground_truth_rationale)}")
        print(ground_truth_rationale)
        min_len = min(len(saliency_scores), len(ground_truth_rationale))
        saliency_scores = saliency_scores[:min_len]
        ground_truth_rationale = ground_truth_rationale[:min_len]

        # Sort tokens by descending saliency scores
        sorted_indices = np.argsort(-np.array(saliency_scores))
        sorted_ground_truth = np.array(ground_truth_rationale)[sorted_indices]

        # Compute cumulative sums for true positives and false positives
        tp_cumsum = np.cumsum(sorted_ground_truth)
        fp_cumsum = np.cumsum(1 - sorted_ground_truth)

        # Total positives in ground truth
        total_positives = np.sum(ground_truth_rationale)

        # Calculate precision and recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)
        recall = tp_cumsum / (total_positives + 1e-9)

        return precision, recall

    def calculate_auprc(self, saliency_scores, ground_truth_rationale):
        """
        Calculate AUPRC for a single instance.

        Args:
            saliency_scores (list of float): Continuous saliency scores for tokens.
            ground_truth_rationale (list of int): Binary ground-truth rationale mask.

        Returns:
            float: AUPRC score for the instance.
        """
        precision, recall = self.calculate_precision_recall(saliency_scores, ground_truth_rationale)
        return auc(recall, precision)

def evaluate(self, split_type="test"):
    """
    Evaluate AUPRC for the entire dataset split.

    Args:
        split_type: Dataset split to evaluate (e.g., "test", "train").

    Returns:
        float: Average AUPRC score across the dataset.
    """
    auprc_scores = []

    print(f"Evaluating {split_type} split with AUPRC...")

    # Iterate over precomputed saliency scores
    for idx, entry in enumerate(self.saliency_scores):
        text = entry["text"]
        saliency_scores = np.array(entry["saliency_scores"])

        # Get ground truth rationale mask from dataset
        review_data = self.dataset.get_review(idx, split_type=split_type)
        ground_truth_rationale = review_data["rationale"]

        # Calculate AUPRC
        auprc = self.calculate_auprc(saliency_scores, ground_truth_rationale)
        auprc_scores.append(auprc)

        # Debug output for individual samples
        print(f"Review {idx + 1}:")
        # print(f"Text: {text}")
        print(f"AUPRC: {auprc:.4f}\n")

    # Average AUPRC across the dataset
    average_auprc = np.mean(auprc_scores)
    print(f"Average AUPRC Across Dataset: {average_auprc:.4f}")
    return average_auprc
