import numpy as np
from sklearn.metrics import auc, precision_recall_curve
import matplotlib.pyplot as plt

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

    def calculate_auprc(self, saliency_scores_batch, ground_truth_rationales_batch):
        """
        Calculate AUPRC for multiple instances using sklearn's precision_recall_curve.

        Args:
            saliency_scores_batch (list of list of float): Saliency scores for multiple instances.
            ground_truth_rationales_batch (list of list of int): Ground-truth rationale masks for multiple instances.

        Returns:
            list of float: AUPRC scores for the batch.
        """
        auprc_scores = []

        for saliency_scores, ground_truth_rationale in zip(saliency_scores_batch, ground_truth_rationales_batch):
            # Remove [CLS] and [SEP] tokens
            saliency_scores = saliency_scores[1:-1]  # Remove first ([CLS]) and last ([SEP]) tokens
            ground_truth_rationale = ground_truth_rationale[:len(saliency_scores)]  # Align lengths

            if len(saliency_scores) == 0 or len(ground_truth_rationale) == 0:
                continue  # Skip if empty

            # Compute precision-recall and AUC
            precision, recall, _ = precision_recall_curve(ground_truth_rationale, saliency_scores)
            auprc = auc(recall, precision)
            auprc_scores.append(auprc)

        return auprc_scores

def evaluate(dataset, saliency_scores, split_type="test", batch_size=32):
    """
    Evaluate AUPRC for the entire dataset split.

    Args:
        split_type: Dataset split to evaluate (e.g., "test", "train").
        batch_size: Number of instances to process in a batch.

    Returns:
        float: Average AUPRC score across the dataset.
    """
    AUPRC_evaluator = AUPRCEvaluator(dataset, saliency_scores)
    auprc_scores = []

    print(f"Evaluating {split_type} split with AUPRC...")

    # Prepare data in batches
    num_batches = len(saliency_scores) // batch_size + int(len(saliency_scores) % batch_size > 0)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(saliency_scores), len(dataset))

        # Batch saliency scores and ground truth rationales
        saliency_scores_batch = [
            entry["saliency_scores"] for entry in saliency_scores[start_idx:end_idx]
        ]
        ground_truth_rationales_batch = [
            dataset.get_instance(idx, split_type=split_type)["rationale"]
            for idx in range(start_idx, end_idx)
        ]

        # Filter out instances with empty rationales
        filtered_batch = [
            (saliency, rationale)
            for saliency, rationale in zip(saliency_scores_batch, ground_truth_rationales_batch)
            if len(rationale) > 0
        ]

        if not filtered_batch:
            continue  # Skip if no valid data in the batch

        # Unzip the filtered batch
        saliency_scores_batch, ground_truth_rationales_batch = zip(*filtered_batch)

        # Compute AUPRC for the batch
        batch_auprc_scores = AUPRC_evaluator.calculate_auprc(
            saliency_scores_batch, ground_truth_rationales_batch
        )
        auprc_scores.extend(batch_auprc_scores)

    # Average AUPRC across the dataset
    average_auprc = np.mean(auprc_scores)
    print(f"Average AUPRC Across Dataset: {average_auprc:.4f}")

    return average_auprc
