import numpy as np
from utils.saliency_utils import top_k_selection

class IOUEvaluator:
    """
    Class to evaluate faithfulness of saliency explanations using discrete metrics,
    including F1 IOU score.
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
        self.avg_rationale_length = self.calculate_avg_rationale_length()

    def calculate_avg_rationale_length(self):
        """
        Calculate the average length of human rationales across the dataset.

        Returns:
            int: Average length of rationales.
        """
        rationale_lengths = [
            len([r for r in self.dataset.get_instance(idx)["rationale"] if r == 1])
            for idx in range(self.dataset.len("test"))
        ]

        return int(np.mean(rationale_lengths))

    @staticmethod
    def calculate_discrete_metrics(predicted, ground_truth):
        """
        Calculate IOU, precision, recall, and F1 for discrete rationale extraction.

        Args:
            predicted (list of int): Binary labels for predicted rationales (1 for important).
            ground_truth (list of int): Binary labels for ground truth rationales.

        Returns:
            dict: Dictionary containing IOU, precision, recall, and F1 score.
        """
        predicted_set = set(np.where(predicted == 1)[0])
        ground_truth_set = set(np.where(ground_truth == 1)[0])

        intersection = len(predicted_set & ground_truth_set)
        union = len(predicted_set | ground_truth_set)

        iou = intersection / union if union > 0 else 0.0
        precision = intersection / len(predicted_set) if predicted_set else 0.0
        recall = intersection / len(ground_truth_set) if ground_truth_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {"IOU": iou, "Precision": precision, "Recall": recall, "F1": f1}

    def evaluate_batch(self, saliency_scores_batch, rationale_batch):
        """
        Evaluate a batch of instances.

        Args:
            saliency_scores_batch: List of saliency scores for the batch.
            rationale_batch: List of ground truth rationales for the batch.

        Returns:
            List[dict]: List of metrics for each instance in the batch.
        """
        metrics_list = []
        k = self.avg_rationale_length
        for saliency_scores, ground_truth_rationale in zip(saliency_scores_batch, rationale_batch):
             # Remove [CLS] and [SEP] tokens
            saliency_scores = saliency_scores[1:-1]  # Remove first ([CLS]) and last ([SEP]) tokens
            ground_truth_rationale = ground_truth_rationale[:len(saliency_scores)]  # Align lengths
            
            predicted_rationale = top_k_selection(np.array(saliency_scores), k)

            metrics = self.calculate_discrete_metrics(predicted_rationale, np.array(ground_truth_rationale))
            metrics_list.append(metrics)

        return metrics_list

    def evaluate(self, split_type="test", batch_size=32):
        """
        Evaluate saliency explanations for the specified dataset split.

        Args:
            split_type: Dataset split to evaluate (e.g., "test", "train").
            batch_size: Batch size for evaluation.

        Returns:
            dict: Average metrics across the dataset split, including F1 IOU.
        """
        num_instances = len(self.saliency_scores)
        num_batches = (num_instances + batch_size - 1) // batch_size

        metrics_list = []

        for batch_idx in range(num_batches):
            print(batch_idx)
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_instances)

            # Batch saliency scores and ground truth rationales
            saliency_scores_batch = [
                entry["saliency_scores"] for entry in self.saliency_scores[start_idx:end_idx]
            ]
            rationale_batch = [
                self.dataset.get_instance(idx, split_type=split_type)["rationale"]
                for idx in range(start_idx, end_idx)
            ]

            # Evaluate batch
            batch_metrics = self.evaluate_batch(saliency_scores_batch, rationale_batch)
            metrics_list.extend(batch_metrics)

        # Aggregate metrics
        average_metrics = {
            "IOU": np.mean([m["IOU"] for m in metrics_list]),
            "F1": np.mean([m["F1"] for m in metrics_list]),
        }

        return average_metrics


def compute_all_IOU(dataset, saliency_scores, device="cpu", batch_size=32):
    """
    Compute IOU and F1 scores for the entire dataset.

    Args:
        dataset: The dataset object.
        saliency_scores: Precomputed saliency scores.
        device: Device to use for computations.
        batch_size: Batch size for evaluation.

    Returns:
        dict: Average metrics across the dataset.
    """
    IOU_evaluator = IOUEvaluator(dataset, saliency_scores)
    IOU_metrics = IOU_evaluator.evaluate(batch_size=batch_size)
    return IOU_metrics
