import numpy as np
from utils.saliency_utils import top_k_selection  # Import the function

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
            threshold: Threshold for binarizing saliency scores.
        """
        self.dataset = dataset
        self.saliency_scores = saliency_scores

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
        # ground_truth = np.ravel(ground_truth)
        predicted_set = set([i for i, p in enumerate(predicted) if p == 1])
        ground_truth_set = set([i for i, g in enumerate(ground_truth) if g == 1])

        intersection = len(predicted_set & ground_truth_set)
        union = len(predicted_set | ground_truth_set)

        iou = intersection / union if union > 0 else 0.0
        precision = intersection / len(predicted_set) if predicted_set else 0.0
        recall = intersection / len(ground_truth_set) if ground_truth_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {"IOU": iou, "Precision": precision, "Recall": recall, "F1": f1}

    def evaluate(self, split_type="test"):
        """
        Evaluate saliency explanations for the specified dataset split.

        Args:
            split_type: Dataset split to evaluate (e.g., "test", "train").

        Returns:
            dict: Average metrics across the dataset split, including F1 IOU.
        """
        metrics_list = []
        # Iterate over precomputed saliency scores
        for idx, entry in enumerate(self.saliency_scores):
            # if idx >= 2:  # Limit to 2 entries
            #     break
            instance = self.dataset.get_instance(idx, split_type=split_type)          
            ground_truth_rationale = instance["rationale"]
            saliency_scores = np.array(entry["saliency_scores"])
            k=int(len(saliency_scores)*0.30)
            predicted_rationale= top_k_selection(saliency_scores,k)
           
            # Ensure the instance and entry are aligned
            if instance["text"] != entry["text"][0]:
                print(f"Mismatch! Instance text: {instance['text']}, Saliency text: {entry['text'][0]}")
                return
            # Calculate metrics
            metrics = self.calculate_discrete_metrics(predicted_rationale, ground_truth_rationale)
            metrics_list.append(metrics)

        # Aggregate metrics
        average_metrics = {
            "IOU": np.mean([m["IOU"] for m in metrics_list]),
            "F1": np.mean([m["F1"] for m in metrics_list]),
        }

        return average_metrics


def compute_all_IOU(dataset, saliency_scores, device="cpu"):
    
    IOU_evaluator = IOUEvaluator(dataset, saliency_scores)
    IOU = IOU_evaluator.evaluate()
    
    return IOU
