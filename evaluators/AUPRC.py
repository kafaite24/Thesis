# import numpy as np
# from sklearn.metrics import auc

# class AUPRCEvaluator:
#     """
#     Class to evaluate saliency explanations using AUPRC (Area Under the Precision-Recall Curve).
#     """

#     def __init__(self, dataset, saliency_scores):
#         """
#         Initialize the evaluator.

#         Args:
#             dataset: Dataset object (e.g., MovieReviews) with ground truth rationales.
#             saliency_scores: Precomputed saliency scores as a list of dictionaries.
#         """
#         self.dataset = dataset
#         self.saliency_scores = saliency_scores

#     def calculate_precision_recall(self, saliency_scores, ground_truth_rationale):
#         """
#         Calculate precision and recall at multiple thresholds for a single sample.

#         Args:
#             saliency_scores (list of float): Continuous saliency scores for tokens.
#             ground_truth_rationale (list of int): Binary ground-truth rationale mask.

#         Returns:
#             tuple: Arrays of precision and recall values.
#         """
#         # Ensure saliency scores and rationale masks have the same length
#         min_len = min(len(saliency_scores), len(ground_truth_rationale))
#         saliency_scores = saliency_scores[:min_len]
#         ground_truth_rationale = ground_truth_rationale[:min_len]

#         # Sort tokens by descending saliency scores
#         sorted_indices = np.argsort(-np.array(saliency_scores))
#         sorted_ground_truth = np.array(ground_truth_rationale)[sorted_indices]

#         # Compute cumulative sums for true positives and false positives
#         tp_cumsum = np.cumsum(sorted_ground_truth)
#         fp_cumsum = np.cumsum(1 - sorted_ground_truth)

#         # Total positives in ground truth
#         total_positives = np.sum(ground_truth_rationale)

#         # Calculate precision and recall
#         precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)
#         recall = tp_cumsum / (total_positives + 1e-9)

#         return precision, recall

#     def calculate_auprc(self, saliency_scores, ground_truth_rationale):
#         """
#         Calculate AUPRC for a single instance.

#         Args:
#             saliency_scores (list of float): Continuous saliency scores for tokens.
#             ground_truth_rationale (list of int): Binary ground-truth rationale mask.

#         Returns:
#             float: AUPRC score for the instance.
#         """
#         precision, recall = self.calculate_precision_recall(saliency_scores, ground_truth_rationale)
#         return auc(recall, precision)

# def evaluate(dataset, saliency_scores, split_type="test"):
#     """
#     Evaluate AUPRC for the entire dataset split.

#     Args:
#         split_type: Dataset split to evaluate (e.g., "test", "train").

#     Returns:
#         float: Average AUPRC score across the dataset.
#     """
#     AUPRC_evaluator = AUPRCEvaluator(dataset, saliency_scores)
#     auprc_scores = []

#     print(f"Evaluating {split_type} split with AUPRC...")
#     # Iterate over precomputed saliency scores
#     for idx, entry in enumerate(saliency_scores):
#         # Get ground truth rationale mask from dataset
#         instance = dataset.get_instance(idx, split_type=split_type)
#         ground_truth_rationale = instance["rationale"]
       
#         if(len(ground_truth_rationale)>0):
#             saliency_score = saliency_scores[idx]["saliency_scores"]
#             # print(f"tokens in instance {instance['tokens']}")
#             # print(f"tokens in saliency scores {saliency_scores[idx]['tokens']}")
#             print(f"saliency scores length {len(saliency_scores[idx]['saliency_scores'])}, ground truth length {len(ground_truth_rationale)}")

#             auprc = AUPRC_evaluator.calculate_auprc(saliency_score, ground_truth_rationale)
#             auprc_scores.append(auprc)
#             print(f"Text {idx + 1}:")
#             print(f"AUPRC: {auprc:.4f}\n")
#         else:
#             continue
#     # Average AUPRC across the dataset
#     average_auprc = np.mean(auprc_scores)
#     print(f"Average AUPRC Across Dataset: {average_auprc:.4f}")
#     return average_auprc


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

    def calculate_auprc(self, saliency_scores, ground_truth_rationale):
        """
        Calculate AUPRC for a single instance using sklearn's precision_recall_curve.

        Args:
            saliency_scores (list of float): Continuous saliency scores for tokens.
            ground_truth_rationale (list of int): Binary ground-truth rationale mask.

        Returns:
            float: AUPRC score for the instance.
        """
        # Ensure saliency scores and rationale masks have the same length
        min_len = min(len(saliency_scores), len(ground_truth_rationale))
        saliency_scores = saliency_scores[:min_len]
        ground_truth_rationale = ground_truth_rationale[:min_len]

        # Use sklearn's precision_recall_curve to calculate precision, recall, and thresholds
        precision, recall, _ = precision_recall_curve(ground_truth_rationale, saliency_scores)

       # Calculate AUPRC (Area Under the Precision-Recall Curve)
        return precision, recall, auc(recall, precision)

def evaluate(dataset, saliency_scores, split_type="test"):
    """
    Evaluate AUPRC for the entire dataset split.

    Args:
        split_type: Dataset split to evaluate (e.g., "test", "train").

    Returns:
        float: Average AUPRC score across the dataset.
    """
    AUPRC_evaluator = AUPRCEvaluator(dataset, saliency_scores)
    auprc_scores = []
    # Initialize lists for plotting
    all_precision = []
    all_recall = []

    print(f"Evaluating {split_type} split with AUPRC...")
    # Iterate over precomputed saliency scores
    for idx, entry in enumerate(saliency_scores[:10]):
        # Get ground truth rationale mask from dataset
        instance = dataset.get_instance(idx, split_type=split_type)
        ground_truth_rationale = instance["rationale"]

        if len(ground_truth_rationale) > 0:
            saliency_score = saliency_scores[idx]["saliency_scores"]
            print(f"saliency scores length {len(saliency_scores[idx]['saliency_scores'])}, ground truth length {len(ground_truth_rationale)}")

            precision, recall, auprc = AUPRC_evaluator.calculate_auprc(saliency_score, ground_truth_rationale)
            auprc_scores.append(auprc)
            all_precision.append(precision)
            all_recall.append(recall)
            print(f"Text {idx + 1}:")
            print(f"AUPRC: {auprc:.4f}\n")
        else:
            continue
    
    # Average AUPRC across the dataset
    average_auprc = np.mean(auprc_scores)
    print(f"Average AUPRC Across Dataset: {average_auprc:.4f}")

    # # Plot AUPRC curves for the first 10 instances
    # plt.figure(figsize=(8, 6))
    # for i in range(len(all_precision)):
    #     plt.plot(all_recall[i], all_precision[i], label=f"Text {i+1} (AUPRC = {auprc_scores[i]:.4f})")
    
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve (AUPRC) for Sample Instances')
    # plt.legend(loc='lower left')
    # plt.grid(True)
    # plt.show()

    return average_auprc
