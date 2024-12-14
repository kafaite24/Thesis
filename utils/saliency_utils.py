import numpy as np
import copy
def top_k_selection(saliency_scores, k=7):
    """
    Select top-k saliency scores based on their absolute values.

    Args:
        saliency_scores (numpy.ndarray): Array of saliency scores.
        k (int): The number of top saliency scores to select.

    Returns:
        numpy.ndarray: A binary array with `1` for the top-k selected scores and `0` for others.
    """
    sorted_indices = np.argsort(np.abs(saliency_scores))  # Sort by absolute value
    top_k_indices = sorted_indices[-k:]  # Get the indices of the top k absolute values
    thresholded = np.zeros_like(saliency_scores, dtype=int)
    thresholded[top_k_indices] = 1  # Set the top k saliency scores to 1
    return thresholded

def min_max_normalize(values):
    """
    Normalize a list of values to the range [0, 1] using min-max normalization.

    Args:
        values (list or np.array): A list or numpy array of numerical values to be normalized.

    Returns:
        np.array: Normalized values in the range [0, 1].
    """
    # Convert to numpy array if it's not already
    values = np.array(values)
    
    # Compute the min and max values
    min_val = np.min(values)
    max_val = np.max(values)
    
    # Avoid division by zero if all values are the same
    if max_val - min_val == 0:
        return np.zeros_like(values)
    
    # Perform min-max normalization
    normalized_values = (values - min_val) / (max_val - min_val)
    
    return normalized_values

def lp_normalize(self,explanations, ord=1):
        """Run Lp-noramlization of explanation attribution scores

        Args:
            explanations (List[Explanation]): list of explanations to normalize
            ord (int, optional): order of the norm. Defaults to 1.

        Returns:
            List[Explanation]: list of normalized explanations
        """

        new_exps = list()
        for exp in explanations:
            new_exp = copy.copy(exp)
            if isinstance(new_exp.scores, np.ndarray) and new_exp.scores.size > 0:
                norm_axis = (
                    -1 if new_exp.scores.ndim == 1 else (0, 1)
                )  # handle axis correctly
                norm = np.linalg.norm(new_exp.scores, axis=norm_axis, ord=ord)
                if norm != 0:  # avoid division by zero
                    new_exp.scores /= norm
            new_exps.append(new_exp)

        return new_exps