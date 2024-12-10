import numpy as np

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