import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.special import softmax
import torch
from scipy.integrate import trapz
class FADEvaluator:
    """
    Class to evaluate the impact of feature (token) dropping on text data
    based on saliency scores (e.g., Gradient x Input).
    """

    def __init__(self, dataset, model, saliency_scores, tokenizer, device):
        """
        Initialize the evaluator.

        Args:
            dataset: The dataset object (e.g., MovieReviews).
            model: The trained text classification model.
            saliency_scores: Precomputed saliency scores for tokens.
            tokenizer: Tokenizer used for the model.
            device: Device to run the evaluation (e.g., 'cuda' or 'cpu').
        """
        self.dataset = dataset
        self.model = model.to(device)
        self.saliency_scores = saliency_scores
        self.tokenizer = tokenizer
        self.device = device

    def _replace_tokens_with_baseline(self, tokens, saliency_scores, percent_to_drop):
        """
        Replace the top percent of important tokens with baseline tokens (e.g., [MASK]).

        Args:
            tokens (list of str): Tokenized text.
            saliency_scores (np.array): Saliency scores for the tokens.
            percent_to_drop (float): Percent of features (tokens) to replace.

        Returns:
            List[str]: Tokens with top features replaced by a baseline token.
        """
        tokens_modified = tokens.copy()
        num_tokens_to_drop = int(len(tokens) * percent_to_drop / 100)
        
        saliency_sorted_indices = np.argsort(-np.abs(saliency_scores))  # Descending order
        tokens_to_replace = saliency_sorted_indices[:num_tokens_to_drop]
        
        # Replace selected tokens with a baseline token (e.g., [MASK])
        baseline_token = self.tokenizer.mask_token
        for idx in tokens_to_replace:
            tokens_modified[idx] = baseline_token
        return tokens_modified

    def evaluate(self, percent_dropped_features, split_type="test"):
        """
        Evaluate the impact of feature (token) dropping on text classification accuracy.

        Args:
            percent_dropped_features (list of float): List of percentages of tokens to drop.
            split_type (str): Dataset split to evaluate (e.g., 'test').

        Returns:
            pd.DataFrame: Accuracy scores for each percentage of tokens dropped.
        """
        results = []
        
        for percent_to_drop in percent_dropped_features:
            predictions, labels = [], []
            print(f"------------------------perc to drop {percent_to_drop}-------------------------------------")
            for idx, entry in enumerate(self.saliency_scores[:1]): 
                print(f"------------------------idx {idx}-------------------------------------")

                text = entry["text"]
                tokens = entry["tokens"]
                saliency_scores = np.array(entry["saliency_scores"])
                label = entry["label"]
                # Replace tokens with baseline values
                modified_tokens = self._replace_tokens_with_baseline(tokens, saliency_scores, percent_to_drop)
               
                # Detokenize back into a string
                modified_text = self.tokenizer.convert_tokens_to_string(modified_tokens)
                # Tokenize and encode input for the model
                encoded_input = self.tokenizer(modified_text, return_tensors="pt", truncation=True, padding=True).to(self.device)

                # Make prediction
                with torch.no_grad():
                    logits = self.model(**encoded_input).logits
                
                prediction = torch.argmax(logits, dim=1).item()
                # Convert label to numerical format if it's a string
                if isinstance(label, str):
                    label = self.model.config.label2id[label]

                predictions.append(prediction)
                labels.append(label)

            # Compute accuracy
            accuracy = accuracy_score(labels, predictions)
            print(f"Percent Dropped: {percent_to_drop} Accuracy: {accuracy}")
            results.append({"Percent Dropped": percent_to_drop, "Accuracy": accuracy})
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        return results_df

    def calculate_n_auc(self, results_df, percent_range=(0,20)):
        """
        Calculate the Normalized Area Under the Curve (N-AUC).

        Args:
            results_df (pd.DataFrame): DataFrame with percent dropped and accuracy.
            percent_range (tuple): Range of percentages to calculate N-AUC (e.g., (0, 20)).

        Returns:
            float: Normalized AUC.
        """
        # Filter the results within the specified range
        filtered_results = results_df[(results_df["Percent Dropped"] >= percent_range[0]) &
                                      (results_df["Percent Dropped"] <= percent_range[1])]
        
        # Percentages and accuracies
        x = filtered_results["Percent Dropped"].values
        y = filtered_results["Accuracy"].values

        # Calculate the area under the curve using trapezoidal integration
        auc = trapz(y, x)
        
        # Calculate the maximum possible area
        max_auc = (x[-1] - x[0]) * max(y)

        # Normalize the AUC
        n_auc = auc / max_auc if max_auc > 0 else 0.0

        return n_auc
    
    def plot_results(self, results_df):
        """
        Plot accuracy as a function of percent dropped tokens.

        Args:
            results_df (pd.DataFrame): DataFrame with percent dropped and accuracy.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(results_df["Percent Dropped"], results_df["Accuracy"], marker="o", label="Accuracy")
        plt.xlabel("Percent of Tokens Dropped")
        plt.ylabel("Accuracy")
        plt.title("Impact of Dropping Top Tokens (Saliency-based) on Accuracy")
        plt.grid()
        plt.legend()
        plt.show()


def compute_all_fad(dataset, model, tokenizer, saliency_scores, device="cpu", percent_dropped_features=None, percent_range=(0, 20)):
    """
    Computes the Normalized AUC (n_auc) for Feature Attribution Drop (FAD).
    
    Args:
        dataset: The dataset object (e.g., MovieReviews).
        model: The trained text classification model.
        tokenizer: Tokenizer used to tokenize inputs for the model.
        saliency_scores: Precomputed saliency scores for each sample in the dataset.
        device: Device to run the computations on (e.g., 'cuda' or 'cpu').
        percent_dropped_features (list, optional): List of percentages of tokens to drop. Default is [0, 5, 10, 15, 20].
        percent_range (tuple, optional): Range of percentages to calculate N-AUC (e.g., (0, 20)).
    
    Returns:
        float: The final Normalized AUC (n_auc) value for the dataset.
    """
    if percent_dropped_features is None:
        percent_dropped_features = list(range(0, 101, 10)) # Default percentages
    
    # Initialize the FADEvaluator
    fade_evaluator = FADEvaluator(dataset, model, saliency_scores, tokenizer, device)
    
    # Perform the evaluation
    results_df = fade_evaluator.evaluate(percent_dropped_features)
    print(f"results df {results_df}")
    # Compute the Normalized AUC (n_auc)
    final_n_auc = fade_evaluator.calculate_n_auc(results_df, percent_range)
    fade_evaluator.plot_results(results_df)
    
    return final_n_auc


