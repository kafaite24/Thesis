import json
from tqdm import tqdm
import json
from tqdm import tqdm
import copy
import numpy as np
class SaliencyScoreManager:
    """
    A generic class to compute, save, and load saliency scores for different models and explainers.
    """

    def __init__(self, model, tokenizer, explainer_class, device="cpu", **explainer_kwargs):
        """
        Initializes the SaliencyScoreManager with a model, tokenizer, and explainer class.
        
        Parameters:
            model: The model to explain (e.g., BERT).
            tokenizer: The tokenizer corresponding to the model.
            explainer_class: The class of the explainer (e.g., GradientExplainer).
            device: The device to run computations on (e.g., "cpu" or "cuda").
            explainer_kwargs: Additional keyword arguments for the explainer.
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.explainer = explainer_class(model, tokenizer, device=device, **explainer_kwargs)
    
    def lp_normalize_saliency_data(self,saliency_data, ord=1):
        """Run Lp-normalization of saliency scores in the given data.

        Args:
            saliency_data (List[Dict]): List of dictionaries containing text, saliency scores, etc.
            ord (int, optional): Order of the norm. Defaults to 1.

        Returns:
            List[Dict]: List of dictionaries with normalized saliency scores.
        """
        new_saliency_data = list()  # Will hold the new data with normalized saliency scores
        for data in saliency_data:
            new_data = copy.copy(data)  # Make a copy of the data entry to preserve original
            
            # Extract the saliency scores and apply Lp normalization
            saliency_scores = np.array(new_data['saliency_scores'])
            
            # Compute the Lp norm
            norm = np.linalg.norm(saliency_scores, ord=ord)
            
            if norm != 0:  # Avoid division by zero
                saliency_scores /= norm  # Normalize the saliency scores
            
            new_data['saliency_scores'] = saliency_scores.tolist()  # Update the saliency scores in the new data
            
            # Add the new data entry to the output list
            new_saliency_data.append(new_data)
        return new_saliency_data

    def compute_and_save_scores(self, dataset, save_path, split_type="test"):
        """
        Computes saliency scores for a dataset and saves them to a file.
        
        Parameters:
            dataset: The dataset to compute scores for.
            save_path: Path to save the scores.
            split_type: The split of the dataset (e.g., "test", "train").
        """
        saliency_data = []
        print(f'dataset length {dataset.len()}')
        # Loop through the dataset
        for idx in tqdm(range(dataset.len())): #dataset.get_data_length()

            instance = dataset.get_instance(idx, split_type=split_type)
            print(f"instance-------------- {instance}")
            text = instance['text']
            target_label = instance['label']

            # Compute saliency scores
            explanation = self.explainer.compute_feature_importance(text, target=target_label)

            # Store text, tokens, and scores
            saliency_data.append({
                "text": text,
                "label": target_label,
                "tokens": explanation.tokens,
                "saliency_scores": explanation.scores.tolist()  # Convert numpy array to list
            })
        explanations = self.lp_normalize_saliency_data(saliency_data)
        #Save to file
        with open(save_path, "w") as f:
            json.dump(saliency_data, f)
        print(f"Saliency scores saved to {save_path}")

    def load_scores(self, file_path):
        """
        Loads saliency scores from a file.
        
        Parameters:
            file_path: Path to the file containing saliency scores.
            
        Returns:
            List of dictionaries with text, tokens, and saliency scores.
        """
        with open(file_path, "r") as f:
            saliency_data = json.load(f)
        return saliency_data
