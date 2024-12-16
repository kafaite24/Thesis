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

    def lp_normalize(self,explanations, ord=1):
        """Run Lp-noramlization of explanation attribution scores

        Args:
            explanations (List[Explanation]): list of explanations to normalize
            ord (int, optional): order of the norm. Defaults to 1.

        Returns:
            List[Explanation]: list of normalized explanations
        """


        new_exp = copy.copy(explanations)
        if isinstance(new_exp.scores, np.ndarray) and new_exp.scores.size > 0:
            norm_axis = (
                -1 if new_exp.scores.ndim == 1 else (0, 1)
            )  # handle axis correctly
            norm = np.linalg.norm(new_exp.scores, axis=norm_axis, ord=ord)
            if norm != 0:  # avoid division by zero
                new_exp.scores /= norm

        return new_exp

        # new_exps = list()
        # for exp in explanations:
        #     new_exp = copy.copy(exp)
        #     if isinstance(new_exp.scores, np.ndarray) and new_exp.scores.size > 0:
        #         norm_axis = (
        #             -1 if new_exp.scores.ndim == 1 else (0, 1)
        #         )  # handle axis correctly
        #         norm = np.linalg.norm(new_exp.scores, axis=norm_axis, ord=ord)
        #         if norm != 0:  # avoid division by zero
        #             new_exp.scores /= norm
        #     new_exps.append(new_exp)

        # return new_exps
    

    def compute_and_save_scores(self, dataset, save_path,order: int = 1, split_type="test"):
        """
        Computes saliency scores for a dataset and saves them to a file.
        
        Parameters:
            dataset: The dataset to compute scores for.
            save_path: Path to save the scores.
            split_type: The split of the dataset (e.g., "test", "train").
        """
        saliency_data = []
        explanations = list()
        print(f'dataset length {dataset.len()}')
    
        # Loop through the dataset
        for idx in tqdm(range(dataset.len())): #dataset.get_data_length()

            instance = dataset.get_instance(idx, split_type=split_type)
            text = instance['text']
            target_label = instance['label']
            print(f"target label {target_label}")
            # Compute saliency scores
            exp = self.explainer.compute_feature_importance(text, target=target_label)
            exp = self.lp_normalize(exp, order)

            # Store text, tokens, and scores
            saliency_data.append({
                "text": exp.text,
                "label": exp.target,
                "tokens": exp.tokens,
                "saliency_scores": exp.scores.tolist()  # Convert numpy array to list
            })
        
        print(f"saliency data {saliency_data}")
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
