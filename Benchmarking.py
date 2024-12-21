import os
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from explainers.saliencymanager import SaliencyScoreManager
from explainers import IntegratedGradientsExplainer, InputXGradientExplainer
from dataset_loaders import MovieReviews, HateXplain
from evaluators.softsufficiency import compute_all_soft_ns
from evaluators.FAD import compute_all_fad
from evaluators.monotonicity import compute_all_monotonicity
from evaluators.IOU import compute_all_IOU
from evaluators.sensitivity import compute_all_sensitivity
from evaluators.AUPRC import evaluate
# from evaluators.recursiveROAR import compute_all_ROAR
import datasets
from typing import Dict, List, Optional, Union
import json
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import warnings
from itertools import product
import numpy as np
# Map long model names to shorter names
MODEL_NAME_MAPPING = {
    "cardiffnlp/twitter-xlm-roberta-base-sentiment": "roberta",
    # "Hate-speech-CNERG/bert-base-uncased-hatexplain": "bert",
    # "xlnet-base-cased": "xlnet",
}

class ExplainerEvaluator:
    def __init__(self, model_name: str, num_labels=3, device=None):
        """
        Initializes the ExplainerEvaluator with model, tokenizer, and device.
        Args:
            model_name (str): Pretrained model name to load.
            num_labels (int): Number of output labels for the classification task.
            device (str or None): Device for model computation ("cuda" or "cpu").
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self.initialize_model_and_tokenizer()

    def initialize_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        return model, tokenizer
    
    def load_dataset(self, dataset_name: str, **kwargs):
        """
        Load a dataset given its name.
        """
        if dataset_name == "HateXplain":
            data = HateXplain(self.tokenizer, **kwargs)
            num_labels = 3  # HateXplain is a binary classification dataset
        elif dataset_name == "MovieReviews":
            data = MovieReviews(self.tokenizer, **kwargs)
            num_labels = 2  # MovieReviews is typically a binary classification dataset
        else:
            try:
                data = datasets.load_dataset(dataset_name)
                num_labels = len(set(example['label'] for example in data['train']))
            except:
                raise ValueError(f"Dataset {dataset_name} is not supported.")
        return data, num_labels

    def load_or_compute_saliency_scores(self, manager, dataset, saliency_file_path):
        """
        Load or compute saliency scores for the given dataset.
        """
        if not os.path.exists(saliency_file_path):
            print("Saliency scores not found. Computing scores...")
            manager.compute_and_save_scores(dataset, saliency_file_path, split_type="test")
        print("Saliency scores found. Loading scores...")
        precomputed_scores = manager.load_scores(saliency_file_path)
        return precomputed_scores


    def compute_metrics(self, model_name, dataset_name, explainer, overwrite=False):
        """Compute metrics like ROAR."""
        dataset, num_label = self.load_dataset(dataset_name)
        saliency_file_path = f"saliency_scores/{model_name}_{explainer.NAME}_{dataset_name}.json"
        print(f"saliency file path {saliency_file_path}")
        #Create the Saliency Manager for generating explanations
        manager = SaliencyScoreManager(
            model=self.model,
            tokenizer=self.tokenizer,
            explainer_class=explainer,
            device=self.device
        )
        #Load or compute saliency scores
        saliency_scores = self.load_or_compute_saliency_scores(manager, dataset, saliency_file_path)

        #Compute ROAR
        roar_score = compute_all_ROAR(dataset, self.model, self.tokenizer, saliency_scores)
    
        IOU_F1 = compute_all_IOU(dataset, saliency_scores)

        return {"ROAR": roar_score, "IOU": IOU_F1['IOU'], "F1":IOU_F1['F1']}

def compute_and_save_results(evaluator, model_name, dataset_name, explainer, overwrite=False):
        """
        Compute metrics and save the results to a file if they don't already exist.

        Args:
            evaluator (ExplainerEvaluator): The evaluator instance.
            model_name (str): Model name.
            dataset_name (str): Dataset name.
            explainer (object): Explainer instance.
            overwrite (bool): Whether to overwrite existing results.

        Returns:
            dict: Computed metrics.
        """
        # Define the results file path
        results_file = f"results/{model_name}_{explainer.NAME}_{dataset_name}.json"

        # Check if results already exist
        if os.path.exists(results_file) and not overwrite:
            print(f"Results already exist for {model_name}, {dataset_name}, {explainer.NAME}. Loading results...")
            with open(results_file, "r") as file:
                metrics = json.load(file)
        else:
            metrics = {}

         # Define the full list of metrics to compute
        all_metrics = ["ROAR", "IOU", "F1"]

        # Identify missing metrics
        missing_metrics = [metric for metric in all_metrics if metric not in metrics]

        # Compute missing metrics
        if missing_metrics:
            print(f"Computing missing metrics: {missing_metrics}")
            new_metrics = evaluator.compute_metrics(model_name, dataset_name, explainer, overwrite)

        # Filter and add only the missing metrics to the results
            for metric in missing_metrics:
                # Convert numpy types to Python native types
                if isinstance(new_metrics.get(metric), (np.float32, np.float64)):
                    metrics[metric] = float(new_metrics[metric])
                else:
                    metrics[metric] = new_metrics.get(metric, None)

            # Save updated results
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            with open(results_file, "w") as file:
                json.dump(metrics, file, indent=4)
            print(f"Updated results saved to {results_file}")
        else:
            print(f"All metrics already computed for {model_name}, {dataset_name}, {explainer.NAME}.")

        return metrics


def benchmark(models: dict, datasets: list, explainers: list, overwrite=False):
    """Run benchmarks for all combinations of models, explainers, and datasets."""
    results = []
    for model_name, short_name in models.items():
        evaluator = ExplainerEvaluator(model_name=model_name)
        for dataset_name, explainer in product(datasets, explainers):
            metrics = compute_and_save_results(
                evaluator, short_name, dataset_name, explainer, overwrite
            )
            results.append({
                "Model": short_name,
                "Dataset": dataset_name,
                "Explainer": explainer.NAME,
                **metrics
            })
    return pd.DataFrame(results)

def main():
    # Define models, datasets, and explainers
    models = MODEL_NAME_MAPPING
    datasets = ["HateXplain"]
    explainers = [IntegratedGradientsExplainer]
    results_df = benchmark(models, datasets, explainers, overwrite=False)

    print("\nBenchmark Results:")
    print(results_df)

if __name__ == "__main__":
    main()
