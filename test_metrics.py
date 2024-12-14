import os
import torch
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
)
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
from evaluators.recursiveROAR import compute_all_ROAR
import datasets
from typing import Dict, List, Optional, Union
import json
import itertools
import warnings

class ExplainerEvaluator:
    def __init__(self, model_name="bert-base-uncased", num_labels=3, device=None,explainers=None, evaluators=None):
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
        self.explainers = explainers or self._default_explainers()
        self.evaluators = evaluators or self._default_evaluators()


    def initialize_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to('cpu')
        
        return model, tokenizer

        # """Initialize the model and tokenizer dynamically based on the model name."""
        # if "roberta" in self.model_name:
        #     model = RobertaForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        #     tokenizer = RobertaTokenizerFast.from_pretrained(self.model_name, clean_up_tokenization_spaces=True,add_prefix_space=True)
        # elif "bert" in self.model_name:
        #     model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        #     tokenizer = BertTokenizerFast.from_pretrained(self.model_name, clean_up_tokenization_spaces=True)
        # else:
        #     raise ValueError(f"Model {self.model_name} is not supported.")
        
        # model.to(self.device)  # Move model to the correct device (GPU/CPU)
        # return model, tokenizer

    def _default_explainers(self):
        """Return a default list of explainers."""
        return [
            InputXGradientExplainer(self.model, self.tokenizer, multiply_by_inputs=True),
            IntegratedGradientsExplainer(self.model, self.tokenizer, multiply_by_inputs=False),
        ]

    def _default_evaluators(self):
        """Return a default list of evaluators."""
        return [
            compute_all_fad,
            compute_all_monotonicity,
            compute_all_sensitivity,
            compute_all_IOU
        ]
    
    def load_dataset(self, dataset_name: str, **kwargs):
        """
        Load a dataset given its name.

        Args:
            dataset_name (str): Name of the dataset to load (e.g., "HateXplain").
            **kwargs: Additional arguments to pass to dataset loaders.

        Returns:
            dataset: Loaded dataset instance.
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
                # Assuming we can infer num_labels from the dataset
                num_labels = len(set(example['label'] for example in data['train']))
            except:
                raise ValueError(f"Dataset {dataset_name} is not supported.")

        return data, num_labels

    def load_or_compute_saliency_scores(self, manager, dataset, saliency_file_path):
        """
        Load or compute saliency scores for the given dataset.

        Args:
            manager (SaliencyScoreManager): The manager to compute saliency scores.
            dataset: Dataset instance for which saliency scores are calculated.
            saliency_file_path (str): Path to the file where saliency scores are saved.

        Returns:
            precomputed_scores: The precomputed saliency scores.
        """
        if not os.path.exists(saliency_file_path):
            print("Saliency scores not found. Computing scores...")
            manager.compute_and_save_scores(dataset, saliency_file_path, split_type="test")
        print("Saliency scores found. Loading scores...")
        precomputed_scores = manager.load_scores(saliency_file_path)
        # print(f'precomputed_Scores------------------{precomputed_scores[:2]}')
        return precomputed_scores

    def compute_metrics(self, dataset, precomputed_scores):
        """Compute evaluation metrics for the dataset."""
        #----------------------------FAD N-AUC-------------------------------------------------------
        # fad_n_auc = compute_all_fad(dataset, self.model, self.tokenizer, precomputed_scores)
        # print(f"Final FAD N-AUC: {fad_n_auc}")
        
        #----------------------------SOFT SUFFICIENCY---------------------------------------------------
        # normalized_sufficiency_scores = compute_all_soft_ns(dataset, self.model, self.tokenizer, precomputed_scores)
        # print(f"Normalized Soft Sufficiency Scores: {normalized_sufficiency_scores}")

        #----------------------------IOU and F1 Score---------------------------------------------------
        # IOU_F1 = compute_all_IOU(dataset, precomputed_scores)
        # print(f"Final IOU and F1 Score: {IOU_F1}")

        #----------------------------Sensitivity---------------------------------------------------
        # sensitivity_score= compute_all_sensitivity(self.model, self.tokenizer, dataset, precomputed_scores)
        # print(f"Final Sensitivity: {sensitivity_score}")

        #----------------------------MONOTONICITY---------------------------------------------------
        # average_monotonicity = compute_all_monotonicity(model, tokenizer, precomputed_scores)
        # print(f"Final monotonicity: {average_monotonicity}")

        #----------------------------AUPRC---------------------------------------------------
        # auprc_score= evaluate(dataset,precomputed_scores)
        # print(f"AUPRC score: {auprc_score}")

        #----------------------------Recursive ROAR---------------------------------------------------
        roar_score= compute_all_ROAR(dataset, self.model, self.tokenizer, precomputed_scores)
        print(f"ROAR score: {roar_score}")

               
    def run_explanations(self, dataset_name,explainer, overwrite=False):
        """
        Run and generate explanations for entire dataset and write to json file.

        Args:
            dataset_name (str): Name of the dataset to use for evaluation.
            file_path (str): name of file to write explanations output to.
        """
        saliency_scores_file_path='saliency_scores/'
        metrics_results_file_path= 'output_files/'
        # Get the explainer name using the NAME attribute
        explainer_name = getattr(explainer, "NAME", explainer.__class__.__name__) 
        #Create a dynamic file name based on model, explainer, and dataset
        results_file = os.path.join(metrics_results_file_path, f"{self.model_name}_{explainer_name}_{dataset_name}_results.json")
        print(f"Running explanations for {dataset_name} using {explainer_name}...")
       
        #If results file exists and overwrite is False, load and return existing results
        if os.path.exists(results_file) and not overwrite:
            with open(results_file, "r") as f:
                results = json.load(f)
            print(f"Loaded existing results from {results_file}")
            return results
        # Load dataset
        dataset,num_labels = self.load_dataset(dataset_name)

        # Create the Saliency Manager for generating explanations
        manager = SaliencyScoreManager(
            model=self.model,
            tokenizer=self.tokenizer,
            explainer_class=explainer,
            device=self.device
        )
       
        # Generate a dynamic file path based on the model, dataset, and explainer
        file_name = f"{self.model_name}_{explainer_name}_{dataset_name}_saliency_scores.json"
        saliency_file = os.path.join(saliency_scores_file_path, file_name)

        precomputed_scores = self.load_or_compute_saliency_scores(manager, dataset, saliency_file)
        #Compute the evaluation metrics
        results = self.compute_metrics(dataset, precomputed_scores)
        # print(f'resultsss-----{results}')
        #Save the results to a JSON file
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Results saved to {results_file}")
        return results

    def benchmark_all(self, datasets: List[str], explainers: List[object], models: List[str], overwrite=True):
        """
        Benchmark all combinations of models, explainers, and datasets.
        """
        
        for model_name, explainer, dataset_name in itertools.product(models, explainers, datasets):
            evaluator = ExplainerEvaluator(model_name=model_name, num_labels=3)
        evaluator.run_explanations(dataset_name=dataset_name, explainer=explainer, overwrite=overwrite)

def main():

    evaluator = ExplainerEvaluator(model_name="bert-base-uncased", num_labels=3)
    
    # Define the datasets, explainers, and models you want to use for benchmarking
    # datasets = ["HateXplain",]
    datasets = ["MovieReviews"]  # Add more datasets here if needed
    explainers = [IntegratedGradientsExplainer]
    # explainers =[InputXGradientExplainer,IntegratedGradientsExplainer]  # Add more explainers here if needed
    models = ["roberta-base"]
        # "bert-base-uncased"]
            # "bert-base-uncased", roberta-base"
    # Run benchmarking for all combinations
    evaluator.benchmark_all(datasets=datasets, explainers=explainers, models=models)


if __name__ == "__main__":
    main()
