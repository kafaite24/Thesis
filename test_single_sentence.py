import torch
import json
import math
from typing import List, Optional, Tuple, Union
import numpy as np
from transformers.tokenization_utils_base import BatchEncoding
from tqdm.autonotebook import tqdm
from explanation import Explanation
import copy
import logging
from explainers.integratedgradient import IntegratedGradientsExplainer
from explainers.deeplift import DeepLiftExplainer
from explainers.gradientxinput import InputXGradientExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.visualization import show_table
import pandas as pd
from utils.text_helpers import SequenceClassificationHelper

class SimpleExplainerEvaluator:
    def __init__(self, model_name="", num_labels=3, device=None, explainers=None):
        """
        Initializes the evaluator with a pre-trained model and tokenizer.
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self.initialize_model_and_tokenizer()
        self.explainers = self._default_explainers()
        self.helper = SequenceClassificationHelper(self.model, self.tokenizer)

    def initialize_model_and_tokenizer(self):
        """Initialize the model and tokenizer dynamically based on the model name."""
        print(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to('cpu')
        
        return model, tokenizer

    def _default_explainers(self):
        """Return a default list of explainers."""
        return [DeepLiftExplainer(self.model, self.tokenizer),
        IntegratedGradientsExplainer(self.model, self.tokenizer),]
                # InputXGradientExplainer(self.model, self.tokenizer)]
    
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

    def explain(
        self,
        text,
        target=1,
        show_progress: bool = True,
        normalize_scores: bool = True,
        order: int = 1,
        target_token: Optional[str] = None,
        target_option: Optional[str] = None,
    ) -> List[Explanation]:
    
        text = self.helper._check_sample(text)
        explanations = list()
        for explainer in tqdm(
            self.explainers,
            total=len(self.explainers),
            desc="Explainer",
            leave=False,
            disable=not show_progress,
        ):
            exp = explainer.compute_feature_importance(
                text, target, target_token
            )
            
            explanations.append(exp)

        if normalize_scores:
            explanations = self.lp_normalize(explanations, order)
        print(explanations)
        return explanations
    
  
    def compute_score_single_sentence(self,text: str,return_dict: bool = True):

        _, logits = self.helper._forward(text, output_hidden_states=False)
        print(f"logits {logits}")
        scores = logits[0].softmax(-1)

        if return_dict:
            scores = {
                self.model.config.id2label[idx]: value.item()
                for idx, value in enumerate(scores)
            }
        return scores


    def show_table(
        self,
        explanations: List[Explanation],
        remove_first_last: bool = True,
        style: None = "heatmap",
    ) -> pd.DataFrame:
        return show_table(explanations, remove_first_last, style)