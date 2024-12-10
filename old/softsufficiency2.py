import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertTokenizerFast, BertForSequenceClassification
import numpy as np
from explainers.saliencymanager import SaliencyScoreManager
from explainers import InputXGradientExplainer
import os
from dataset_loaders import MovieReviews

class SoftSufficiencyEvaluator:
    def __init__(self, model, tokenizer, max_len, importance_scores, device='cpu'):
        """
        Initializes the Soft Normalized Sufficiency computation.
        
        Args:
            model (nn.Module): The pre-trained model (e.g., BERT).
            tokenizer (transformers.Tokenizer): Tokenizer used to tokenize text inputs.
            max_len (int): Maximum token length to which the input should be padded/truncated.
            importance_scores (torch.Tensor): Importance scores for tokens (shape: batch_size, seq_len).
            device (str): Device to use for computation ('cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.importance_scores = importance_scores  # Ensure importance scores are on the correct device
        self.device = device

    def soft_perturb(self, embeddings, importance_scores, attention_mask):
        """
        Applies soft perturbation to the token embeddings based on the importance scores.
        
        Args:
            embeddings (torch.Tensor): The token embeddings (batch_size, seq_len, embedding_dim).
            importance_scores (torch.Tensor): Importance scores for each token in the sequence (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask to indicate the padding positions (batch_size, seq_len).
        
        Returns:
            torch.Tensor: Perturbed token embeddings.
        """
        batch_size, seq_len, embed_dim = embeddings.size()  # Get dimensions of embeddings
        
        # Ensure that importance_scores is of shape (batch_size, seq_len)
        importance_scores = importance_scores.unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)
        
        # Apply mask to ignore padding tokens during perturbation
        attention_mask = attention_mask.unsqueeze(-1).float()  # Shape: (batch_size, seq_len, 1)

        if importance_scores.size(1) != attention_mask.size(1):
            # Pad or truncate importance_scores to match attention_mask size
            padding_len = attention_mask.size(1) - importance_scores.size(1)
            if padding_len > 0:
                # Padding the importance scores
                padding = torch.zeros(importance_scores.size(0), padding_len, 1).to(self.device)
                importance_scores = torch.cat((importance_scores, padding), dim=1)
            elif padding_len < 0:
                # Truncating importance scores
                importance_scores = importance_scores[:, :attention_mask.size(1), :]
        
        # Create a Bernoulli mask based on importance scores (probability of keeping each element)
        mask = torch.bernoulli(importance_scores).to(embeddings.device)  # Shape: (batch_size, seq_len, 1)
        
        # Apply the attention mask (this will zero out the padding tokens)
        mask = mask * attention_mask  # Shape: (batch_size, seq_len, 1)
       
        perturbed_embeddings = embeddings * mask

        return perturbed_embeddings

    def compute_sufficiency(self, original_input, perturbed_input):
        """
        Computes the soft sufficiency score based on the change in model predictions.

        Args:
            original_input (dict): The input dictionary for the model with original tokens.
            perturbed_input (dict): The input dictionary for the model with perturbed tokens.
        
        Returns:
            float: The computed sufficiency score.
        """
        # Get model prediction on original input
        original_output = self.model(**original_input)
        original_probs = F.softmax(original_output.logits, dim=-1).detach().cpu().numpy()
        
        # Get model prediction on perturbed input
        perturbed_output = self.model(**perturbed_input)
        perturbed_probs = F.softmax(perturbed_output.logits, dim=-1).detach().cpu().numpy()
        
        # Assuming we are doing classification, calculate the difference for the correct class
        original_pred = np.argmax(original_probs, axis=-1)
        perturbed_pred = np.argmax(perturbed_probs, axis=-1)
        
        # For sufficiency, we compute the drop in probability
        sufficiency = 1 - max(0, original_probs[0, original_pred[0]] - perturbed_probs[0, perturbed_pred[0]])
        
        return sufficiency

    def normalize_sufficiency(self, sufficiency, baseline_sufficiency):
        """
        Normalizes the sufficiency score to the range [0, 1].

        Args:
            sufficiency (float): The raw sufficiency score.
            baseline_sufficiency (float): The baseline sufficiency score (when no perturbation).
        
        Returns:
            float: The normalized sufficiency score.
        """
        normalized_suff = (sufficiency - baseline_sufficiency) / (1 - baseline_sufficiency)
        normalized_suff = np.clip(normalized_suff, 0, 1)  # Ensure it is between 0 and 1
        
        return normalized_suff

    def compute(self, original_sentences, batch_size=1):
        """
        Computes Soft Normalized Sufficiency for the given input sentences.
        
        Args:
            original_sentences (list or torch.Tensor): List of raw sentences.
            batch_size (int): Number of sentences to process in each batch.
        
        Returns:
            tuple: The normalized sufficiency scores and model predictions.
        """
        # Tokenize the sentences
        inputs = self.tokenizer(original_sentences, padding=True, truncation=True, 
                                max_length=self.max_len, return_tensors="pt").to(self.device)
        # Get model predictions on original input
        original_input = inputs
        original_output = self.model(**original_input)
        original_probs = F.softmax(original_output.logits, dim=-1).detach().cpu().numpy()

        # Get the baseline sufficiency (before any perturbation)
        baseline_sufficiency = 1 - max(0, original_probs[0, np.argmax(original_probs[0])])
        
        # Get embeddings (using the model's outputs)
        with torch.no_grad():
            outputs = self.model.bert(**original_input)  # Get outputs from the BERT model
            embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_len, embed_dim)
        
        # Apply soft perturbation based on importance scores
        perturbed_embeddings = self.soft_perturb(embeddings, self.importance_scores, inputs['attention_mask'])

        # Create a perturbed input dictionary (copy the original input and update input_ids)
        perturbed_input = inputs.copy()
        perturbed_input['input_ids'] = perturbed_embeddings.argmax(dim=-1)  # Convert embeddings to token IDs for input
        
        # Compute the sufficiency score based on the perturbed input
        sufficiency = self.compute_sufficiency(original_input, perturbed_input)

        # Normalize the sufficiency score
        normalized_sufficiency = self.normalize_sufficiency(sufficiency, baseline_sufficiency)
        
        return [normalized_sufficiency], original_probs
    
def compute_all_soft_ns(dataset, model, tokenizer, precomputed_scores, max_len=512, batch_size=8):
    """
    Computes Soft Normalized Sufficiency for all samples in the dataset.
    
    Args:
        dataset (MovieReviews): The dataset object.
        model (nn.Module): The pre-trained model (e.g., BERT).
        tokenizer (transformers.Tokenizer): Tokenizer used to tokenize text inputs.
        precomputed_scores (list): List of precomputed saliency scores for each sample.
        max_len (int): Maximum token length to which the input should be padded/truncated.
        batch_size (int): Number of sentences to process in each batch.
    
    Returns:
        list: List of normalized sufficiency scores for all samples.
        list: List of model predictions for all samples.
    """
    all_normalized_sufficiency = []
    all_predictions = []

    num_samples = dataset.get_data_length(split_type="test")

    # Iterate over the dataset in batches
    for i in range(0, num_samples, batch_size):
        print(i)
        batch_samples = [dataset.get_review(j, split_type="test") for j in range(i, min(i + batch_size, num_samples))]
        
        # Extract the text for each sample in the batch
        original_sentences = [sample['text'] for sample in batch_samples]
        
        # Extract the importance scores for each sample in the batch
        saliency_scores = [entry['saliency_scores'] for entry in precomputed_scores[i:i + batch_size]]
        
        # Pad the importance scores to the max_len (512)
        padded_saliency_scores = []
        for score in saliency_scores:
            if len(score) < max_len:
                # Pad with zeros (assuming no importance for padding tokens)
                padded_score = torch.cat([torch.tensor(score), torch.zeros(max_len - len(score))])
            else:
                # Truncate if the length exceeds max_len
                padded_score = torch.tensor(score[:max_len])
            padded_saliency_scores.append(padded_score)

        # Convert to a tensor and move to the correct device
        importance_scores = torch.stack(padded_saliency_scores).to(model.device)

        # Initialize SoftNS with the importance scores
        soft_ns = SoftSufficiencyEvaluator(model, tokenizer, max_len, importance_scores)

        # Compute Soft Normalized Sufficiency for the batch
        normalized_sufficiency, model_predictions = soft_ns.compute(original_sentences)
        print(f"normalized sufficiency for {i} is {normalized_sufficiency}")
        # Append results to the lists
        all_normalized_sufficiency.extend(normalized_sufficiency)
        all_predictions.extend(model_predictions)

    # Calculate the cumulative value (average) of sufficiency scores
    cumulative_sufficiency = np.mean(all_normalized_sufficiency)
    
    return cumulative_sufficiency
