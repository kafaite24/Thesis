# import torch
# import numpy as np
# import torch.nn.functional as F
# from transformers import BertTokenizerFast, BertForSequenceClassification
# import torch
# from explainers import GradientExplainer
# import os
# from dataset_loaders import MovieReviews
# from explainers.saliencymanager import SaliencyScoreManager

# # class SoftNC:
# #     def __init__(self, model, tokenizer, max_len, importance_scores, device='cpu', normalise=True):
# #         self.model = model.to(device)
# #         self.tokenizer = tokenizer
# #         self.max_len = max_len
# #         self.importance_scores = importance_scores
# #         self.device = device
# #         self.normalise = normalise

# #     def bernoulli_perturbation(self, embeddings, importance_scores):
# #         """
# #         Applies Bernoulli perturbation based on importance scores.
# #         """
# #         batch_size, seq_len, embed_dim = embeddings.size()  # Get dimensions of embeddings
        
# #         # Ensure that importance_scores is of shape (batch_size, seq_len)
# #         importance_scores = importance_scores.unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)
        
# #         # Apply Bernoulli perturbation: p = 1 - importance_score
# #         q = 1 - importance_scores  # p (perturbation probability) = 1 - importance_score
# #         mask = torch.bernoulli(q).to(embeddings.device)  # Shape: (batch_size, seq_len, 1)

# #         # Adjust mask size to match embeddings (batch_size, seq_len, embed_dim)
# #         mask = mask.expand(-1, -1, embed_dim)  # Shape: (batch_size, seq_len, embed_dim)

# #         # Ensure mask sequence length matches the embeddings length
# #         if mask.size(1) != embeddings.size(1):
# #             mask = mask[:, :embeddings.size(1), :]  # Trim or pad mask to match the embeddings' sequence length
        
# #         print(embeddings.shape)
# #         print(mask.shape)
# #         # Apply the mask to the embeddings
# #         perturbed_embeddings = embeddings * mask  # Element-wise multiplication
        
# #         return perturbed_embeddings

# #     def normalize_importance_scores(self, importance_scores):
# #         """
# #         Normalize importance scores to range [0, 1].
# #         """
# #         if self.normalise:
# #             min_value = importance_scores.min()
# #             max_value = importance_scores.max()
# #             importance_scores = (importance_scores - min_value) / (max_value - min_value)
# #         return importance_scores

# #     def compute(self, original_sentences, rows, full_text_probs, full_text_class, suff_y_zero, importance_scores):
# #         """
# #         Computes the Soft Normalized Comprehensiveness for the given sentences.
# #         """
# #         inputs = self.tokenizer(original_sentences, padding=True, truncation=True, 
# #                                 max_length=self.max_len, return_tensors="pt").to(self.device)

# #         # Normalize importance scores
# #         normalized_importance_scores = self.normalize_importance_scores(importance_scores)
        
# #         # Ensure importance_scores are padded or truncated to max_len
# #         if normalized_importance_scores.size(1) < self.max_len:
# #             padding_len = self.max_len - normalized_importance_scores.size(1)
# #             padding = torch.zeros(normalized_importance_scores.size(0), padding_len).to(self.device)
# #             normalized_importance_scores = torch.cat((normalized_importance_scores, padding), dim=1)
# #         elif normalized_importance_scores.size(1) > self.max_len:
# #             normalized_importance_scores = normalized_importance_scores[:, :self.max_len]

# #         # Run model on the original input
# #         original_output = self.model(**inputs)
# #         original_probs = torch.softmax(original_output.logits, dim=-1).detach().cpu().numpy()

# #         # Get reduced probabilities for correct classes
# #         reduced_probs = original_probs[rows, full_text_class]
        
# #         # Apply Bernoulli perturbation to the embeddings based on importance scores
# #         with torch.no_grad():
# #             embeddings = self.model.bert(**inputs).last_hidden_state  # Get embeddings from BERT
# #             perturbed_embeddings = self.bernoulli_perturbation(embeddings, normalized_importance_scores)

# #         # Recompute the model's predictions on the perturbed input
# #         perturbed_inputs = inputs.copy()
# #         perturbed_inputs["input_ids"] = perturbed_embeddings.argmax(dim=-1)  # Convert embeddings to token IDs
# #         perturbed_output = self.model(**perturbed_inputs)
# #         perturbed_probs = torch.softmax(perturbed_output.logits, dim=-1).detach().cpu().numpy()
        
# #         # Compute comprehensiveness score
# #         comp_y_a = self.compute_comprehensiveness(full_text_probs, reduced_probs)

# #         # Compute Soft Normalized Comprehensiveness
# #         suff_y_zero = np.clip(suff_y_zero, a_min=0, a_max=1)
# #         norm_comp = np.maximum(0, comp_y_a / (1 - suff_y_zero))  # Normalize by baseline sufficiency

# #         norm_comp = np.clip(norm_comp, a_min=0, a_max=1)

# #         return norm_comp, original_probs

# #     def compute_comprehensiveness(self, full_text_probs, reduced_probs):
# #         """
# #         Computes the comprehensiveness score.
# #         """
# #         return np.abs(full_text_probs - reduced_probs)  # Example: absolute difference in probabilities

# class SoftNC:
#     def __init__(self, model, tokenizer, max_len, importance_scores, device='cpu'):
#         """
#         Initializes the Soft Normalized Comprehensiveness computation.
        
#         Args:
#             model (nn.Module): The pre-trained model (e.g., BERT).
#             tokenizer (transformers.Tokenizer): Tokenizer used to tokenize text inputs.
#             max_len (int): Maximum token length to which the input should be padded/truncated.
#             importance_scores (torch.Tensor): Importance scores for tokens (shape: batch_size, seq_len).
#             device (str): Device to use for computation ('cuda' or 'cpu').
#         """
#         self.model = model.to(device)
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.importance_scores = importance_scores  # Ensure importance scores are on the correct device
#         self.device = device

#     def soft_perturb(self, embeddings, importance_scores, attention_mask):
#         """
#         Applies soft perturbation to the token embeddings based on the importance scores.
        
#         Args:
#             embeddings (torch.Tensor): The token embeddings (batch_size, seq_len, embedding_dim).
#             importance_scores (torch.Tensor): Importance scores for each token in the sequence (batch_size, seq_len).
#             attention_mask (torch.Tensor): Attention mask to indicate the padding positions (batch_size, seq_len).
        
#         Returns:
#             torch.Tensor: Perturbed token embeddings.
#         """
#         batch_size, seq_len, embed_dim = embeddings.size()  # Get dimensions of embeddings
#         print('importance scores in soft perturb---', importance_scores.shape)
#         # # Ensure that importance_scores is of shape (batch_size, seq_len)
#         importance_scores = importance_scores.unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)
        
#         # Apply mask to ignore padding tokens during perturbation
#         attention_mask = attention_mask.unsqueeze(-1).float()  # Shape: (batch_size, seq_len, 1)

#         if importance_scores.size(1) != attention_mask.size(1):
#             # Pad or truncate importance_scores to match attention_mask size
#             padding_len = attention_mask.size(1) - importance_scores.size(1)
#             if padding_len > 0:
#                 # Padding the importance scores
#                 padding = torch.zeros(importance_scores.size(0), padding_len, 1).to(self.device)
#                 importance_scores = torch.cat((importance_scores, padding), dim=1)
#             elif padding_len < 0:
#                 # Truncating importance scores
#                 importance_scores = importance_scores[:, :attention_mask.size(1), :]
        
#         # Apply Bernoulli perturbation: p = 1 - importance_score
#         q = 1 - importance_scores  # p (perturbation probability) = 1 - importance_score
#         mask = torch.bernoulli(q).to(embeddings.device)  # Shape: (batch_size, seq_len, 1)
        
#         # Apply the attention mask (this will zero out the padding tokens)
#         mask = mask * attention_mask  # Shape: (batch_size, seq_len, 1)
        
#         perturbed_embeddings = embeddings * mask
#         print('perturbed embeddings---------', perturbed_embeddings.shape)
#         return perturbed_embeddings

#     def compute_comprehensiveness(self, original_input, perturbed_input):
#         """
#         Computes the soft comprehensiveness score based on the change in model predictions.

#         Args:
#             original_input (dict): The input dictionary for the model with original tokens.
#             perturbed_input (dict): The input dictionary for the model with perturbed tokens.
        
#         Returns:
#             float: The computed comprehensiveness score.
#         """
#         # Get model prediction on original input
#         original_output = self.model(**original_input)
#         original_probs = F.softmax(original_output.logits, dim=-1).detach().cpu().numpy()
#         print('original prob',original_probs)
#         # Process data in smaller batches
#         batch_size = 8  # or any other reasonable batch size
#         for i in range(0, len(perturbed_input['input_ids']), batch_size):
#             print(i)
#             batch_input = {key: val[i:i+batch_size] for key, val in perturbed_input.items()}
        
#             with torch.no_grad():
#                 perturbed_output = self.model(**batch_input)
#                 perturbed_probs = F.softmax(perturbed_output.logits, dim=-1).detach().cpu().numpy()
#         # Get model prediction on perturbed input
#         print('perturbed prob',perturbed_probs)  
#         # Assuming we are doing classification, calculate the difference for the correct class
#         original_pred = np.argmax(original_probs, axis=-1)
#         perturbed_pred = np.argmax(perturbed_probs, axis=-1)
        
#         print('original pred',original_pred)
#         print('perturbed pred',perturbed_pred)        # For comprehensiveness, we compute the change in probability
#         comprehensiveness = max(0, original_probs[0, original_pred[0]] - perturbed_probs[0, perturbed_pred[0]])
#         print('comprehensiveness',comprehensiveness)
#         return comprehensiveness

#     def normalize_comprehensiveness(self, comprehensiveness, baseline_comprehensiveness):
#         """
#         Normalizes the comprehensiveness score to the range [0, 1].

#         Args:
#             comprehensiveness (float): The raw comprehensiveness score.
#             baseline_comprehensiveness (float): The baseline comprehensiveness score (when no perturbation).
        
#         Returns:
#             float: The normalized comprehensiveness score.
#         """
#         normalized_comprehensiveness = (comprehensiveness - baseline_comprehensiveness) / (1 - baseline_comprehensiveness)
#         normalized_comprehensiveness = np.clip(normalized_comprehensiveness, 0, 1)  # Ensure it is between 0 and 1
        
#         return normalized_comprehensiveness


#     def compute(self, original_sentences, batch_size=1):
#         """
#         Computes Soft Normalized Comprehensiveness for the given input sentences.
        
#         Args:
#             original_sentences (list or torch.Tensor): List of raw sentences.
#             batch_size (int): Number of sentences to process in each batch.
        
#         Returns:
#             tuple: The normalized comprehensiveness scores and model predictions.
#         """
#         # Tokenize the sentences
#         inputs = self.tokenizer(original_sentences, padding=True, truncation=True, 
#                                 max_length=512, return_tensors="pt").to(self.device)
#         # Get model predictions on original input
#         original_input = inputs
#         original_output = self.model(**original_input)
#         original_probs = F.softmax(original_output.logits, dim=-1).detach().cpu().numpy()

#         # Get the baseline comprehensiveness (before any perturbation)
#         baseline_comprehensiveness = original_probs[0, np.argmax(original_probs[0])]
#         print('baseline_comprehensivess', baseline_comprehensiveness)
#         # Get embeddings (using the model's outputs)
#         with torch.no_grad():
#             outputs = self.model.bert(**original_input)  # Get outputs from the BERT model
#             embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_len, embed_dim)
        
#         # Apply soft perturbation based on importance scores
#         perturbed_embeddings = self.soft_perturb(embeddings, self.importance_scores, inputs['attention_mask'])
#         # Create a perturbed input dictionary (copy the original input and update input_ids)
#         perturbed_input = inputs.copy()
#         perturbed_input['input_ids'] = perturbed_embeddings.argmax(dim=-1)  # Convert embeddings to token IDs for input
        
#         # Compute the comprehensiveness score based on the perturbed input
#         comprehensiveness = self.compute_comprehensiveness(original_input, perturbed_input)
#         print('comprehensiveness----------', comprehensiveness)
#         # Normalize the comprehensiveness score
#         normalized_comprehensiveness = self.normalize_comprehensiveness(comprehensiveness, baseline_comprehensiveness)
        
#         return [normalized_comprehensiveness], original_probs
    
# def normalize_importance_scores(importance_scores):
#         """
#         Normalizes the importance scores to the range [0, 1].
        
#         Args:
#             importance_scores (torch.Tensor): The tensor of importance scores to normalize.
            
#         Returns:
#             torch.Tensor: The normalized importance scores.
#         """
#         min_val = importance_scores.min()  # Find the minimum value
#         max_val = importance_scores.max()  # Find the maximum value
        
#         # Apply min-max normalization
#         normalized_scores = (importance_scores - min_val) / (max_val - min_val)
        
#         return normalized_scores
# # Example usage of SoftNC class
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# dataset = MovieReviews(tokenizer)

# saliency_file_path = "data/saliency_scores.json"

# # Initialize Saliency Manager
# manager = SaliencyScoreManager(
#     model=model,
#     tokenizer=tokenizer,
#     explainer_class=GradientExplainer,
#     device="cuda" if torch.cuda.is_available() else "cpu"
# )
# # max_len = 512
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Dummy importance scores for testing
# # importance_scores = torch.rand(1, 9)  # Random importance scores between 0 and 1

# # # Initialize SoftNC with the model, tokenizer, and importance scores
# # max_len = 9  # Example sequence length
# # soft_nc = SoftNC(model, tokenizer, max_len, importance_scores, device='cpu')

# # # Dummy input sentences (batch of sentences)
# # original_sentences = ["This is a test sentence."]

# # # Compute Soft Normalized Comprehensiveness
# # normalized_comprehensiveness, original_probs = soft_nc.compute(original_sentences)

# # # Print results
# # print("Normalized Comprehensiveness:", normalized_comprehensiveness)
# # print("Original Model Probabilities:", original_probs)

# original_sentences= dataset.get_review(0, split_type="test")['text']
# # print(original_sentences)
# # Initialize SoftNC with the model, tokenizer, and importance scores

# if not os.path.exists(saliency_file_path):
#     manager.compute_and_save_scores(dataset, saliency_file_path, split_type="test")
# precomputed_scores = manager.load_scores(saliency_file_path)
# importance_scores= precomputed_scores[0]['saliency_scores']
# importance_scores = torch.tensor(importance_scores).to(model.device)
# importance_scores = normalize_importance_scores(importance_scores)
# importance_scores = importance_scores.unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)
# print('importance scores in main---', importance_scores.shape)
# max_len = 512  # Example sequence length

# soft_nc = SoftNC(model, tokenizer, max_len, importance_scores, device='cpu')

# # Compute Soft Normalized Comprehensiveness
# norm_comp, yhat = soft_nc.compute(original_sentences)
# print("Normalized Comprehensiveness:", norm_comp)
# print("Model Predictions:", yhat)

# # # Compute or load saliency scores
# # if not os.path.exists(saliency_file_path):
# #     manager.compute_and_save_scores(dataset, saliency_file_path, split_type="test")
# # precomputed_scores = manager.load_scores(saliency_file_path)
# # importance_scores= precomputed_scores[0]['saliency_scores']

# # soft_nc = SoftNC(model, tokenizer, max_len, importance_scores, device)
# # # Compute Soft Normalized Comprehensiveness
# # norm_comp, yhat = soft_nc.compute(original_sentences)
# # print("Normalized Comprehensiveness:", norm_comp)
# # print("Model Predictions:", yhat)

# # # Compute Soft Normalized Sufficiency for all samples
# # normalized_sufficiency_scores = compute_all_soft_ns(dataset, model, tokenizer, precomputed_scores)

# # # Example of accessing the results
# # print(f"Normalized Sufficiency Scores for all samples: {normalized_sufficiency_scores}")





# # max_len = 512
# # importance_scores = torch.rand(1, max_len)  # Example importance scores
# # device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # soft_nc = SoftNC(model, tokenizer, max_len, importance_scores, device)

# # # Example inputs
# # original_sentences = ["The cat sat on the mat."]
# # rows = np.array([0])
# # full_text_probs = np.random.rand(1, 2)  # Example probabilities
# # full_text_class = np.array([1])  # Example predicted class
# # suff_y_zero = np.array([0.5])  # Example sufficiency scores

# # # Compute Soft Normalized Comprehensiveness
# # norm_comp, yhat = soft_nc.compute(original_sentences, rows, full_text_probs, full_text_class, suff_y_zero, importance_scores)
# # print("Normalized Comprehensiveness:", norm_comp)
# # print("Model Predictions:", yhat)




import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertTokenizerFast, BertForSequenceClassification
import numpy as np
from explainers.saliencymanager import SaliencyScoreManager
from explainers import InputXGradientExplainer, IntegratedGradientsExplainer
import os
from dataset_loaders import MovieReviews

class SoftNS:
    def __init__(self, model, tokenizer, max_len, importance_scores, device='cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.importance_scores = importance_scores.to(device)
        self.device = device

    def soft_perturb(self, embeddings, importance_scores, attention_mask):
        # Softens importance scores, applies mask, and perturbs embeddings
        # print(f"importance scores before {importance_scores}")
        importance_scores = torch.sigmoid(importance_scores).unsqueeze(-1)
        attention_mask = attention_mask.unsqueeze(-1).float()
        print(f"embeddings before {embeddings}")

        # Apply padding mask if necessary
        if importance_scores.size(1) != attention_mask.size(1):
            padding_len = attention_mask.size(1) - importance_scores.size(1)
            importance_scores = torch.cat([importance_scores, torch.zeros(importance_scores.size(0), padding_len, 1).to(self.device)], dim=1)
        # print(f"importance scores {importance_scores}")
        mask = torch.bernoulli(importance_scores).to(embeddings.device) * attention_mask
        perturbed_embeddings= embeddings * mask
        print(f"mask {mask}")
        print(f"embeddings after {perturbed_embeddings}")
        return perturbed_embeddings

    def compute_sufficiency(self, original_input, perturbed_input):
        # Compute the sufficiency score by comparing the model's predictions
        
        original_probs = torch.softmax(self.model(**original_input).logits, dim=-1).detach().cpu().numpy()
        perturbed_probs = torch.softmax(self.model(**perturbed_input).logits, dim=-1).detach().cpu().numpy()
        print(f"original prob {original_probs}")
        print(f"perturbed prob {perturbed_probs}")
        original_pred = np.argmax(original_probs, axis=-1)
        perturbed_pred = np.argmax(perturbed_probs, axis=-1)
        print(f"original pred {original_pred}")
        print(f"perturbed pred {perturbed_pred}")
        sufficiency = 1 - np.maximum(0, original_probs[np.arange(original_probs.shape[0]), original_pred] - perturbed_probs[np.arange(perturbed_probs.shape[0]), perturbed_pred])
        print(f"sufficiency {sufficiency}")
        return sufficiency

    def normalize_sufficiency(self, sufficiency, baseline_sufficiency):
        # Normalize sufficiency to the range [0, 1]
        return np.clip((sufficiency - baseline_sufficiency) / (1 - baseline_sufficiency), 0, 1)

    def compute(self, original_sentences, batch_size=8):
        inputs = self.tokenizer(original_sentences, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)

        original_output = self.model(**inputs)
        original_probs = torch.softmax(original_output.logits, dim=-1).detach().cpu().numpy()
        baseline_sufficiency = 1 - np.maximum(0, original_probs[0, np.argmax(original_probs[0])])

        with torch.no_grad():
            embeddings = self.model.bert(**inputs).last_hidden_state

        perturbed_embeddings = self.soft_perturb(embeddings, self.importance_scores, inputs['attention_mask'])

        perturbed_input = {key: value for key, value in inputs.items() if key != 'input_ids'}
        perturbed_input['inputs_embeds'] = perturbed_embeddings

        sufficiency = self.compute_sufficiency(inputs, perturbed_input)
        normalized_sufficiency = self.normalize_sufficiency(sufficiency, baseline_sufficiency)
        print(f"nornalized sufficiency {normalized_sufficiency}")
        return [normalized_sufficiency], original_probs

def compute_all_soft_ns(dataset, model, tokenizer, precomputed_scores, max_len=512, batch_size=8):
    all_normalized_sufficiency = []
    all_predictions = []

    num_samples = dataset.get_data_length(split_type="test")
    for i in range(0, num_samples, batch_size):
        print(f"for batch {i}")
        batch_samples = [dataset.get_review(j, split_type="test") for j in range(i, min(i + batch_size, num_samples))]
        original_sentences = [sample['text'] for sample in batch_samples]
        saliency_scores = [entry['saliency_scores'] for entry in precomputed_scores[i:i + batch_size]]
        saliency_scores = torch.stack([torch.tensor(score[:max_len] if len(score) > max_len else score + [0] * (max_len - len(score))) for score in saliency_scores]).to(model.device)
        print(f"saliency scores before normalization {saliency_scores}")
        # Normalize importance scores
        # Apply softmax to the importance scores to retain variation
        # importance_scores = torch.tensor(saliency_scores)  # Assuming saliency_scores is already a tensor
        # importance_scores = torch.sigmoid(saliency_scores)  # Apply softmax across tokens for each sample

        importance_scores = (saliency_scores - saliency_scores.min()) / (saliency_scores.max() - saliency_scores.min())
        print(f"saliency scores after normalization {importance_scores}")
        soft_ns = SoftNS(model, tokenizer, max_len, importance_scores)

        normalized_sufficiency, model_predictions = soft_ns.compute(original_sentences)
        
        all_normalized_sufficiency.extend(normalized_sufficiency)
        all_predictions.extend(model_predictions)

    return np.mean(all_normalized_sufficiency)

# Initialize model and tokenizer (e.g., BERT)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
dataset = MovieReviews(tokenizer)

saliency_file_path = "data/saliency_scores_IG_2.json"

# Initialize Saliency Manager
manager = SaliencyScoreManager(
    model=model,
    tokenizer=tokenizer,
    explainer_class=IntegratedGradientsExplainer,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Compute or load saliency scores
if not os.path.exists(saliency_file_path):
    manager.compute_and_save_scores(dataset, saliency_file_path, split_type="test")
precomputed_scores = manager.load_scores(saliency_file_path)

# Compute Soft Normalized Sufficiency for all samples
normalized_sufficiency_scores = compute_all_soft_ns(dataset, model, tokenizer, precomputed_scores)

print(f"Normalized Sufficiency Scores for all samples: {normalized_sufficiency_scores}")
