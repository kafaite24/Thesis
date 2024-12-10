# # test_movie_reviews_loader.py
from explainers import GradientExplainer
from dataset_loaders import MovieReviews
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from evaluators import InfidelityEvaluator, SensitivityEvaluator, MonotonicityEvaluator
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.metrics import auc
import os
import argparse

parser = argparse.ArgumentParser()
# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()

# Load BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased").to(device)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


# Function to mask tokens based on their saliency scores
def mask_tokens_based_on_saliency(tokens, saliency_scores, threshold):
    """
    Masks tokens in the input text based on the given saliency scores and threshold.
    Tokens with saliency scores below the threshold are replaced with "[MASK]".
    """
    token_saliency_pairs = list(zip(tokens, saliency_scores))

    # Mask tokens with saliency scores below the threshold
    masked_tokens = [
        token if score >= threshold else "[MASK]"
        for token, score in token_saliency_pairs
    ]
    
    # Convert masked tokens back to a string
    masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
    return masked_text

# Faithfulness evaluation function
def evaluate_faithfulness(model, dataset, thresholds):
    """
    Evaluates the faithfulness of saliency scores using a range of thresholds.
    Computes the model's performance as tokens are progressively masked based on their saliency scores.
    Returns the faithfulness score (AUC) and model scores across thresholds.
    """
    explainer = GradientExplainer(model, tokenizer, device=device)
    model_scores = []
    dataset_length = dataset.get_data_length()
    print(dataset_length)
    for threshold in thresholds:
        thresholded_scores = []
        
        # Explicitly loop over dataset indices
        for idx in range(dataset_length):
            # Retrieve the review and label using `get_review`
            review = dataset.get_review(idx, split_type="test")
            text = review['text']
            target_label = review['label']

            # Compute saliency scores using the GradientExplainer
            explanation = explainer.compute_feature_importance(text, target=target_label)
            tokens = explanation.tokens
            saliency_scores = explanation.scores
            print(saliency_scores)
            # Mask tokens based on saliency scores
            masked_text = mask_tokens_based_on_saliency(tokens, saliency_scores, threshold)

            # Tokenize the masked text
            inputs = tokenizer(masked_text, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_label = torch.argmax(outputs.logits, dim=1).item()
            
            # Check if the prediction matches the true label
            thresholded_scores.append(1 if predicted_label == target_label else 0)
        
        # Calculate accuracy for the current threshold
        accuracy = np.mean(thresholded_scores)
        model_scores.append(accuracy)

    # Compute AUC for faithfulness score
    faithfulness_auc = auc(thresholds, model_scores)
    return faithfulness_auc, model_scores

# Main function to perform faithfulness evaluation
def main():
    # Initialize the MovieReviews dataset
    movie_reviews_dataset = MovieReviews(tokenizer)
  
    #Define thresholds for faithfulness evaluation
    thresholds = np.linspace(0, 1, num=20)  # 20 thresholds from 0.0 to 1.0

    # Run faithfulness evaluation
    faithfulness_score, model_scores = evaluate_faithfulness(
        model, movie_reviews_dataset, thresholds
    )

    # Print the results
    print(f"Faithfulness Score (AUC): {faithfulness_score}")
    print(f"Model scores across thresholds: {model_scores}")

if __name__ == "__main__":
    main()














# def test_movie_reviews():

#     # Initialize the model and tokenizer
#     model_name = "bert-base-uncased"
#     model = AutoModelForSequenceClassification.from_pretrained(model_name)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     # Initialize the MovieReviews dataset
#     movie_reviews_dataset = MovieReviews(tokenizer)

#     # Get a sample review from the test split of the dataset
#     sample_idx = 0  # You can select any index to test
#     review = movie_reviews_dataset.get_review(sample_idx, split_type="test")

#     # print("Review Text:", review["text"])
#     # print("Tokens:", review["tokens"])
#     # print("Label:", review["label"])

#     # Initialize the explainer
#     explainer = GradientExplainer(model, tokenizer)

#     # Text to analyze and target label
#     text = review['text']
#     target_label = review['label']

#     # Tokenize the input text and ensure the result is in LongTensor format
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)["input_ids"]
#     inputs = inputs.to(explainer.device).long()  # Ensure `inputs` is LongTensor
    
#     # Recompute feature importance using Gradient x Input for truncated input
#     explanation = explainer.compute_feature_importance(text, target=target_label)
    
#     # Convert explanation scores to tensor and truncate if needed
#     attributions = torch.tensor(explanation.scores).to(explainer.device)
#     if attributions.shape[0] > inputs.shape[1]:  # Trim attributions to match inputs
#         attributions = attributions[: inputs.shape[1]]
    
#     # Add a batch dimension to attributions to match the shape of inputs
#     attributions = attributions.unsqueeze(0)  # Shape now matches [1, 512]

#     # Initialize InfidelityCalculator and calculate infidelity score
#     infidelity_calculator = InfidelityEvaluator(model, target_label, device=explainer.device)
#     infidelity_score = infidelity_calculator.calculate_infidelity(inputs, attributions)

#     # Print the infidelity score
#     print("Infidelity Score:", infidelity_score)


#     # # Initialize SensitivityCalculator and calculate sensitivity score
#     # sensitivity_calculator = SensitivityEvaluator(model, explainer, target_label, device=explainer.device)
#     # sensitivity_score = sensitivity_calculator.calculate_sensitivity(inputs)
#     # print("Sensitivity Score:", sensitivity_score)

# if __name__ == "__main__":
#     test_movie_reviews()







#Monotonicity-----------------------------------------------------

# # Initialize the model and tokenizer
# model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# # Initialize the MovieReviews dataset
# movie_reviews_dataset = MovieReviews(tokenizer)

# # Instantiate the MonotonicityEvaluator
# evaluator = MonotonicityEvaluator(model_name=model_name)
# # Get a sample review from the test split of the dataset
# sample_idx = 6  # You can select any index to test
# review = movie_reviews_dataset.get_review(sample_idx, split_type="test")

# # Extract the base text from the review
# base_text = review["text"]

# # Define more contextual positive tokens
# positive_tokens = [
#     "a good movie", "a really good movie", "a fantastic movie",
#     "an outstanding movie", "an amazing masterpiece", "a must-watch film"
# ]

# # Define a negative token test for comparison
# negative_tokens = [
#     "a bad movie", "a really bad movie", "a terrible movie",
#     "the worst movie I've seen"
# ]
# # positive_tokens= ["bad", "not good", "good", "great", "fantastic"]
# # positive_tokens = ["good", "very good", "excellent", "outstanding"]

# # Evaluate monotonicity for the base text
# results = evaluator.evaluate_monotonicity(base_text, positive_tokens)

# # # Define base text and tokens to test monotonicity
# # base_text = "The movie was"
# # added_tokens = ["good", "very good", "excellent", "outstanding"]

# # # Evaluate monotonicity
# # results = evaluator.evaluate_monotonicity(base_text, added_tokens)

# # Print the results
# print(f"Monotonicity ratio: {results['monotonic_ratio']}")
# print(f"Monotonicity strength: {results['monotonic_strength']}")
# print(f"Non-monotonicity penalty: {results['non_monotonicity_penalty']}")
# print(f"Combined Monotonicity Score: {results['combined_score']}")


# def mask_tokens_based_on_saliency(text, saliency_scores, threshold):
#     """
#     Masks tokens in the input text based on the given saliency scores and threshold.
#     Tokens with saliency scores below the threshold are replaced with "[MASK]".
#     """
#     tokens = tokenizer.tokenize(text)
#     token_saliency_pairs = list(zip(tokens, saliency_scores))

#     # Mask tokens with saliency scores below the threshold
#     masked_tokens = [
#         token if score >= threshold else "[MASK]"
#         for token, score in token_saliency_pairs
#     ]
    
#     # Convert masked tokens back to a string
#     masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
#     return masked_text


# Faithfulness evaluation function
# def evaluate_faithfulness(model, dataset, saliency_scores, thresholds):
#     """
#     Evaluates the faithfulness of saliency scores using a range of thresholds.
#     Computes the model's performance as tokens are progressively masked based on their saliency scores.
#     Returns the faithfulness score (AUC) and model scores across thresholds.
#     """
#     model_scores = []

#     for threshold in thresholds:
#         thresholded_scores = []
        
#         for text, label, saliency in zip(dataset["text"], dataset["label"], saliency_scores):
#             # Mask tokens below the saliency threshold
#             masked_text = mask_tokens_based_on_saliency(text, saliency, threshold)
            
#             # Tokenize masked text
#             inputs = tokenizer(masked_text, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
            
#             # Get model predictions
#             with torch.no_grad():
#                 outputs = model(**inputs)
#                 predicted_label = torch.argmax(outputs.logits, dim=1).item()
            
#             # Check if the prediction matches the true label
#             thresholded_scores.append(1 if predicted_label == label else 0)
        
#         # Calculate accuracy at this threshold
#         accuracy = np.mean(thresholded_scores)
#         model_scores.append(accuracy)

#     # Calculate AUC for faithfulness score
#     faithfulness_auc = auc(thresholds, model_scores)
#     return faithfulness_auc, model_scores


# # Example main function to perform faithfulness evaluation
# def main():
#      # Initialize the MovieReviews dataset
#     movie_reviews_dataset = MovieReviews(tokenizer)

#     # Define indices of samples you want to test
#     indices_to_test = [0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]  # or any other list of indices
#     saliency_scores = []
#     explainer = GradientExplainer(model, tokenizer)

#     # Loop over specified indices to get text, label, and saliency scores
#     for idx in indices_to_test:
#         review = movie_reviews_dataset.get_review(idx, split_type="test")
#         text = review['text']
#         target_label = review['label']
        
#         # Compute saliency scores for the current review
#         explanation = explainer.compute_feature_importance(text, target=target_label)
#         tokens = explanation.tokens  # Extract tokens from Explanation
#         scores = explanation.scores  # Extract saliency scores from Explanation
#         saliency_scores.append(scores)

#     # Define thresholds for evaluation
#     thresholds = list(range(0, 110, 10))

#     # Run faithfulness evaluation using manually retrieved samples
#     faithfulness_score, model_scores = evaluate_faithfulness(
#         model,
#         {"text": [movie_reviews_dataset.get_review(idx, split_type="test")['text'] for idx in indices_to_test],
#          "label": [movie_reviews_dataset.get_review(idx, split_type="test")['label'] for idx in indices_to_test]},
#         saliency_scores,
#         thresholds
#     )

#     print(f"Faithfulness Score (AUC): {faithfulness_score}")
#     print(f"Model scores across thresholds: {model_scores}")

# if __name__ == "__main__":
#     main()
