
from explainers.saliencymanager import SaliencyScoreManager
from evaluators.faithfulness_auc import FaithfulnessEvaluator
from evaluators.monotonicity import MonotonicityEvaluator
from evaluators.faithfulness_correlation import FaithfulnessCorrelationEvaluator
from evaluators.IOU import IOUEvaluator
from evaluators.AUPRC import AUPRCEvaluator
from evaluators.FAD import FADEvaluator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from explainers import GradientExplainer
from dataset_loaders import MovieReviews
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import BertTokenizerFast, BertForSequenceClassification
from EvalXAI.evaluators.sensitivity import SensitivityEvaluator
# # Load model and tokenizer
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Load the model and tokenizer (for example, BERT)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased").to("cpu")  # Use BertForSequenceClassification
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

saliency_file_path = "data/saliency_scores.json"

# Load dataset
dataset = MovieReviews(tokenizer)
sample_idx = 0  # Just an example; you can choose any index
sample_data = dataset.get_review(sample_idx, split_type="test", combine_rationales=True)
# # Initialize Saliency Manager
manager = SaliencyScoreManager(
    model=model,
    tokenizer=tokenizer,
    explainer_class=GradientExplainer,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
# # Compute or load saliency scores
if not os.path.exists(saliency_file_path):
    manager.compute_and_save_scores(dataset, saliency_file_path, split_type="test")
precomputed_scores = manager.load_scores(saliency_file_path)

saliency_scores= precomputed_scores
# print(saliency_scores)
# Example saliency scores (should be precomputed)
# saliency_scores = [
#     {"text": "This movie is great!", "tokens": ["this", "movie", "is", "great", "!"], "saliency_scores": [0.1, 0.9, 0.2, 0.8, 0.5], "label": 1},
#     {"text": "I hate this film.", "tokens": ["I", "hate", "this", "film", "."], "saliency_scores": [0.3, 0.7, 0.2, 0.6, 0.4], "label": 0}
# ]

# Dataset object (just for illustration, replace with your actual dataset)
dataset = None  # Assuming dataset is not used directly here for simplicity

# Instantiate the evaluator
evaluator = SensitivityEvaluator(model, tokenizer, epsilon=0.1, num_steps=10, alpha=0.01, device="cpu")

# Compute sensitivity
sensitivity_score = evaluator.compute_sensitivity(sample_data, saliency_scores)

print(f"Average Sensitivity: {sensitivity_score}")

# # Get a sample from the test set (or any other set)
# sample_idx = 0  # Just an example; you can choose any index
# sample_data = dataset.get_review(sample_idx, split_type="test", combine_rationales=True)

# # Print the sample review
# # print(f"Review Text: {sample_data['text']}")
# # print(f"Tokens: {sample_data['tokens']}")
# # print(f"Rationale Mask: {sample_data['rationale']}")
# # print(f"Label: {sample_data['label']}")

# tokens = sample_data["tokens"]
# rationale_mask = sample_data["rationale"]
# label = sample_data["label"]


# # Prepare the inputs for the model
# inputs = tokenizer(sample_data["text"], padding=True, truncation=True, return_tensors="pt", max_length=6)

# # Pass the inputs through the model to get the prediction probabilities
# outputs = model(**inputs)
# yhat = F.softmax(outputs.logits.detach().cpu(), dim=-1).numpy()

# # Get the full text probabilities (probabilities for all classes)
# full_text_probs = yhat[0]  # First row (assuming batch size of 1)

# # Determine the true class label
# full_text_class = np.array([label])
# # Define rows (e.g., selecting the first row of predictions)
# rows = np.array([0])

# # Define the baseline sufficiency value (could be computed or fixed)
# suff_y_zero = np.array([0.5])

# # # Assuming `full_text_probs` is a list of probabilities for each class (for simplicity, let's use dummy values)
# # full_text_probs = np.array([0.6, 0.4])  # Example: Probability distribution over two classes
# # full_text_class = np.array([label])  # True label for the review
# # rows = np.array([0])  # Assuming a single row
# # suff_y_zero = np.array([0.5])  # Initial sufficiency value for the sample

# print("Full text probs:", full_text_probs)

# # Create the inputs dictionary for the model
# inputs = tokenizer(
#     sample_data["text"],  # Original review text
#     padding=True,
#     truncation=True,
#     return_tensors="pt",
#     max_length=6  # Using max_length 6 as an example
# )

# # Here, we mock the `only_query_mask` as an example. You can modify this based on your use case.
# only_query_mask = torch.zeros((1, len(tokens)))  # Example: all zeros for simplicity
# # Assuming precomputed_scores is loaded as described
# saliency_scores = precomputed_scores[0]['saliency_scores']

# # Convert the list to a tensor
# saliency_scores_tensor = torch.tensor(saliency_scores)

# # Apply sigmoid
# importance_scores = torch.sigmoid(saliency_scores_tensor)
# # Now, pass the prepared data to the SoftSufficiency class
# soft_sufficiency = SoftSufficiency(
#     model=model,
#     tokenizer=tokenizer,
#     original_sentences=tokens,  # Tokenized sentences (including special tokens like [CLS], [SEP])
#     rationale_mask=torch.tensor(rationale_mask),  # Rationale mask
#     inputs=inputs,  # Tokenizer's inputs (e.g., input_ids, attention_mask)
#     full_text_probs=full_text_probs,
#     full_text_class=full_text_class,
#     rows=rows,
#     suff_y_zero=suff_y_zero,
#     importance_scores=importance_scores,
#     only_query_mask=only_query_mask,
#     max_len=6,  # Maximum sequence length
#     normalise=1
# )

# # Compute soft sufficiency
# norm_suff, yhat = soft_sufficiency.compute_soft_sufficiency()

# # Print the results
# print("Normalized Sufficiency:", norm_suff)
# # print("Model Predictions (Probabilities):", yhat)








# evaluator = FaithfulnessCorrelationEvaluator(model, tokenizer)
# # Apply faithfulness correlation metric
# faithfulness_score = evaluator.compute_faithfulness_score(precomputed_scores)
# print(f"Faithfulness Correlation Score: {faithfulness_score}")


# # # Initialize FaithfulnessEvaluator
# # evaluator = FaithfulnessEvaluator(model, tokenizer)

# # # Define thresholds
# # thresholds = list(range(0, 110, 10))  # 0%, 10%, ..., 100%

# # # Evaluate performance
# # performance_scores = evaluator.evaluate_performance(precomputed_scores, thresholds)

# # # Compute AUC-TP
# # auc_tp = evaluator.compute_auc_tp(thresholds, performance_scores)
# # print(f"AUC-TP: {auc_tp}")

# # # Plot the performance curve
# # evaluator.plot_performance(thresholds, performance_scores)

# Load dataset
# dataset = MovieReviews(tokenizer)

# # Initialize Saliency Manager
# manager = SaliencyScoreManager(
#     model=model,
#     tokenizer=tokenizer,
#     explainer_class=GradientExplainer,
#     device="cuda" if torch.cuda.is_available() else "cpu"
# )
# # Compute or load saliency scores
# if not os.path.exists(saliency_file_path):
#     manager.compute_and_save_scores(dataset, saliency_file_path, split_type="test")
# precomputed_scores = manager.load_scores(saliency_file_path)

# # Initialize the evaluator
# evaluator = MonotonicityEvaluator(model, tokenizer)

# # Evaluate the dataset
# average_monotonicity = evaluator.evaluate_dataset(evaluator, precomputed_scores)


# Initialize evaluator
# evaluator = IOUEvaluator(dataset, precomputed_scores, threshold=0.5)

# # Evaluate on the test set
# average_metrics = evaluator.evaluate(split_type="test")
# print("Final Average Metrics:", average_metrics)

# evaluator = AUPRCEvaluator(dataset=dataset, saliency_scores=precomputed_scores)

# # Evaluate on the "test" split
# average_auprc = evaluator.evaluate(split_type="test")
# print(f"Final Average AUPRC on Test Set: {average_auprc}")

# Initialize the evaluator
# evaluator = FADEvaluator(
#     dataset=dataset,  # Your MovieReviews dataset
#     model=model,  # Your pretrained text classification model
#     saliency_scores=precomputed_scores,  # Precomputed saliency scores
#     tokenizer=tokenizer,  # Tokenizer for your model
#     device="cuda" if torch.cuda.is_available() else "cpu"
# )

# # Percentages of tokens to drop
# percent_dropped_features = list(range(0, 101, 10))

# # Evaluate the impact of feature dropping
# results_df = evaluator.evaluate(percent_dropped_features=percent_dropped_features)

# n_auc = evaluator.calculate_n_auc(results_df, percent_range=(0, 20))
# print(f"N-AUC (0-20%): {n_auc}")
# # Plot the results
# evaluator.plot_results(results_df)
