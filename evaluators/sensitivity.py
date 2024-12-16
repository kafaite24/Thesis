
import torch
import torch.nn.functional as F
from utils.text_helpers import SequenceClassificationHelper


class SensitivityEvaluator:
    def __init__(self, model, tokenizer, epsilon=0.1, num_steps=10, alpha=0.01, device="cpu"):
        """
        Initialize the SensitivityEvaluator.

        Args:
            model: The model to attack.
            tokenizer: The tokenizer to process the input text.
            epsilon: The perturbation size.
            num_steps: Number of PGD steps.
            alpha: Step size for PGD.
            device: Device for computations ('cpu' or 'cuda').
        """
        self.model = model
        self.tokenizer = tokenizer
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.alpha = alpha
        self.device = device
        self.helper = SequenceClassificationHelper(self.model, self.tokenizer)
        self.model.to(self.device)

    def pgd_attack(self, input_ids, attention_mask, labels, relevant_token_indices):
        """
        PGD attack to generate adversarial perturbations using binary search.
        """
        # Clone input IDs and detach
        input_ids = input_ids.clone().detach()

        # Get embeddings
        self.model.eval()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1]  # Last hidden state (embeddings)

        # Initialize perturbation
        perturbation = torch.zeros_like(embeddings, requires_grad=True).to(input_ids.device)
        cumulative_perturbation = torch.zeros_like(embeddings).to(input_ids.device)

        # Binary search parameters
        low, high, threshold = 0.0, self.epsilon, 1e-3

        for step in range(self.num_steps):
            mid = (low + high) / 2.0

            # Apply perturbation to embeddings
            perturbed_input = embeddings + perturbation

            # Forward pass
            logits = self.model(inputs_embeds=perturbed_input, attention_mask=attention_mask).logits
            loss = F.cross_entropy(logits, labels)

            # Zero gradients
            self.model.zero_grad()
            if perturbation.grad is not None:
                perturbation.grad.zero_()

            # Backward pass
            print(f"Before backward: perturbation.grad is {perturbation.grad}")
            loss.backward(retain_graph=True)
            print(f"After backward: perturbation.grad is {perturbation.grad}")

            if perturbation.grad is None:
                raise RuntimeError("Gradients not computed for perturbation!")

            # Generate perturbation update
            perturbation_update = mid * torch.sign(perturbation.grad)

            # Apply the perturbation only to relevant token indices
            batch_indices = torch.arange(cumulative_perturbation.size(0)).unsqueeze(-1).to(input_ids.device)
            new_perturbation = torch.zeros_like(embeddings).to(input_ids.device)
            new_perturbation[batch_indices, relevant_token_indices, :] = perturbation_update[batch_indices, relevant_token_indices, :]

            # Accumulate the perturbation
            cumulative_perturbation += new_perturbation

            # Update perturbation tensor
            perturbation.data = cumulative_perturbation

            # Binary search logic
            if logits.argmax(dim=1) != labels:
                high = mid  # Prediction changed
            else:
                low = mid  # Prediction unchanged

            # Exit if binary search converges
            if high - low < threshold:
                break

            print(f"Step {step}")
            print(f"Cumulative perturbation norm: {torch.norm(cumulative_perturbation, p='fro').item()}")

        # Compute norm of total perturbation
        perturbation_norm = torch.norm(cumulative_perturbation, p='fro').item()

        return embeddings + cumulative_perturbation, perturbation_norm

    def compute_sensitivity(self, dataset, saliency_scores):
        """
        Compute Sensitivity for all samples based on saliency scores.

        Args:
            dataset: Dataset containing the samples.
            saliency_scores: Precomputed saliency scores to determine relevant tokens.

        Returns:
            sensitivity_scores: List of sensitivity scores (perturbation norms) for each sample.
        """
        sensitivity_scores = []

        # Loop through the dataset
        for entry in saliency_scores[:30]:  # Limit to first 10 samples for simplicity
            # Get the text, tokens, and saliency scores for the current sample
            text = entry["text"]
            tokens = entry["tokens"]
            saliency_scores_sample = torch.tensor(entry["saliency_scores"])
            label = entry["label"]

            # Tokenize the input text using the tokenizer
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            # Convert label to numerical format
            if isinstance(label, str):
                label = self.model.config.label2id[label]
            print(f"text {text}")
            # Identify relevant tokens based on saliency scores (top k tokens)
            relevant_token_indices = torch.argsort(torch.abs(saliency_scores_sample), descending=True)[:int(len(tokens) * 0.2)]  # Top 20% relevant tokens
            tokens_sorted_by_importance = [tokens[idx] for idx in relevant_token_indices]
            scores_sorted = [saliency_scores_sample[idx] for idx in relevant_token_indices] 
            # print(f"tokens sorted {tokens_sorted_by_importance}")
            # print(f"scores sorted {scores_sorted}")
            # Compute the perturbation (PGD attack with binary search)
            perturbed_input, perturbation_norm = self.pgd_attack(input_ids, attention_mask, torch.tensor([label]), relevant_token_indices)

            # Check if the model prediction changes
            logits = self.model(inputs_embeds=perturbed_input, attention_mask=attention_mask).logits
            print(f"pred {logits.argmax(dim=1).item()}")
            prediction_changed = logits.argmax(dim=1).item() != label

            print(f"Original label: {label}, Perturbation norm: {perturbation_norm}, Prediction changed: {prediction_changed}")

            # Append the perturbation norm (sensitivity score)
            sensitivity_scores.append(perturbation_norm)

        # Calculate the average sensitivity score
        average_sensitivity = sum(sensitivity_scores) / len(sensitivity_scores)
        return average_sensitivity


def compute_all_sensitivity(model, tokenizer, dataset, saliency_scores):
    """
    Compute sensitivity for the entire dataset.

    Args:
        model: Model to evaluate.
        tokenizer: Tokenizer for input preprocessing.
        dataset: Dataset for evaluation.
        saliency_scores: Precomputed saliency scores.

    Returns:
        sensitivity_score: The average sensitivity score across the dataset.
    """
    # Instantiate the evaluator
    evaluator = SensitivityEvaluator(model, tokenizer, epsilon=0.1, num_steps=10, alpha=0.01, device="cpu")

    # Compute sensitivity
    sensitivity_score = evaluator.compute_sensitivity(dataset, saliency_scores)

    return sensitivity_score