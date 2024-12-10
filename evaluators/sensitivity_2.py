import torch
import torch.nn.functional as F

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
        self.model.to(self.device)

    def pgd_attack(self, input_ids, attention_mask, labels, relevant_token_indices):
        """
        PGD attack to generate adversarial perturbations using binary search.

        Args:
            input_ids: The tokenized input for the sample (Tensor).
            attention_mask: The attention mask for the input (Tensor).
            labels: The true labels (Tensor).
            relevant_token_indices: The indices of the relevant tokens (Tensor).

        Returns:
            perturbed_input: The adversarially perturbed input embeddings.
            perturbation_norm: The norm of the perturbation used to change the model's prediction.
        """
        # Don't apply gradients to input_ids, only apply it to the embeddings
        input_ids = input_ids.clone().detach()

        # Forward pass to obtain token embeddings from the model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # List of all hidden states, including the last hidden state

        # Get the last hidden state (the token embeddings)
        embeddings = hidden_states[-1]  # Last hidden state is the token embeddings

        # Create the perturbation tensor (this will require gradients)
        perturbation = torch.zeros_like(embeddings).to(input_ids.device)
        perturbation.requires_grad_()

        # Convert labels to the correct device
        labels = labels.to(input_ids.device)

        # Binary Search Parameters
        low = 0.0
        high = self.epsilon

        for _ in range(self.num_steps):
            # Find the midpoint for binary search
            mid = (low + high) / 2.0

            # Create a perturbation tensor (no in-place operation)
            perturbed_input = embeddings.clone()

            # Apply perturbation only to the relevant tokens (relevant_token_indices)
            perturbation_tensor = torch.zeros_like(perturbed_input).to(input_ids.device)
            perturbation_tensor[:, relevant_token_indices] = mid * torch.randn_like(perturbation_tensor[:, relevant_token_indices])

           
            # Ensure perturbation_tensor is differentiable by setting requires_grad_
            perturbation_tensor.requires_grad_()

            # Add perturbation to the embeddings
            perturbed_input += perturbation_tensor

            # Forward pass with the current perturbed embeddings
            logits = self.model(inputs_embeds=perturbed_input).logits  # Get logits from the classification head

            # Calculate loss
            loss = F.cross_entropy(logits, labels)

            # Zero previous gradients
            self.model.zero_grad()

            # Compute gradients with respect to perturbations
            loss.backward(retain_graph=True)  # Retain the graph for multiple backward passes

            # Now check if gradients are available for perturbation_tensor
            if perturbation_tensor.grad is None:
                raise RuntimeError("Gradients for perturbation_tensor not computed correctly!")

            # Check if the prediction changes
            if logits.argmax(dim=1) != labels:
                # If the prediction changes, decrease perturbation size
                high = mid
            else:
                # If the prediction does not change, increase perturbation size
                low = mid

            # After backward pass, zero out gradients for the next step
            perturbation_tensor.grad.zero_()

        # Calculate the perturbation norm (Frobenius norm)
        perturbation_norm = torch.norm(perturbation_tensor, p='fro').item()

        # Return the perturbed input (embeddings with perturbations) and the perturbation norm
        return perturbed_input, perturbation_norm


    def compute_sensitivity(self, dataset, saliency_scores):
        """
        Compute Sensitivity for all samples based on saliency scores.

        Args:
            saliency_scores: Precomputed saliency scores to determine relevant tokens.

        Returns:
            sensitivity_scores: List of sensitivity scores (perturbation norms) for each sample.
        """
        sensitivity_scores = []

        # Loop through the dataset
        for entry in saliency_scores[:20]:  # Limit to first 5 samples (for simplicity)
            # Get the text, tokens, and saliency scores for the current sample
            text = entry["text"]
            tokens = entry["tokens"]
            saliency_scores_sample = torch.tensor(entry["saliency_scores"]).to(self.device)
            label = entry["label"]

            # Tokenize the input text using the tokenizer
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)

            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            # Identify relevant tokens based on saliency scores (top k tokens)
            relevant_token_indices = torch.argsort(saliency_scores_sample, descending=True)[:int(len(tokens) * 0.2)]  # Top 20% relevant tokens

            # Compute the perturbation (PGD attack with binary search)
            perturbed_input, perturbation_norm = self.pgd_attack(input_ids, attention_mask, torch.tensor([label]).to(self.device), relevant_token_indices)

            # Append the perturbation norm (sensitivity score)
            sensitivity_scores.append(perturbation_norm)

        # Calculate the average sensitivity score
        average_sensitivity = sum(sensitivity_scores) / len(sensitivity_scores)
        return average_sensitivity


def compute_all_sensitivity(model, tokenizer, dataset, saliency_scores):
    
    # Instantiate the evaluator
    evaluator = SensitivityEvaluator(model, tokenizer, epsilon=0.1, num_steps=10, alpha=0.01, device="cpu")

    # Compute sensitivity
    sensitivity_score = evaluator.compute_sensitivity(dataset, saliency_scores)

    return sensitivity_score












#-------------------------------WITHOUT BINARY SEARCH----------------------------------------------
    # def pgd_attack(self, input_ids, attention_mask, labels):
    #     """
    #     PGD attack to generate adversarial perturbations.

    #     Args:
    #         input_ids: The tokenized input for the sample (Tensor).
    #         attention_mask: The attention mask for the input (Tensor).
    #         labels: The true labels (Tensor).

    #     Returns:
    #         perturbed_input: The adversarially perturbed input embeddings.
    #         perturbation_norm: The norm of the perturbation used to change the model's prediction.
    #     """
    #     # Don't apply gradients to input_ids, only apply it to the embeddings
    #     input_ids = input_ids.clone().detach()

    #     # Forward pass to obtain token embeddings from the model
    #     outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    #     hidden_states = outputs.hidden_states  # List of all hidden states, including the last hidden state

    #     # Get the last hidden state (the token embeddings)
    #     embeddings = hidden_states[-1]  # Last hidden state is the token embeddings

    #     # Create the perturbation tensor (this will require gradients)
    #     perturbation = torch.zeros_like(embeddings).to(input_ids.device)
    #     perturbation.requires_grad_()

    #     # Convert labels to the correct device
    #     labels = labels.to(input_ids.device)

    #     for _ in range(self.num_steps):
    #         # Forward pass with the current perturbed embeddings
    #         perturbed_input = embeddings + perturbation
    #         logits = self.model(inputs_embeds=perturbed_input).logits  # Get logits from the classification head

    #         # Calculate loss
    #         loss = F.cross_entropy(logits, labels)

    #         # Zero previous gradients
    #         self.model.zero_grad()

    #         # Compute gradients with respect to perturbations
    #         loss.backward(retain_graph=True)  # Retain the graph for multiple backward passes

    #         # Update the perturbation using the gradient
    #         perturbation.data = perturbation.data + self.alpha * torch.sign(perturbation.grad.data)
    #         perturbation.data = torch.clamp(perturbation.data, -self.epsilon, self.epsilon)  # Clip to max perturbation epsilon

    #         # Re-zero gradients for the next step
    #         perturbation.grad.data.zero_()

    #     # Calculate the perturbation norm (Frobenius norm)
    #     perturbation_norm = torch.norm(perturbation, p='fro').item()

    #     # Return the perturbed input (embeddings with perturbations) and the perturbation norm
    #     return perturbed_input, perturbation_norm

    # def compute_sensitivity(self, dataset, saliency_scores):
    #     """
    #     Compute Sensitivity for all samples based on saliency scores.

    #     Args:
    #         saliency_scores: Precomputed saliency scores to determine relevant tokens.

    #     Returns:
    #         sensitivity_scores: List of sensitivity scores (perturbation norms) for each sample.
    #     """
    #     sensitivity_scores = []

    #     # Loop through the dataset
    #     for entry in saliency_scores[:5]:
    #         # Get the text, tokens, and saliency scores for the current sample
    #         text = entry["text"]
    #         tokens = entry["tokens"]
    #         saliency_scores_sample = torch.tensor(entry["saliency_scores"]).to(self.device)
    #         label = entry["label"]

    #         # Tokenize the input text using the tokenizer
    #         inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)

    #         input_ids = inputs['input_ids']
    #         attention_mask = inputs['attention_mask']

    #         # Identify relevant tokens based on saliency scores (top k tokens)
    #         relevant_token_indices = torch.argsort(saliency_scores_sample, descending=True)[:int(len(tokens) * 0.2)]  # Top 20% relevant tokens

    #         # Compute the perturbation (PGD attack)
    #         perturbed_input, perturbation_norm = self.pgd_attack(input_ids, attention_mask, torch.tensor([label]).to(self.device))

    #         # Append the perturbation norm (sensitivity score)
    #         sensitivity_scores.append(perturbation_norm)

    #     # Calculate the average sensitivity score
    #     average_sensitivity = sum(sensitivity_scores) / len(sensitivity_scores)
    #     return average_sensitivity
