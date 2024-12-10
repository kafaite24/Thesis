

#Evaluator# sensitivity_calculator.py

import torch
from captum.metrics import sensitivity_max

class SensitivityEvaluator:
    def __init__(self, model, explainer, target_label, device="cpu"):
        self.model = model.to(device)
        self.explainer = explainer  # Instance of GradientExplainer
        self.target_label = target_label
        self.device = device

    def explanation_func(self, perturbed_inputs):
        # Handle tuple input if needed
        if isinstance(perturbed_inputs, tuple):
            perturbed_inputs = perturbed_inputs[0]  # Get first element if inputs is a tuple

        all_attributions = []

        # Loop over each perturbed input in the batch
        for i in range(perturbed_inputs.size(0)):
            token_ids = perturbed_inputs[i].to(self.device).long()  # LongTensor for each example

            # Decode token IDs back to text
            text = self.explainer.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
            
            # Generate attribution for each perturbed input individually
            explanation = self.explainer.compute_feature_importance(text, target=self.target_label)
            attribution_scores = torch.tensor(explanation.scores).to(self.device)

            # Pad or truncate attribution to match the perturbed input's length
            if attribution_scores.size(0) < token_ids.size(0):
                padding = torch.zeros(token_ids.size(0) - attribution_scores.size(0), device=self.device)
                attribution_scores = torch.cat([attribution_scores, padding])
            elif attribution_scores.size(0) > token_ids.size(0):
                attribution_scores = attribution_scores[:token_ids.size(0)]

            # Add batch dimension for concatenation
            all_attributions.append(attribution_scores.unsqueeze(0))

        # Concatenate all attributions into one tensor matching perturbed inputs shape
        all_attributions = torch.cat(all_attributions, dim=0)  # Shape: [batch_size, sequence_length]
        return all_attributions

    def calculate_sensitivity(self, inputs):
        print(f"Shape of inputs in calculate_sensitivity: {inputs.shape}")

        # Compute sensitivity score using Captum's sensitivity_max
        sensitivity_score = sensitivity_max(
            explanation_func=self.explanation_func,  # Attribution generation function
            inputs=inputs,                           # Original inputs (token IDs)
            perturb_radius=0.02,                     # Perturbation radius
            n_perturb_samples=10                     # Number of samples
        )
        return sensitivity_score.item()  # Return scalar score
