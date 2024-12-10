# infidelity_calculator.py

import torch
from captum.metrics import infidelity

class InfidelityEvaluator:
    def __init__(self, model, target_label, device="cpu"):
        self.model = model.to(device)
        self.target_label = target_label
        self.device = device

    def forward_func(self, inputs, attention_mask=None):
        # Ensure inputs are in LongTensor format
        inputs = inputs.long()
        
        # Process inputs as a batch and select logits for the target label
        outputs = self.model(inputs, attention_mask=attention_mask).logits
        return outputs[:, self.target_label].unsqueeze(1)  # Ensure 2D output [batch_size, 1]

    def perturb_func(self, inputs):
        noise = torch.normal(mean=0, std=0.01, size=inputs.shape).to(self.device)
        perturbed_inputs = inputs + noise
        return noise.long(), perturbed_inputs.long()  # Ensure outputs are LongTensor

    def calculate_infidelity(self, inputs, attributions):
        # Compute infidelity score using the forward and perturbation functions
        infidelity_score = infidelity(
            forward_func=self.forward_func,    # Forward function for model predictions
            perturb_func=self.perturb_func,    # Function to generate perturbations
            inputs=inputs,                     # Truncated inputs (token IDs)
            attributions=attributions,         # Truncated and batched attributions
            target=self.target_label           # Specify the target label
        )
        return infidelity_score.item()  # Return score as a scalar
