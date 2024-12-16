import torch
import numpy as np
from utils.saliency_utils import min_max_normalize
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


class ROAREvaluator:
    def __init__(self, model, tokenizer, dataset, importance_scores, k=0, recursive_step_size=1, max_mask=0.5, use_gpu=False):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.importance_scores = importance_scores
        self.k = k
        self.recursive_step_size = recursive_step_size
        self.max_mask = max_mask
        self.use_gpu = use_gpu

    def mask_tokens(self, tokens, importance_scores, mask_proportion):
        """
        Masks tokens based on importance scores.
        
        Args:
            tokens (List[str]): List of tokens in the sentence.
            importance_scores (dict): A dictionary with token indices as keys and importance scores as values.
            mask_proportion (float): Proportion of tokens to mask.
        
        Returns:
            List[str]: List of tokens with some masked based on importance scores.
        """
        num_tokens = len(tokens)
        tokens_with_mask = tokens.copy()

        # Get the indices of [CLS] (index 0) and [SEP] (last token index)
        cls_index = 0
        sep_index = len(tokens) - 1
        # # Sort the tokens by importance scores in descending order
        # sorted_indices = np.argsort(importance_scores)[::-1]  # Sort indices based on importance_scores        
        # Sort indices based on absolute values
        sorted_indices = sorted(range(len(importance_scores)), key=lambda i: abs(importance_scores[i]), reverse=True)

         # Print tokens in descending order of importance scores
        # tokens_sorted_by_importance = [tokens[idx] for idx in sorted_indices if idx != cls_index and idx != sep_index]
        # scores_sorted = [importance_scores[idx] for idx in sorted_indices if idx != cls_index and idx != sep_index]
        # print("Tokens sorted by importance scores (descending):")
        # for token, score in zip(tokens_sorted_by_importance, scores_sorted):
        #     print(f"{token}: {score}")
        # Determine the number of tokens to mask
        num_tokens_to_mask = int(mask_proportion * (num_tokens))  # Don't mask [CLS] and [SEP]
        
        # Mask the tokens with the lowest importance scores
        masked_indices = 0
        for idx in sorted_indices:
            if idx != cls_index and idx != sep_index:
                if masked_indices < num_tokens_to_mask:
                    tokens_with_mask[idx] = '[MASK]'  # Mask this token
                    masked_indices += 1
                else:
                    break  # Stop once we have masked the correct number of tokens
        
        # print(f"tokens {tokens}")
        # print(f"importance scores {np.sort(importance_scores)[::-1]}")
        # print(f"tokens with Mask {tokens_with_mask}")
        return tokens_with_mask

    def evaluate(self, batch):
        """
        Evaluate the model performance (accuracy) for a given batch of data.
        
        Args:
            batch (dict): The batch containing 'tokens' and 'label'.
        
        Returns:
            float: Accuracy of the model on this batch.
        """
        # Convert tokens to input IDs for the model
        inputs = self.tokenizer([batch['tokens']], padding=True, truncation=True, return_tensors="pt", is_split_into_words=True)
        
        # Ensure 'labels' are included in the inputs
        labels = torch.tensor(batch['label']).unsqueeze(0)  # Ensure the label is a tensor with shape (1,)

        # Forward pass through the model
        outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=-1)
        # print(f"labels {labels} preds {preds}")
        # accuracy = (preds == batch['label']).float().mean().item()
        # Compute accuracy
        accuracy = (preds == labels).float().mean()
        # print(f"preds {preds} labels {labels} acc {(preds==labels).float().mean()} loss {loss}")

        # Compute F1 Score
        f1 = f1_score(labels, preds, average='weighted')  # Use 'weighted' for multi-class datasets
        # Compute Precision
        precision = precision_score(labels.cpu(), preds.cpu(), average='weighted',zero_division=0)

        # Compute Recall
        recall = recall_score(labels.cpu(), preds.cpu(), average='weighted',zero_division=0)

        return loss.item(), accuracy, f1, precision, recall

    def recursive_masking(self, idx):
        """
        Perform recursive ROAR masking on a specific data instance.

        Args:
            idx (int): Index of the data instance in the dataset.

        Returns:
            List[dict]: A list of dictionaries containing the evaluation metrics (loss, accuracy) for each masking step.
        """
        results = []
        
        # Get the data instance
        batch= self.dataset.get_instance(idx, split_type='test')
        # batch = self.dataset[idx]
        tokens = batch['tokens'] 
        importance_scores = self.importance_scores[idx]['saliency_scores']
        # importance_scores = min_max_normalize(importance_scores)
        mask_proportion = 0.0  # Start with 0% masking
        step_size = 0.1  # Increase by 10% at each step

        while mask_proportion <= self.max_mask:
            # print(f"tokens {tokens}")
            # Mask tokens based on the current mask proportion
            masked_tokens = self.mask_tokens(tokens, importance_scores, mask_proportion)
            batch['tokens'] = masked_tokens  # Update the batch with masked tokens
            # print(f"masked tokens {masked_tokens}")
            # Evaluate the model with the masked tokens
            loss, accuracy, f1, precision, recall = self.evaluate(batch)
            
            # print(f"accuracy in recursive {accuracy}")
            # print(f"f1 in recursive {f1}")
            # Store the results for the current mask proportion
            results.append({
                "mask_proportion": mask_proportion,
                "loss": loss,
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            })
            # Increase the mask proportion for the next step
            mask_proportion += step_size
        return results

    def random_baseline(self, idx):
        """
        Compute the random baseline performance by randomly masking tokens at each masking ratio.
        
        Returns:
            dict: Random baseline performance curves for both accuracy and F1 score.
        """
        batch= self.dataset.get_instance(idx, split_type='test')
        tokens = batch['tokens']
        random_performance_accuracy = []
        random_performance_f1 = []

        for mask_proportion in np.arange(0, self.max_mask + 0.1, 0.1):  # From 0% to 100% masking, step by 10%
            tokens_with_mask = tokens.copy()
            num_tokens = len(tokens)
            
            # Get the indices of [CLS] (index 0) and [SEP] (last token index)
            cls_index = 0
            sep_index = len(tokens) - 1

            # Randomly select tokens to mask (excluding [CLS] and [SEP])
            num_tokens_to_mask = int(mask_proportion * (num_tokens-2))  # Don't mask [CLS] and [SEP]
            random_indices = np.random.choice(
                [i for i in range(len(tokens)) if i != cls_index and i != sep_index], 
                size=num_tokens_to_mask, 
                replace=False
            )
            for idx in random_indices:
                tokens_with_mask[idx] = '[MASK]'  # Mask the token

            batch['tokens'] = tokens_with_mask
            loss, accuracy, f1,precision, recall = self.evaluate(batch)  # Get accuracy and F1 score
            # print(f"accuracy in baseline {accuracy}")
            # print(f"f1 in baseline {f1}")
            # print(f"mask_proportion: {mask_proportion}, loss {loss} accuracy {accuracy}")
            random_performance_accuracy.append(accuracy)
            random_performance_f1.append(f1)

        return {
            'accuracy_curve': random_performance_accuracy,
            'f1_curve': random_performance_f1
        }

    def evaluate_on_dataset(self):
        """
        Evaluate the model on the entire dataset and compute the RACU score for the entire dataset.
        
        Returns:
            dict: Final evaluation metrics including overall loss, accuracy, F1 score, and RACU score.
        """
        total_loss = 0.0
        total_accuracy = 0.0
        total_f1_score = 0.0
        total_samples = 0
        all_results = []

        # Initialize lists to store the aggregated importance and baseline curves
        all_importance_curves_accuracy = []
        all_baseline_curves_accuracy = []
        all_importance_curves_f1 = []
        all_baseline_curves_f1 = []
        all_importance_curves_precision = []
        all_importance_curves_recall = []
        # Iterate over the entire dataset
        for idx in range(self.dataset.len()):  # Evaluate all samples in the dataset
            print(f"Processing instance {idx}...")
            results = self.recursive_masking(idx)
            # print(f"results {results}")
            # Perform random baseline evaluation for the current instance
            random_baseline_results = self.random_baseline(idx)
            baseline_curve_accuracy = random_baseline_results['accuracy_curve']
            baseline_curve_f1 = random_baseline_results['f1_curve']
            # print(f"random baseline results {random_baseline_results}")

            # Extract accuracy and F1 values for importance and baseline curves
            importance_curve_accuracy = [result['accuracy'] for result in results]
            importance_curve_f1 = [result['f1_score'] for result in results]
            importance_curve_precision= [result['precision'] for result in results]
            importance_curve_recall= [result['recall'] for result in results]
            # print(f"baseline accuracy {baseline_curve_accuracy}  recursive accuracy {importance_curve_accuracy}")
            # print(f"baseline f1 {baseline_curve_accuracy}   recursive f1 {importance_curve_accuracy}")
             # Append the curves for later aggregation
            all_importance_curves_accuracy.append(importance_curve_accuracy)
            all_baseline_curves_accuracy.append(baseline_curve_accuracy)
            all_importance_curves_f1.append(importance_curve_f1)
            all_baseline_curves_f1.append(baseline_curve_f1)
            all_importance_curves_precision.append(importance_curve_precision)
            all_importance_curves_recall.append(importance_curve_recall)
            # Accumulate total loss, accuracy, and F1 score
            for result in results:
                total_loss += result['loss']
                total_accuracy += result['accuracy']
                total_f1_score += result['f1_score']
                total_samples += 1
                all_results.append(result)

        # Aggregate the curves across all instances
        avg_importance_curve_accuracy = np.mean(np.array(all_importance_curves_accuracy), axis=0)
        avg_baseline_curve_accuracy = np.mean(np.array(all_baseline_curves_accuracy), axis=0)
        avg_importance_curve_f1 = np.mean(np.array(all_importance_curves_f1), axis=0)
        avg_baseline_curve_f1 = np.mean(np.array(all_baseline_curves_f1), axis=0)
        avg_importance_curve_precision = np.mean(np.array(all_importance_curves_precision), axis=0)
        avg_importance_curve_recall = np.mean(np.array(all_importance_curves_recall), axis=0)

        # print(f"avg_importance_curves_accuracy {avg_importance_curve_accuracy}")
        # print(f"avg_importance_curves_precision {avg_importance_curve_precision}")
        # print(f"avg_importance_curves_recall {avg_importance_curve_recall}")
        # # # Accumulate total loss, accuracy, and F1 score
        # for result in results:
        #     total_loss += result['loss']
        #     total_accuracy += result['accuracy']
        #     total_f1_score += result['f1_score']
        #     total_samples += 1
        #     all_results.append(result)

        # print(f"imp f1 {avg_importance_curve_f1} baseline f1 {avg_baseline_curve_f1}")

        # Compute RACU score for the entire dataset
        racu_score_accuracy = self.compute_racu(avg_importance_curve_accuracy, avg_baseline_curve_accuracy)
        # racu_score_f1 = self.compute_racu(avg_importance_curve_f1, avg_baseline_curve_f1)
        print(f"RACU scores - Accuracy: {racu_score_accuracy}")

        # avg_loss = total_loss / total_samples
        # avg_accuracy = total_accuracy / total_samples
        # avg_f1_score = total_f1_score / total_samples

        # # Print final results
        # print(f"Final Evaluation Results: Average Loss = {avg_loss:.4f}, Average Accuracy = {avg_accuracy:.4f}, Average F1 Score = {avg_f1_score:.4f}")

        # # Plot separate curves for accuracy and F1 scores
        mask_proportions = np.arange(0, 1.1, 0.1)  # From 0% to 100% masking, step by 10%
        self.plot_accuracy_curves(mask_proportions, avg_importance_curve_accuracy, avg_baseline_curve_accuracy)
        # self.plot_f1_curves(mask_proportions, avg_importance_curve_f1, avg_baseline_curve_f1)
        
        return racu_score_accuracy
    # {
    #         # 'average_loss': avg_loss,
    #         # 'average_accuracy': avg_accuracy,
    #         # 'average_f1_score': avg_f1_score,
    #         'racu_score_accuracy': racu_score_accuracy,
    #         # 'racu_score_f1': racu_score_f1,
    #         # 'all_results': all_results
    #     }

    def plot_accuracy_curves(self, mask_proportions, importance_curve_accuracy, baseline_curve_accuracy):
        """
        Visualize the accuracy curves for the entire dataset.
        
        Args:
            mask_proportions (list): List of masking proportions (usually [0, 0.1, ..., 1.0]).
            importance_curve_accuracy (list): Combined importance curve for accuracy.
            baseline_curve_accuracy (list): Combined baseline curve for accuracy.
        """
        plt.figure(figsize=(10, 8))
        plt.plot(mask_proportions, importance_curve_accuracy, label="Importance Curve (Accuracy)", marker='o', linestyle='-', linewidth=2, color='blue')
        plt.plot(mask_proportions, baseline_curve_accuracy, label="Baseline Curve (Accuracy)", marker='x', linestyle='--', linewidth=2, color='red')
        plt.title("Performance Curves: Accuracy vs Mask Proportion", fontsize=16)
        plt.xlabel("Mask Proportion (%)", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.show()


    def plot_f1_curves(self, mask_proportions, importance_curve_f1, baseline_curve_f1):
        """
        Visualize the F1 score curves for the entire dataset.
        
        Args:
            mask_proportions (list): List of masking proportions (usually [0, 0.1, ..., 1.0]).
            importance_curve_f1 (list): Combined importance curve for F1 scores.
            baseline_curve_f1 (list): Combined baseline curve for F1 scores.
        """
        plt.figure(figsize=(10, 8))
        plt.plot(mask_proportions, importance_curve_f1, label="Importance Curve (F1 Score)", marker='o', linestyle='-', linewidth=2, color='green')
        plt.plot(mask_proportions, baseline_curve_f1, label="Baseline Curve (F1 Score)", marker='x', linestyle='--', linewidth=2, color='orange')
        plt.title("Performance Curves: F1 Score vs Mask Proportion", fontsize=16)
        plt.xlabel("Mask Proportion (%)", fontsize=14)
        plt.ylabel("F1 Score", fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.show()

    def compute_area_between_curves(self, importance_curve, baseline_curve):
        """
        Compute the area between two curves using the trapezoidal rule (numerical integration).
        Args:
            importance_curve (list): The performance curve based on the importance measure.
            baseline_curve (list): The performance curve based on the random baseline.

        Returns:
            tuple: (area_faithfulness, area_baseline)
        """
        area_faithfulness = 0.0
        area_baseline = 0.0
        # Calculate areas using the trapezoidal rule
        for i in range(1, len(importance_curve)):
            delta_x = 10  # Since the step size is 10%
            # Calculate the deltas between consecutive points for both curves
            delta_p = importance_curve[i] - importance_curve[i - 1]
            delta_b = baseline_curve[i] - baseline_curve[i - 1]
            # Compute areas using the trapezoidal rule
            area_faithfulness += 0.5 * delta_x * (importance_curve[i - 1] + importance_curve[i])
            area_baseline += 0.5 * delta_x * (baseline_curve[i - 1] + baseline_curve[i])
        
        return area_faithfulness, area_baseline

    def compute_racu(self, importance_curve, baseline_curve):
        """
        Compute the RACU metric by normalizing the faithfulness area by the baseline area.
        
        Args:
            importance_curve (list): The performance curve based on the importance measure.
            baseline_curve (list): The performance curve based on the random baseline.
        
        Returns:
            float: The RACU score.
        """
        area_faithfulness, area_baseline = self.compute_area_between_curves(importance_curve, baseline_curve)
        # Avoid division by zero in case the baseline area is zero
        if area_baseline == 0:
            return 0
        # Normalize the area by the baseline area
        racu = area_faithfulness / area_baseline
        return racu

    def plot_curves(self,importance_curve, baseline_curve):
        """
        Visualize the importance curve and the baseline curve.
        
        Args:
            importance_curve (list): The performance curve based on importance-based masking.
            baseline_curve (list): The performance curve based on random token shuffling.
        """
        # Define the masking proportions (from 0% to 100% in steps of 10%)
        mask_proportions = np.arange(0, 1.1, 0.1)  # 0%, 10%, ..., 100%
        plt.figure(figsize=(8, 6))
        plt.plot(mask_proportions, importance_curve, label="Importance Curve", color='blue', marker='o', linestyle='-', linewidth=2)
        plt.plot(mask_proportions, baseline_curve, label="Baseline Curve", color='red', marker='x', linestyle='--', linewidth=2)
        plt.title("Performance Curves: Importance vs Baseline", fontsize=16)
        plt.xlabel("Mask Proportion (%)", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.show()

def compute_all_ROAR(dataset, model, tokenizer, saliency_scores):
    # Initialize the FADEvaluator
    roar_evaluator = ROAREvaluator(
            model=model,
            dataset=dataset,
            tokenizer=tokenizer,
            importance_scores=saliency_scores, 
            k=3, 
            recursive_step_size=1, 
            max_mask=1.0,
            use_gpu=True
        )

    return roar_evaluator.evaluate_on_dataset()


