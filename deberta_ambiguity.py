import os
import random
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
)
from typing import Dict, List, Tuple
import logging
import warnings

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration class for DeBERTa training."""

    def __init__(self, experiment_type="deberta"):
        """
        Initialize training configuration.

        Args:
            experiment_type: "deberta", "modernbert"
        """
        self.experiment_type = experiment_type
        self._set_parameters()

    def _set_parameters(self):
        """Set parameters based on experiment type."""
        self.num_epochs = 50
        self.batch_size = 16
        self.learning_rate = 5e-5
        self.warmup_ratio = 0.1
        self.weight_decay = 0.01
        self.max_length = 128
        self.early_stopping_patience = 5
        if self.experiment_type == "deberta":
            self.model_name = "microsoft/deberta-v3-large"
        elif self.experiment_type == "modernbert":
            self.model_name = "answerdotai/ModernBERT-large"


class AmbiguityDataset(Dataset):
    """Dataset class for ambiguity prediction compatible with Hugging Face Trainer."""

    def __init__(
        self, sentences: List[str], labels: List[int], tokenizer, max_length: int = 256
    ):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        label = int(self.labels[idx])

        # Tokenize the sentence
        encoding = self.tokenizer(
            sentence,
            truncation=True,
            padding=False,  # Padding handled by DataCollator
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def compute_metrics(eval_pred):
    """Compute evaluation metrics for Trainer."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary"),
        "precision": precision_score(
            labels, predictions, average="binary", zero_division=0
        ),
        "recall": recall_score(labels, predictions, average="binary", zero_division=0),
    }

    return metrics


def predict(model, sentence) -> Tuple[int, float]:
    """Predict ambiguity for a single sentence."""
    # Move to device (model is already on correct device via Trainer)
    device = next(model.parameters()).device
    encoding = {k: v.to(device) for k, v in sentence.items()}

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence = torch.max(predictions).item()
        predicted_class = torch.argmax(predictions, dim=-1).item()

    return predicted_class, confidence


def detailed_evaluation(model, tokenizer, test_data) -> Dict:
    """Perform detailed evaluation with per-sentence analysis."""

    logger.info("\n" + "=" * 60)
    logger.info("DETAILED EVALUATION")
    logger.info("=" * 60)
    test_sentences = test_data["usage"]
    test_labels = test_data["labels"]

    # Get predictions for each test sentence
    predictions = []
    confidences = []
    device = model.device

    for example in test_data:
        word = example["lemma"]
        sentence = example["usage"]
        sentence = tokenizer(word, sentence, return_tensors="pt").to(device)
        pred, conf = predict(model, sentence)
        predictions.append(pred)
        confidences.append(conf)

    # Per-sentence analysis
    logger.info("\nPer-sentence Analysis:")
    logger.info("[ACTUAL] -> PREDICTED (confidence)")
    correct_predictions = 0

    for i, (sentence, true_label, pred, conf) in enumerate(
        zip(test_sentences, test_labels, predictions, confidences)
    ):
        status = "✓" if pred == true_label else "✗"
        ambig_type = "AMBIGUOUS" if true_label else "CLEAR"
        pred_type = "AMBIGUOUS" if pred else "CLEAR"

        if pred == true_label:
            correct_predictions += 1

        logger.info(
            f"{status} [{ambig_type:9}] -> {pred_type:9} (conf: {conf:.3f}) '{sentence}'"
        )

    # Classification report with error handling
    logger.info("\nClassification Report:")
    try:
        unique_labels = sorted(list(set(test_labels + predictions)))
        if len(unique_labels) == 1:
            logger.info(
                f"Warning: Only one class predicted. All predictions: {unique_labels[0]}"
            )
            logger.info("Model needs more training or different hyperparameters.")
        else:
            target_names = (
                ["Clear", "Ambiguous"]
                if len(unique_labels) == 2
                else [f"Class_{i}" for i in unique_labels]
            )
            logger.info(
                "\n"
                + classification_report(
                    test_labels,
                    predictions,
                    target_names=target_names,
                    labels=unique_labels,
                )
            )
    except Exception as e:
        logger.warning(f"Could not generate classification report: {e}")


def tokenize(example, tokenizer):
    output = tokenizer(
        example["lemma"],
        example["usage"],
        truncation=True,
        padding=True,
        max_length=128,
    )
    output["labels"] = int(example["answer"] == "ambiguous")
    return output


def main(raw_args=None):
    # Set random seeds for reproducibility
    args = argparse.ArgumentParser(raw_args)
    args.add_argument("--model", type=str, default="deberta")
    args.add_argument("--seed", type=int, default=42)
    args = args.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    config = TrainingConfig(args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=2
    )
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    output_dir = f"results_{args.model}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=None,  # Disable wandb/tensorboard logging
        seed=42,
        data_seed=42,
    )
    # Prepare data
    # train_data = load_dataset(
    #     "json", data_files="data/amb.train.json", split="train"
    # ).shuffle(seed=42)
    # test_data = load_dataset(
    #     "json", data_files="data/amb.test.json", split="train"
    # ).shuffle(seed=42)
    # train_dataset = AmbiguityDataset(
    #     train_data["sentence"], train_data["labels"], tokenizer
    # )
    # test_dataset = AmbiguityDataset(
    #     test_data["sentence"], test_data["labels"], tokenizer
    # )
    #
    data = (
        load_dataset(
            "json", data_files="data/wic_chatgpt_annotation.train.jsonl", split="train"
        )
        .train_test_split(test_size=0.2, seed=42)
        .map(lambda x: tokenize(x, tokenizer))
    )
    print(data)
    print(data["train"])
    print(data["train"][0])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            )
        ],
    )

    trainer.train()
    metrics = trainer.evaluate(data["test"])
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    # Detailed evaluation
    detailed_evaluation(model, tokenizer, data["test"])

    # Save model using trainer
    trainer.save_model(f"{output_dir}/best_model")
    logger.info(f"Model saved to {output_dir}/best_model")

    # Test on some example sentences
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE PREDICTIONS")
    logger.info("=" * 60)


if __name__ == "__main__":
    main("deberta")
