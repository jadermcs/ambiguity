import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
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
        self.early_stopping_patience = 3
        if self.experiment_type == "deberta":
            self.model_name = "microsoft/deberta-v3-small"
        elif self.experiment_type == "modernbert":
            self.model_name = "answerdotai/ModernBERT-small"


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
    test_sentences = test_data.sentences
    test_labels = test_data.labels

    # Get predictions for each test sentence
    predictions = []
    confidences = []
    device = model.device

    for sentence in test_sentences:
        sentence = tokenizer(sentence, return_tensors="pt").to(device)
        pred, conf = predict(model, sentence)
        predictions.append(pred)
        confidences.append(conf)

    # Per-sentence analysis
    logger.info("\nPer-sentence Analysis:")
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
                classification_report(
                    test_labels,
                    predictions,
                    target_names=target_names,
                    labels=unique_labels,
                )
            )
    except Exception as e:
        logger.warning(f"Could not generate classification report: {e}")


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def prepare_data(df):
    df_stacked = pd.concat(
        [
            df["T"].to_frame().assign(labels=1).rename(columns={"T": "sentence"}),
            df["F"].to_frame().assign(labels=0).rename(columns={"F": "sentence"}),
        ]
    )
    return df_stacked


def main(experiment_type: str = "deberta", **custom_params):
    config = TrainingConfig(experiment_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=2
    )
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    output_dir = f"results_{experiment_type}"
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
    data = pd.read_csv("experiment_1_data.csv")[
        ["underspecified sentence", "control sentence"]
    ].rename(columns={"underspecified sentence": "T", "control sentence": "F"})
    val_data = data.sample(frac=0.2, random_state=42)
    train_data = data.drop(val_data.index)
    val_data = prepare_data(val_data)
    train_data = prepare_data(train_data)
    train_dataset = AmbiguityDataset(
        train_data["sentence"].values, train_data["labels"].values, tokenizer
    )
    val_dataset = AmbiguityDataset(
        val_data["sentence"].values, val_data["labels"].values, tokenizer
    )
    data = pd.read_csv("ambiguous.txt", sep=";")
    test_dataset = AmbiguityDataset(
        data["sentence"].values, data["labels"].values, tokenizer
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
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
    metrics = trainer.evaluate(test_dataset)
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    # Detailed evaluation
    detailed_evaluation(model, tokenizer, test_dataset)

    # Save model using trainer
    trainer.save_model(f"{output_dir}/best_model")
    logger.info(f"Model saved to {output_dir}/best_model")

    # Test on some example sentences
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE PREDICTIONS")
    logger.info("=" * 60)

    example_sentences = [
        "I saw the man with the telescope",
        "The cat sat on the mat",
        "Flying planes can be dangerous",
        "She reads books every day",
    ]

    for sentence in example_sentences:
        sentence = tokenizer(
            sentence,
            max_length=config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        pred, conf = predict(model, sentence)
        pred_label = "AMBIGUOUS" if pred else "CLEAR"
        logger.info(f"{pred_label} (conf: {conf:.3f}): '{sentence}'")


if __name__ == "__main__":
    main("deberta")
