"""Fine-tune DistilBERT for sentiment classification using HuggingFace Trainer."""
import os
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from preprocess import load_and_tokenize, MODEL_NAME


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def main():
    train_ds, eval_ds, _ = load_and_tokenize(sample_size=int(os.getenv("SAMPLE_SIZE", 0)) or None)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("./model_output")
    print("Training complete. Model saved to ./model_output")


if __name__ == "__main__":
    main()
