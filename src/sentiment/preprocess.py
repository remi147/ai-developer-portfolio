"""Tokenization and dataset preparation for sentiment analysis."""
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import Optional


MODEL_NAME = "distilbert-base-uncased"


def load_and_tokenize(
    dataset_name: str = "imdb",
    max_length: int = 256,
    sample_size: Optional[int] = None,
) -> tuple[Dataset, Dataset, AutoTokenizer]:
    """Load a HuggingFace dataset and tokenize it for BERT-family models."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    raw = load_dataset(dataset_name)

    train = raw["train"]
    test = raw["test"]

    if sample_size:
        train = train.shuffle(seed=42).select(range(sample_size))
        test = test.shuffle(seed=42).select(range(sample_size // 5))

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    train = train.map(tokenize, batched=True, batch_size=256)
    test = test.map(tokenize, batched=True, batch_size=256)

    for ds in (train, test):
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return train, test, tokenizer
