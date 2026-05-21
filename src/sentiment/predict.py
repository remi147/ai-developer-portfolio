"""Run inference with the fine-tuned sentiment model."""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["NEGATIVE", "POSITIVE"]


class SentimentPredictor:
    def __init__(self, model_dir: str = "./model_output"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        """Return label and confidence score for a single text input."""
        enc = self.tokenizer(
            text, truncation=True, padding=True, max_length=256, return_tensors="pt"
        )
        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        label_idx = int(torch.argmax(logits))
        return {"label": LABELS[label_idx], "confidence": round(probs[label_idx], 4)}

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]


if __name__ == "__main__":
    predictor = SentimentPredictor()
    examples = [
        "This product is absolutely fantastic!",
        "Worst experience I have ever had.",
        "It was okay, nothing special.",
    ]
    for text in examples:
        result = predictor.predict(text)
        print(f"{text!r:50s}  →  {result['label']} ({result['confidence']:.2%})")
