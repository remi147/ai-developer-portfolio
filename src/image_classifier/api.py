"""FastAPI endpoint for image classification inference."""
import io
import json
from pathlib import Path
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from torchvision import transforms
from model import build_model

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT = Path("checkpoints/best.pt")
CLASS_NAMES_FILE = Path("class_names.json")
IMG_SIZE = 224

# ── Transforms ───────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Load model ────────────────────────────────────────────────────────────────
class_names: list[str] = json.loads(CLASS_NAMES_FILE.read_text()) if CLASS_NAMES_FILE.exists() else []
model = build_model(num_classes=max(len(class_names), 2))
if CHECKPOINT.exists():
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
model.eval()

app = FastAPI(title="Image Classifier API", version="1.0.0")


class PredictionOut(BaseModel):
    label: str
    confidence: float
    top5: list[dict]


@app.post("/predict", response_model=PredictionOut)
async def predict(file: UploadFile = File(...)) -> PredictionOut:
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    top5_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:5]
    top5 = [{"label": class_names[i] if class_names else str(i), "confidence": round(probs[i], 4)} for i in top5_idx]
    return PredictionOut(label=top5[0]["label"], confidence=top5[0]["confidence"], top5=top5)


@app.get("/health")
async def health():
    return {"status": "ok"}
