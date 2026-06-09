import torch
import torch.nn.functional as F
import cv2
import numpy as np
import base64
import io
import string
import webbrowser
import threading
import time
import os
import sys
import threading
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from PIL import Image
from torchvision import transforms

from utils.adversarial import AdversarialTTA
from utils.model_arch import ImprovedLetterCNN
from utils.augmentations import get_tta_transforms

CONFIG = {
    "model_path": "checkpoints/letter_cnn_pgd_fixed_multi_adv_tta.pth",
    "use_tta": True,
    "static_dir": "static",
    "port": 8010,
    "host": "127.0.0.1",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_classes": 26,
    "input_size": 28,
    "emnist_mean": 0.1307,
    "emnist_std": 0.3081,
}

DEVICE = torch.device(CONFIG["device"])
TTA_TRANSFORMS = get_tta_transforms() if CONFIG["use_tta"] else None
LETTERS = list(string.ascii_uppercase)


def normalize_emnist(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = torch.tensor([CONFIG["emnist_mean"]], device=device).view(1, 1, 1, 1)
    std = torch.tensor([CONFIG["emnist_std"]], device=device).view(1, 1, 1, 1)
    return (tensor - mean) / std


def create_app():
    app = FastAPI(title="EMNIST Letter Recognition API")
    static_dir = Path(CONFIG["static_dir"])

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        print(f"Static files served from {static_dir}")
    else:
        print(f"Static folder {static_dir} not found - API only mode")

    return app


app = create_app()


def load_model():
    print(f"Loading model: {CONFIG['model_path']}")
    model = ImprovedLetterCNN(num_classes=CONFIG["num_classes"]).to(DEVICE)

    model_path = Path(CONFIG["model_path"])
    if model_path.exists():
        try:
            state_dict = torch.load(
                CONFIG["model_path"],
                map_location=DEVICE,
                weights_only=False
            )
            model.load_state_dict(state_dict)
            print(f"Model loaded from {CONFIG['model_path']}")

            first_weight = next(model.parameters()).detach().cpu().numpy()
            weight_std = np.std(first_weight)
            if weight_std < 0.01:
                print(f"Weights appear random (std={weight_std:.4f}). Verify training completed.")
            else:
                print(f"Weights verified (std={weight_std:.4f})")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Using random initialization - predictions will be uniform")
    else:
        print(f"Model file not found: {model_path.resolve()}")
        print("Available checkpoints:")
        if Path("checkpoints").exists():
            for f in Path("checkpoints").glob("*.pth"):
                print(f"   - {f.name}")
        print("Using random initialization")

    model.eval()
    print(f"Model ready on {DEVICE}")
    adv_tta_processor = AdversarialTTA(
        model=model,
        device=DEVICE,
        epsilon=0.03,
        n_adv=3,
        attack_type='fgsm'
    )
    return model


model = load_model()


def preprocess_base64(base64_str: str, device: torch.device, use_tta: bool = False):
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    try:
        padding = 4 - (len(base64_str) % 4)
        if padding != 4:
            base64_str += "=" * padding
        img_bytes = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        arr = np.array(img)
    except Exception as e:
        print(f"Decode error: {e}")
        return None

    _, binary = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) < 10:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary[y:y + h, x:x + w]
    size = max(w, h)
    padded = np.zeros((size, size), dtype=np.uint8)
    offset_y = (size - h) // 2
    offset_x = (size - w) // 2
    padded[offset_y:offset_y + h, offset_x:offset_x + w] = cropped
    resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)

    if use_tta and TTA_TRANSFORMS:
        tta_tensors = []
        resized_inverted = 255 - resized
        pil_emnist_format = Image.fromarray(resized_inverted.astype(np.uint8)).convert("L")

        for t in TTA_TRANSFORMS:
            aug_tensor = t(pil_emnist_format)
            aug_tensor = torch.rot90(aug_tensor, k=1, dims=[-2, -1])
            aug_tensor = torch.flip(aug_tensor, dims=[1])
            aug_tensor = aug_tensor.unsqueeze(0).to(device)
            tta_tensors.append(aug_tensor)

        return tta_tensors
    else:
        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = torch.rot90(tensor, k=1, dims=[-2, -1])
        tensor = torch.flip(tensor, dims=[-1])
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(device)
        return normalize_emnist(tensor, device)


@app.get("/", response_class=HTMLResponse)
async def serve_html():
    html_path = Path(CONFIG["static_dir"]) / "index.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>static/index.html not found</h1>")


@app.post("/predict")
async def predict(req: Request):
    try:
        data = await req.json()
        img_b64 = data.get("image")

        if not img_b64:
            return {"error": "No image provided", "probabilities": {}}

        tensor = preprocess_base64(img_b64, DEVICE, use_tta=False)

        if tensor is None:
            return {
                "probabilities": {l: 0.0 for l in LETTERS},
                "top_prediction": None,
                "confidence": 0.0
            }

        with torch.no_grad():
            if CONFIG["use_tta"]:
                avg_logits = adv_tta_processor.predict(tensor)
                probs_tensor = F.softmax(avg_logits, dim=1).squeeze()
            else:
                logits = model(tensor)
                probs_tensor = F.softmax(logits, dim=1).squeeze()

        probs_np = probs_tensor.cpu().numpy()
        probs_dict = {
            letter: round(float(prob) * 100, 2)
            for letter, prob in zip(LETTERS, probs_np)
        }

        top_idx = probs_tensor.argmax().item()
        top_letter = LETTERS[top_idx]
        top_conf = round(float(probs_tensor[top_idx]) * 100, 2)

        return {
            "probabilities": probs_dict,
            "top_prediction": top_letter,
            "confidence": top_conf,
            "tta_used": CONFIG["use_tta"]
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "probabilities": {}}


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": Path(CONFIG["model_path"]).exists(),
        "device": str(DEVICE),
        "tta_enabled": CONFIG["use_tta"]
    }


@app.post("/shutdown")
async def shutdown_server():
    print("Shutting down")

    def delayed_exit():
        time.sleep(0.5)
        os._exit(0)

    threading.Thread(target=delayed_exit, daemon=True).start()
    return {"status": "shutting down"}


def open_browser():
    time.sleep(2)
    webbrowser.open(f"http://{CONFIG['host']}:{CONFIG['port']}")


if __name__ == "__main__":
    import uvicorn

    print(f"Starting server at http://{CONFIG['host']}:{CONFIG['port']}")
    print(f"Model path: {Path(CONFIG['model_path']).resolve()}")
    print(f"TTA: {'On' if CONFIG['use_tta'] else 'Off'}")

    threading.Timer(1, open_browser).start()

    reload_mode = not sys.platform.startswith("win")

    uvicorn.run(
        "model:app",
        host=CONFIG["host"],
        port=CONFIG["port"],
        reload=reload_mode,
        log_level="info"
    )