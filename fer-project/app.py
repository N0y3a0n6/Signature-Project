"""Gradio demo for real-time facial emotion recognition."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import torch
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.config import Config
from src.models.efficientnet_model import EfficientNetFER

# ── Config & constants ────────────────────────────────────────────────────────
CONFIG_PATH  = Path(__file__).parent / 'config/config.yaml'
CHECKPOINT   = Path(__file__).parent / 'models/checkpoints/best_model.pth'
HAAR_CASCADE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
EMOTION_COLORS = ["#e74c3c", "#8e44ad", "#2980b9", "#f1c40f", "#3498db", "#e67e22", "#95a5a6"]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print(f"Loading model on {DEVICE}...")
config = Config(str(CONFIG_PATH))
model  = EfficientNetFER(
    model_name  = config.get('model.architecture'),
    num_classes = config.get('emotions.num_classes'),
    pretrained  = False,
    dropout     = config.get('model.dropout'),
)
ckpt = torch.load(str(CHECKPOINT), map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(DEVICE)
model.eval()
print("Model ready.")

face_detector = cv2.CascadeClassifier(HAAR_CASCADE)


# ── Helpers ───────────────────────────────────────────────────────────────────

def preprocess(face_bgr: np.ndarray) -> torch.Tensor:
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (224, 224))
    face_f   = face_rgb.astype(np.float32) / 255.0
    face_f   = (face_f - MEAN) / STD
    t        = torch.from_numpy(face_f.transpose(2, 0, 1)).unsqueeze(0)
    return t.to(DEVICE)


def bar_chart(probs: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(EMOTIONS, probs, color=EMOTION_COLORS, edgecolor='none')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Confidence', fontsize=9)
    ax.set_title('Emotion Probabilities', fontsize=10, pad=6)
    ax.tick_params(axis='y', labelsize=9)
    for i, p in enumerate(probs):
        ax.text(p + 0.01, i, f'{p:.2f}', va='center', fontsize=8)
    fig.tight_layout()
    return fig


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(image: np.ndarray):
    """image is an RGB numpy array from Gradio (webcam or upload)."""
    if image is None:
        return None, None, "No image — take a photo or upload one."

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
    )

    if len(faces) == 0:
        # Still try the whole image centre crop as fallback
        h, w = img_bgr.shape[:2]
        size = min(h, w)
        y0   = (h - size) // 2
        x0   = (w - size) // 2
        face_crop = img_bgr[y0:y0+size, x0:x0+size]
        status_prefix = "No face detected — using centre crop. "
    else:
        # Use largest face
        areas = [w * h for (_, _, w, h) in faces]
        x, y, w, h = faces[int(np.argmax(areas))]
        pad = int(0.1 * max(w, h))
        x1  = max(0, x - pad);  y1 = max(0, y - pad)
        x2  = min(img_bgr.shape[1], x + w + pad)
        y2  = min(img_bgr.shape[0], y + h + pad)
        face_crop = img_bgr[y1:y2, x1:x2]
        status_prefix = ""

        # Draw box on output image
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 200, 100), 2)

    # Inference
    with torch.no_grad():
        probs = torch.softmax(model(preprocess(face_crop)), dim=1).cpu().numpy()[0]

    top = int(np.argmax(probs))
    emotion, conf = EMOTIONS[top], probs[top]

    # Label on image
    label = f"{emotion.upper()}  {conf:.0%}"
    cv2.putText(img_bgr, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 100), 2, cv2.LINE_AA)

    out_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    status  = f"{status_prefix}**{emotion.upper()}** — {conf:.1%} confidence | device: `{DEVICE}`"

    return out_rgb, bar_chart(probs), status


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="FER Demo") as demo:
    gr.Markdown("## Facial Expression Recognition")
    gr.Markdown(
        f"EfficientNet-B2 · `{CHECKPOINT.name}` · device `{DEVICE}`\n\n"
        "**How to use:** Select *Webcam*, allow camera access, click the shutter button, "
        "then press **Analyse**. Or switch to *Upload* and choose a photo."
    )

    with gr.Row():
        with gr.Column(scale=1):
            cam = gr.Image(
                sources=["webcam", "upload"],
                type="numpy",
                label="Input (Webcam or Upload)",
                interactive=True,
            )
            btn = gr.Button("Analyse", variant="primary", size="lg")

        with gr.Column(scale=1):
            out_img   = gr.Image(label="Result", interactive=False)
            out_chart = gr.Plot(label="Probabilities")
            out_text  = gr.Markdown("Result will appear here.")

    # Trigger on button click OR automatically when the image changes
    btn.click(fn=predict, inputs=cam, outputs=[out_img, out_chart, out_text])
    cam.change(fn=predict, inputs=cam, outputs=[out_img, out_chart, out_text])

    gr.Markdown("---\n*Haar cascade · EfficientNet-B2 trained on FER2013 + CK+48*")

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True, theme=gr.themes.Soft())
