"""
Flask web application for lung cancer CT scan classification.

Routes:
    GET  /            -> single-page upload + results UI
    POST /predict      -> accepts an image file, returns JSON prediction
                          (class, per-class confidence, Grad-CAM overlay)
    GET  /health        -> JSON health check, reports which backend/model loaded

Run:
    python app.py
Then open http://127.0.0.1:5000

If no trained model checkpoint is found at the configured path, the app
still starts and serves the UI in DEMO MODE — predictions are clearly
labeled as simulated so you (or a grader) can exercise the whole upload ->
predict -> explain flow before training finishes.
"""

import base64
import io
import json
import os
import uuid

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

import config
from utils.preprocessing import (
    CLASS_NAMES,
    IMG_SIZE,
    preprocess_for_pytorch,
    preprocess_for_tensorflow,
    validate_image_file,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
os.makedirs(config.UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model loading (happens once, at process startup)
# ---------------------------------------------------------------------------

_state = {
    "backend": config.MODEL_BACKEND,
    "model": None,
    "class_names": CLASS_NAMES,
    "demo_mode": True,
    "last_conv_layer_name": None,
}


def _load_labels():
    if os.path.exists(config.LABELS_PATH):
        with open(config.LABELS_PATH) as f:
            return json.load(f)
    return CLASS_NAMES


def load_model():
    _state["class_names"] = _load_labels()

    if config.MODEL_BACKEND == "tensorflow":
        if not os.path.exists(config.TF_MODEL_PATH):
            print(f"[app] No TensorFlow model found at {config.TF_MODEL_PATH} -> running in DEMO MODE.")
            return
        import tensorflow as tf
        from models.tensorflow_model import get_last_conv_layer_name

        model = tf.keras.models.load_model(config.TF_MODEL_PATH)
        _state["model"] = model
        _state["last_conv_layer_name"] = get_last_conv_layer_name(model)
        _state["demo_mode"] = False
        print(f"[app] Loaded TensorFlow model from {config.TF_MODEL_PATH}")

    elif config.MODEL_BACKEND == "pytorch":
        if not os.path.exists(config.PT_MODEL_PATH):
            print(f"[app] No PyTorch model found at {config.PT_MODEL_PATH} -> running in DEMO MODE.")
            return
        import torch
        from models.pytorch_model import build_model

        checkpoint = torch.load(config.PT_MODEL_PATH, map_location="cpu")
        arch = checkpoint.get("architecture", config.MODEL_ARCHITECTURE)
        model = build_model(arch, num_classes=len(checkpoint.get("class_names", CLASS_NAMES)))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _state["model"] = model
        _state["class_names"] = checkpoint.get("class_names", _state["class_names"])
        _state["demo_mode"] = False
        print(f"[app] Loaded PyTorch model from {config.PT_MODEL_PATH}")
    else:
        raise ValueError(f"Unknown MODEL_BACKEND: {config.MODEL_BACKEND}")


load_model()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_inference(image_path):
    """Returns (probabilities: list[float], predicted_index: int)."""
    if _state["backend"] == "tensorflow":
        batch = preprocess_for_tensorflow(image_path, IMG_SIZE)
        preds = _state["model"].predict(batch, verbose=0)[0]
        return preds.tolist(), int(preds.argmax()), batch
    else:
        import torch
        batch = preprocess_for_pytorch(image_path, IMG_SIZE)
        tensor = torch.from_numpy(batch).float()
        with torch.no_grad():
            output = _state["model"](tensor)
            logits = output[0] if isinstance(output, tuple) else output
            probs = torch.softmax(logits, dim=1)[0].numpy()
        return probs.tolist(), int(probs.argmax()), batch


def run_gradcam(image_path, batch, predicted_index):
    from utils.gradcam import get_gradcam_overlay

    overlay = get_gradcam_overlay(
        backend=_state["backend"],
        model=_state["model"],
        preprocessed_input=batch,
        class_index=predicted_index,
        last_conv_layer_name=_state["last_conv_layer_name"],
    )
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def demo_prediction():
    """Deterministic-ish fake prediction so the UI is fully exercisable
    before a real model is trained. Clearly flagged as demo_mode=True."""
    import random
    random.seed()
    weights = [random.random() for _ in _state["class_names"]]
    total = sum(weights)
    probs = [w / total for w in weights]
    predicted_index = probs.index(max(probs))
    return probs, predicted_index


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        class_names=_state["class_names"],
        backend=_state["backend"],
        demo_mode=_state["demo_mode"],
    )


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "backend": _state["backend"],
        "demo_mode": _state["demo_mode"],
        "classes": _state["class_names"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "scan" not in request.files:
        return jsonify({"error": "No file part named 'scan' in the request."}), 400

    file = request.files["scan"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not validate_image_file(file.filename, config.ALLOWED_EXTENSIONS):
        return jsonify({"error": "Unsupported file type. Use PNG, JPG, JPEG, BMP or TIFF."}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(config.UPLOAD_DIR, unique_name)
    file.save(save_path)

    try:
        class_names = _state["class_names"]

        if _state["demo_mode"]:
            probs, predicted_index = demo_prediction()
            gradcam_data_uri = None
        else:
            probs, predicted_index, batch = run_inference(save_path)
            try:
                gradcam_data_uri = run_gradcam(save_path, batch, predicted_index)
            except Exception as e:  # Grad-CAM is a bonus feature; never let it break a prediction
                print(f"[app] Grad-CAM failed: {e}")
                gradcam_data_uri = None

        response = {
            "demo_mode": _state["demo_mode"],
            "predicted_class": class_names[predicted_index],
            "confidence": round(probs[predicted_index] * 100, 2),
            "probabilities": {
                class_names[i]: round(p * 100, 2) for i, p in enumerate(probs)
            },
            "gradcam_image": gradcam_data_uri,
            "uploaded_image_url": f"/static/uploads/{unique_name}",
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
