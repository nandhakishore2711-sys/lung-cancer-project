# Pulmonary CT Analysis — Novel Lung Cancer Prediction Using Deep Learning

A CNN-based pipeline that classifies thoracic CT scan slices as **Benign**,
**Malignant**, or **Normal**, with a Flask web app for live predictions and
Grad-CAM visual explanations. Implemented in **both TensorFlow/Keras and
PyTorch** so you can train/compare/report on either (or both).

```
Early Detection & Accuracy   -> custom CNN + transfer-learning (EfficientNet) options
Radiologist Support          -> Grad-CAM heatmap shows which region drove the prediction
Deployment & Web API         -> Flask app: upload a scan, get a prediction instantly
```

---

## 1. Project layout

```
lung_cancer_prediction/
├── app.py                       # Flask web app (the deployed product)
├── config.py                    # backend/model paths — edit this to switch TF <-> PyTorch
├── requirements.txt
├── train_tensorflow.py          # training script — Keras
├── train_pytorch.py             # training script — PyTorch
├── models/
│   ├── tensorflow_model.py      # custom CNN + EfficientNetB0 transfer model (Keras)
│   └── pytorch_model.py         # custom CNN + EfficientNet-B0 transfer model (PyTorch)
├── utils/
│   ├── preprocessing.py         # shared image loading/resizing (train + inference use the SAME code)
│   └── gradcam.py                # Grad-CAM for both backends
├── scripts/
│   └── split_dataset.py         # turns a flat per-class folder into train/val/test
├── data/
│   └── README.md                # how to get the IQ-OTH/NCCD dataset (or use your own)
├── saved_models/                # trained weights + metrics land here after training
├── templates/index.html         # single-page upload/results UI
└── static/                      # CSS, JS, uploaded scans
```

## 2. Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you only want one framework, delete the other's lines from
`requirements.txt` first (see the comment at the bottom of that file) — the
app only ever loads whichever backend is set in `config.py`.

## 3. Get a dataset

See **`data/README.md`** — it walks through downloading the IQ-OTH/NCCD
Lung Cancer Dataset from Kaggle and splitting it into `data/train|val|test/`.
You can substitute any dataset as long as it follows the same
`class-name-per-folder` layout.

## 4. Train a model

**TensorFlow:**
```bash
python train_tensorflow.py --data_dir data --epochs 30 --architecture custom
```

**PyTorch:**
```bash
python train_pytorch.py --data_dir data --epochs 30 --architecture custom
```

Either script also accepts `--architecture efficientnet` to train the
transfer-learning baseline instead — useful if your report wants a
"proposed model vs. baseline" comparison table.

Each run writes to `saved_models/`:
- the trained model (`lung_cnn_tf_custom.keras` or `lung_cnn_pt_custom.pt`)
- `labels.json` (class name <-> index mapping)
- `training_curves*.png` (loss/accuracy vs. epoch — good for your report)
- `confusion_matrix*.png`
- `classification_report*.txt` (precision/recall/F1 per class)

## 5. Run the web app

Edit `config.py` (or set an env var) to pick which trained model to serve:

```bash
export MODEL_BACKEND=tensorflow      # or pytorch
export MODEL_ARCHITECTURE=custom     # or efficientnet
python app.py
```

Open **http://127.0.0.1:5000**, drop in a CT slice, and click **Run
Analysis**. You'll get:
- predicted class + confidence
- a per-class confidence bar
- a Grad-CAM overlay toggle showing which region of the scan the model
  attended to

> **Before you've trained a model**, the app still runs — it serves the
> full UI in **DEMO MODE** with a clearly-labeled simulated prediction, so
> you can test the interface end-to-end while training runs separately.

## 6. How the pieces fit together

- `utils/preprocessing.py` is imported by *both* the training scripts and
  `app.py`. This matters: a model is only as good as the guarantee that
  training-time and inference-time preprocessing are identical, so there's
  exactly one place that resizes/normalizes images.
- `config.py` is the single switch between the two frameworks. `app.py`
  never hardcodes "tensorflow" or "pytorch" — it reads `_state["backend"]`
  and calls the matching branch in `run_inference()` / `run_gradcam()`.
- Grad-CAM (`utils/gradcam.py`) hooks the last convolutional layer of
  whichever model is loaded (`get_last_conv_layer_name()` for Keras, a
  named module reference for PyTorch) — you don't need to change anything
  when swapping architectures.

## 7. Suggested report structure

Since this looks like a semester/capstone project, here's a section outline
that maps directly onto artifacts this repo produces:

1. **Introduction** — motivation (early detection improves survival rates), problem statement.
2. **Literature Review** — prior CNN-based CT classification work (cite a few papers using IQ-OTH/NCCD for easy comparison).
3. **Dataset** — IQ-OTH/NCCD description, class distribution, `data/README.md` split methodology.
4. **Methodology** — architecture diagrams from `models/*.py` (draw the 4-block CNN), preprocessing pipeline, class-weighting for imbalance, augmentation strategy.
5. **Results** — `saved_models/training_curves*.png`, `confusion_matrix*.png`, `classification_report*.txt`; compare `custom` vs. `efficientnet` runs.
6. **Deployment** — screenshot the Flask app, describe the `/predict` API contract, mention Grad-CAM for interpretability.
7. **Conclusion & Future Work** — limitations (dataset size, 2D slices vs. full 3D volumes, single-institution data), next steps.

## 8. Deployment

The app is deploy-ready with a production WSGI server (`wsgi.py` + Gunicorn),
a `Dockerfile`, and configs for the common platforms. It will run in
**DEMO MODE** on first deploy if `saved_models/` is empty — that's expected;
train a model and redeploy (or mount/upload the checkpoint) to serve real
predictions.

**Before deploying**, trim `requirements.txt` to the ONE framework you're
actually using (see the note at the bottom of that file) — shipping both
TensorFlow and PyTorch roughly doubles image size and memory use, which
matters a lot on free-tier hosts.

### Option A — Docker (works on any host: Fly.io, Render, Railway, a VPS, etc.)
```bash
docker build -t lung-cancer-app .
docker run -p 8000:8000 \
  -e MODEL_BACKEND=tensorflow \
  -e MODEL_ARCHITECTURE=custom \
  lung-cancer-app
```
Open **http://localhost:8000**. To serve a trained model, mount it in:
```bash
docker run -p 8000:8000 -v $(pwd)/saved_models:/app/saved_models lung-cancer-app
```

### Option B — Render
1. Push this folder to a GitHub repo.
2. In Render: **New → Web Service → connect repo**. `render.yaml` is
   already set up (Docker runtime, port, health check at `/health`) —
   Render will auto-detect it.
3. Free tier gives 512MB RAM; if the build OOMs, trim `requirements.txt`
   to one framework as noted above.

### Option C — Railway
1. Push to GitHub, then in Railway: **New Project → Deploy from repo**.
2. Railway auto-detects the `Dockerfile`. Set env vars `MODEL_BACKEND`,
   `MODEL_ARCHITECTURE` under **Variables** if you need non-defaults.
3. Railway sets `$PORT` automatically — the Dockerfile's `CMD` already
   reads it.

### Option D — Heroku (buildpack, no Docker)
```bash
heroku create your-app-name
git push heroku main
```
`Procfile` and `runtime.txt` are already in place — Heroku will use the
Python buildpack, install `requirements.txt`, and run Gunicorn per the
`Procfile`.

### Health check
Every option above can be probed at `GET /health` — it returns which
backend loaded and whether the app is in demo mode:
```json
{"status": "ok", "backend": "tensorflow", "demo_mode": true, "classes": [...]}
```

## 9. Important limitations (say this in your report and viva)

- This is a **research/educational tool**, not a certified diagnostic
  device — the UI disclaimer says so explicitly, and it should.
- Public CT datasets like IQ-OTH/NCCD are relatively small and
  single-institution; real-world generalization needs external validation.
- The pipeline classifies 2D slices, not full 3D CT volumes — a common,
  reasonable simplification for a coursework-scale project, but worth
  naming as a limitation.
