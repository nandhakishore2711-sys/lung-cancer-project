# Getting a dataset

The code expects this folder layout under `data/` (this is the standard
`ImageFolder` / `image_dataset_from_directory` layout, so it works unchanged
for both the TensorFlow and PyTorch training scripts):

```
data/
  train/
    Benign/       *.png or *.jpg
    Malignant/
    Normal/
  val/
    Benign/
    Malignant/
    Normal/
  test/
    Benign/
    Malignant/
    Normal/
```

Class folder names must match exactly (case-sensitive) so they line up with
`utils/preprocessing.py::CLASS_NAMES`.

## Recommended dataset: IQ-OTH/NCCD Lung Cancer Dataset

This is the standard public dataset for exactly this task — thoracic CT
slices labeled Benign / Malignant / Normal, collected at the Iraq-Oncology
Teaching Hospital / National Center for Cancer Diseases. It's widely used in
published lung-cancer CNN papers, which makes it easy to cite comparable
accuracy numbers in your report.

1. Go to Kaggle and search **"IQ-OTH/NCCD Lung Cancer Dataset"** (or visit
   `kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset`).
2. Download it (you'll need a free Kaggle account). Options:
   - Manually download the zip from the website, or
   - Use the Kaggle CLI:
     ```bash
     pip install kaggle
     # place your kaggle.json API token in ~/.kaggle/kaggle.json first
     kaggle datasets download -d hamdallak/the-iqothnccd-lung-cancer-dataset
     unzip the-iqothnccd-lung-cancer-dataset.zip -d data_raw
     ```
3. The raw download comes as one folder per class with no train/val/test
   split. Use `scripts/split_dataset.py` (included in this project) to turn
   that into the `data/train|val|test/...` layout automatically:
   ```bash
   python scripts/split_dataset.py --input_dir data_raw --output_dir data --train 0.7 --val 0.15 --test 0.15
   ```

## Using your own dataset instead

Any CT-scan (or other medical image) classification dataset works as long
as you arrange it into the same `train/val/test` + class-subfolder layout
above. If your class names differ from Benign/Malignant/Normal, update
`CLASS_NAMES` in `utils/preprocessing.py` — the training scripts detect
class names from the folder structure automatically, but the Flask app's
UI badge coloring (`static/css/style.css`, `.tag.<ClassName>`) assumes those
three names, so add matching CSS rules if you rename classes.

## A note on class imbalance

Medical imaging datasets are almost always imbalanced (far more "Normal" or
"Benign" scans than "Malignant" ones). Both training scripts already handle
this with `class_weight="balanced"` — you don't need to manually balance the
folders, just make sure every class has at least a few dozen validation and
test images so the metrics are meaningful.
