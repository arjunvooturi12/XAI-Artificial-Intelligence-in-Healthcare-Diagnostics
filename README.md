# XAI Healthcare Diagnostics — Project

> **Paper:** Explainable Artificial Intelligence in Healthcare Diagnostics  
> **Author:** Venkat Ram Reddy Baireddy, University of North Florida

---

## Project Structure

```
xai_project/
├── data/
│   ├── chestxray/          ← NIH ChestX-ray14 (download here)
│   └── isic/               ← ISIC 2019 Skin Lesion (download here)
│
├── src/
│   ├── preprocessing/
│   │   └── preprocess.py   ← Dataset classes, transforms, dataloaders
│   ├── models/             ← ResNet50, EfficientNet, ViT (Phase 2)
│   ├── explainability/     ← Grad-CAM, SHAP, LIME, TCAV (Phase 3)
│   └── evaluation/         ← Metrics, clinician eval tools (Phase 4)
│
├── notebooks/
│   ├── 01_preprocessing.ipynb     ← Phase 1 ✅
│   ├── 02_model_training.ipynb    ← Phase 2
│   ├── 03_explainability.ipynb    ← Phase 3
│   └── 04_evaluation.ipynb        ← Phase 4
│
├── scripts/
│   ├── setup.sh            ← One-time environment setup
│   └── download_data.py    ← Kaggle dataset downloader
│
├── outputs/
│   ├── models/             ← Saved model checkpoints
│   ├── explanations/       ← Grad-CAM heatmaps, SHAP plots
│   └── reports/            ← EDA plots, evaluation figures
│
└── requirements.txt
```

---

## Quickstart

### Step 1 — Install dependencies
```bash
cd xai_project
bash scripts/setup.sh
```

### Step 2 — Set up Kaggle API
1. Go to https://www.kaggle.com/settings → **Create New Token**
2. Place `kaggle.json` at `~/.kaggle/kaggle.json`
3. `chmod 600 ~/.kaggle/kaggle.json`

### Step 3 — Download datasets
```bash
python scripts/download_data.py --dataset both
```
> ChestX-ray14 ≈ 45 GB · ISIC 2019 ≈ 10 GB

### Step 4 — Run Phase 1 notebook
```bash
jupyter notebook notebooks/01_preprocessing.ipynb
```

---

## Phase Roadmap

| Phase | Weeks | Notebook | Status |
|-------|-------|----------|--------|
| 1. Data Collection & Preprocessing | 1–3 | `01_preprocessing.ipynb` | 🔄 In Progress |
| 2. Model Development (ResNet, EfficientNet, ViT) | 4–7 | `02_model_training.ipynb` | ⏳ |
| 3. Explainability (Grad-CAM, SHAP, LIME, TCAV) | 8–10 | `03_explainability.ipynb` | ⏳ |
| 4. Clinician Evaluation | 11–13 | `04_evaluation.ipynb` | ⏳ |
| 5. Robustness & Bias Analysis | 14–15 | `04_evaluation.ipynb` | ⏳ |
| 6. Integration & Dissemination | 16–18 | — | ⏳ |

---

## Google Colab Setup (Recommended for GPU)

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone or upload project
# Then install:
!pip install torch torchvision albumentations grad-cam shap lime captum -q

# Set up Kaggle
from google.colab import files
files.upload()  # upload kaggle.json
!mkdir ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

---

## Datasets

| Dataset | Task | Images | Classes | Size |
|---------|------|--------|---------|------|
| [NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data) | Multi-label classification | 112,120 | 14 diseases | ~45 GB |
| [ISIC 2019](https://www.kaggle.com/datasets/andrewmvd/isic-2019) | Multi-class classification | 25,331 | 8 lesion types | ~10 GB |
