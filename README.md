# XAI Healthcare Diagnostics

> **Explainable Artificial Intelligence in Healthcare Diagnostics**  
> Arjun Vooturi — University of North Florida

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview
A unified XAI framework that diagnoses diseases from medical images and explains why — validated by clinicians and tested for fairness across two imaging domains.

## Novel Contribution
No existing framework jointly integrates:
- ✅ Multi-method XAI (Grad-CAM + SHAP + LIME + TCAV)
- ✅ Two imaging domains (Radiology + Dermatology)
- ✅ Clinician-centered validation
- ✅ Fairness & bias testing

## Datasets
Download and place in `data/`:
- [NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data) — 112,120 X-rays, 14 diseases
- [ISIC 2019](https://www.kaggle.com/datasets/andrewmvd/isic-2019) — 25,331 skin images, 8 classes

## Installation
```bash
git clone https://github.com/arjunvooturi12/XAI-Artificial-Intelligence-in-Healthcare-Diagnostics.git
cd XAI-Artificial-Intelligence-in-Healthcare-Diagnostics
pip install -r requirements.txt
```

## Project Structure
