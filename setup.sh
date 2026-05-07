#!/usr/bin/env bash
# =============================================================
#  XAI Healthcare Project — Environment Setup Script
#  Run this ONCE to set up everything from scratch.
#  Compatible with: Google Colab, local Linux/Mac, WSL
# =============================================================

set -e  # Exit on error

echo "======================================================"
echo "  XAI Healthcare Diagnostics — Setup Script"
echo "======================================================"

# ── 1. Python version check ───────────────────────────────────
PYTHON=$(python3 --version 2>&1)
echo "[1/6] Python: $PYTHON"

# ── 2. Upgrade pip ───────────────────────────────────────────
echo "[2/6] Upgrading pip..."
pip install --upgrade pip -q

# ── 3. Install dependencies ───────────────────────────────────
echo "[3/6] Installing requirements..."
pip install -r requirements.txt -q

# ── 4. Kaggle API setup ───────────────────────────────────────
echo "[4/6] Setting up Kaggle API..."
echo ""
echo "  To download datasets automatically, you need a Kaggle API key."
echo "  Steps:"
echo "    1. Go to https://www.kaggle.com/settings"
echo "    2. Click 'Create New Token' → downloads kaggle.json"
echo "    3. Place kaggle.json at: ~/.kaggle/kaggle.json"
echo "    4. Run: chmod 600 ~/.kaggle/kaggle.json"
echo ""

if [ -f ~/.kaggle/kaggle.json ]; then
    echo "  ✓ kaggle.json found!"
else
    echo "  ✗ kaggle.json NOT found — you'll need to add it before downloading."
fi

# ── 5. Create data directories ───────────────────────────────
echo "[5/6] Creating data directories..."
mkdir -p data/chestxray/{train,val,test}
mkdir -p data/isic/{train,val,test}
mkdir -p outputs/{models,explanations,reports}

# ── 6. Verify PyTorch + GPU ───────────────────────────────────
echo "[6/6] Checking PyTorch + GPU..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  Running on CPU (training will be slow — Colab GPU recommended)')
"

echo ""
echo "======================================================"
echo "  Setup complete! Next: run scripts/download_data.py"
echo "======================================================"
