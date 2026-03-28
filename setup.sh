#!/usr/bin/env bash
# =============================================================================
#  Local AI Dashcam Incident Explainer — Setup Script
#  Run: chmod +x setup.sh && ./setup.sh
# =============================================================================
set -e

echo "============================================================"
echo "  Local AI Dashcam Incident Explainer — Setup"
echo "============================================================"

# ── 1. Python version check ──────────────────────────────────────────────────
PYTHON=$(command -v python3 || true)
if [ -z "$PYTHON" ]; then
  echo "[ERROR] python3 not found. Install Python 3.10+ first."
  exit 1
fi
PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
MIN_VERSION="3.10"
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"; then
  echo "[OK] Python $PY_VERSION"
else
  echo "[ERROR] Python 3.10+ required, found $PY_VERSION"
  exit 1
fi

# ── 2. FFmpeg check ──────────────────────────────────────────────────────────
if ! command -v ffmpeg &> /dev/null; then
  echo "[WARN] ffmpeg not found. Installing via Homebrew..."
  if command -v brew &> /dev/null; then
    brew install ffmpeg
  else
    echo "[ERROR] Homebrew not found. Install ffmpeg manually: https://ffmpeg.org/"
    exit 1
  fi
else
  echo "[OK] ffmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
fi

# ── 3. Ollama check ──────────────────────────────────────────────────────────
if ! command -v ollama &> /dev/null; then
  echo "[WARN] ollama not found. Install from https://ollama.com/ then re-run."
  echo "       After install, run: ollama pull minicpm-v"
  echo "       Then re-run this script."
  exit 1
else
  echo "[OK] Ollama $(ollama --version 2>&1 | head -1)"
fi

# ── 4. Virtual environment ───────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "[*] Creating virtual environment..."
  $PYTHON -m venv .venv
fi
source .venv/bin/activate
echo "[OK] Virtual env activated"

# ── 5. Install Python dependencies ──────────────────────────────────────────
echo "[*] Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "[OK] Dependencies installed"

# ── 6. NLTK data for evaluation ─────────────────────────────────────────────
echo "[*] Downloading NLTK punkt tokenizer..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"
echo "[OK] NLTK data ready"

# ── 7. Pull MiniCPM-V via Ollama ────────────────────────────────────────────
echo "[*] Pulling MiniCPM-V 2.6 (vision model) via Ollama (may take a few minutes)..."
ollama pull minicpm-v 2>&1 | tail -5 || echo "[WARN] Could not pull minicpm-v — check Ollama is running."

echo "[*] Pulling Mistral (report LLM) via Ollama..."
ollama pull mistral 2>&1 | tail -5 || echo "[WARN] Could not pull mistral."

# ── 8. Create output dirs ────────────────────────────────────────────────────
mkdir -p outputs/keyframes outputs/incidents outputs/reports data/samples

echo ""
echo "============================================================"
echo "  ✅  Setup complete!"
echo "  Activate env:   source .venv/bin/activate"
echo "  Run pipeline:   python3 src/phase3/pipeline.py --video data/samples/test.mp4"
echo "  Launch UI:      python3 src/phase4/app.py"
echo "============================================================"
