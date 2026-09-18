#!/usr/bin/env bash
# LungCT UNet++ Diagnostic Studio 1-Click Launcher for macOS and Linux

echo "====================================================================="
echo "          LungCT UNet++ Next-Gen AI Diagnostic Studio"
echo "====================================================================="
echo ""

# Check python
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Python 3 is not installed."
    exit 1
fi

echo "[1/3] Python executable: $($PYTHON_CMD --version)"

# Check virtualenv or dependencies
if [ ! -d ".venv" ]; then
    echo "[2/3] Creating virtual environment (.venv)..."
    $PYTHON_CMD -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt

echo "[3/3] Starting LungCT UNet++ Server..."
echo "====================================================================="
echo " Open your browser at: http://localhost:5000"
echo "====================================================================="
echo ""

python app.py --port 5000
