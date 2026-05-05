#!/bin/bash
set -e

echo ""
echo "============================================"
echo "  NeuralHive — Linux/Mac Installer"
echo "  Free Local AI Coding Agent"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found."
    echo "Install with: sudo apt install python3 python3-pip"
    exit 1
fi

PYTHON=$(command -v python3 || command -v python)
PIP=$(command -v pip3 || command -v pip)

echo "[1/4] Python found: $($PYTHON --version)"
echo "Installing base dependencies..."
$PIP install -r requirements.txt

echo ""
echo "[2/4] Installing llama-cpp-python (CPU optimized)..."
echo "Detecting CPU capabilities..."

# Check for AVX2 support
if grep -q "avx2" /proc/cpuinfo 2>/dev/null; then
    echo "AVX2 detected — using optimized build"
    CMAKE_ARGS="-DLLAMA_AVX2=on" $PIP install llama-cpp-python --force-reinstall --no-cache-dir
elif sysctl -n machdep.cpu.features 2>/dev/null | grep -q "AVX2"; then
    echo "AVX2 detected (Mac) — using optimized build"
    CMAKE_ARGS="-DLLAMA_AVX2=on" $PIP install llama-cpp-python --force-reinstall --no-cache-dir
else
    echo "Standard build (no AVX2)"
    $PIP install llama-cpp-python --force-reinstall --no-cache-dir
fi

echo ""
echo "[3/4] Installing NeuralHive CLI..."
$PIP install -e .

# Make scripts executable
chmod +x cli/main.py 2>/dev/null || true

echo ""
echo "[4/4] Verifying installation..."
$PYTHON -c "import llama_cpp; print('✅ llama-cpp-python')"
$PYTHON -c "import rich; print('✅ rich')"
$PYTHON -c "import click; print('✅ click')"
$PYTHON -c "import psutil; print('✅ psutil')"

echo ""
echo "============================================"
echo "  Installation Complete!"
echo ""
echo "  Run setup:  neuralhive setup"
echo "  Get help:   neuralhive --help"
echo "============================================"
echo ""