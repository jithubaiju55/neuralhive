@echo off
echo.
echo ============================================
echo   NeuralHive — Windows Installer
echo   Free Local AI Coding Agent
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found.
    echo Download from https://python.org and try again.
    pause
    exit /b 1
)

echo [1/4] Python found. Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo Trying with --break-system-packages flag...
    pip install -r requirements.txt --break-system-packages
)

echo.
echo [2/4] Installing llama-cpp-python with CPU optimizations...
echo This may take a few minutes...

:: Try AVX2 first (most modern CPUs)
set CMAKE_ARGS=-DLLAMA_AVX2=on
pip install llama-cpp-python --force-reinstall --no-cache-dir

if errorlevel 1 (
    echo AVX2 build failed, trying standard build...
    set CMAKE_ARGS=
    pip install llama-cpp-python --force-reinstall --no-cache-dir
)

echo.
echo [3/4] Installing NeuralHive CLI...
pip install -e .

echo.
echo [4/4] Verifying installation...
python -c "import llama_cpp; print('llama-cpp-python OK')"
python -c "import rich; print('rich OK')"
python -c "import click; print('click OK')"
python -c "import psutil; print('psutil OK')"

echo.
echo ============================================
echo   Installation Complete!
echo.
echo   Run setup:  neuralhive setup
echo   Get help:   neuralhive --help
echo ============================================
echo.
pause