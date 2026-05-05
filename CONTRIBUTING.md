# Contributing to NeuralHive

First — thank you. Every contribution matters.

## Quick Start

```bash
git clone https://github.com/yourusername/neuralhive
cd neuralhive
pip install -r requirements.txt
pip install -e .
python -m pytest tests/ -v
```

If all tests pass, you're ready to contribute.

## What We Need Most

### 1. Hardware Testers (No coding required)

Run NeuralHive on your system and report:

- Your hardware specs (RAM, CPU, OS)
- Which model you ran
- Actual tokens/second you got
- Any errors or issues

Open a GitHub Issue with label `benchmark`.

### 2. Runtime Optimization (C++/Python)

The core innovation is in `core/runtime/`.
Key areas:

- `layer_skipper.py` — improve layer importance scoring
- `ram_manager.py` — better prefetch prediction
- `quantizer.py` — smarter per-layer precision assignment

### 3. Agent Loop Improvements (Python)

`agent/loop.py` — make the coding agent smarter:

- Better error parsing
- Smarter run command detection
- Multi-file dependency awareness
- Project context understanding

### 4. New Model Support

Add models to `core/models/selector.py`:

```python
ModelConfig(
    name="your-model-id",
    display_name="Your Model Name",
    huggingface_repo="org/repo-name",
    filename="model-file.gguf",
    size_gb=X.X,
    ram_required_gb=XX.0,
    tier=ModelTier.MEDIUM,
    coding_score=XX,
    speed_tps=X,
    description="Brief description",
)
```

### 5. Documentation & Translation

- Improve README clarity
- Add language-specific docs (Hindi, Portuguese, Arabic, etc.)
- Write tutorials

## Code Style

- Python 3.9+
- Type hints where practical
- Docstrings on all classes and public methods
- Tests for new features (`tests/test_core.py`)

## Pull Request Process

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Make changes
4. Run tests: `python -m pytest tests/ -v`
5. Commit: `git commit -m "Add: brief description"`
6. Push and open PR

## Reporting Bugs

Open a GitHub Issue with:

- Your OS and Python version
- Your hardware (RAM, CPU)
- Exact error message
- Steps to reproduce

## Questions?

Open a GitHub Discussion. No question is too basic.

---
