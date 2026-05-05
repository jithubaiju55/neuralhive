"""
Model Selector & Downloader — NeuralHive
==========================================
Automatically picks the best model for your hardware.
Downloads from HuggingFace. Supports all major open models.
"""

import os
import sys
import json
import time
import psutil
import requests
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum


class ModelTier(Enum):
    MICRO = "micro"       # 8GB RAM — 7-8B models
    SMALL = "small"       # 16GB RAM — 13-14B models
    MEDIUM = "medium"     # 24GB RAM — 32B models
    LARGE = "large"       # 48GB RAM — 70B models
    XLARGE = "xlarge"     # 128GB+ RAM — future 405B


@dataclass
class ModelConfig:
    name: str
    display_name: str
    huggingface_repo: str
    filename: str
    size_gb: float
    ram_required_gb: float
    tier: ModelTier
    coding_score: float      # 0-100, coding benchmark score
    speed_tps: float         # approximate tokens/sec on mid CPU
    description: str
    is_recommended: bool = False


# All free, open weight models — best for coding
AVAILABLE_MODELS: List[ModelConfig] = [
    # MICRO TIER — 8GB RAM
    ModelConfig(
        name="llama3.1-8b",
        display_name="Llama 3.1 8B",
        huggingface_repo="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        size_gb=4.9,
        ram_required_gb=7.0,
        tier=ModelTier.MICRO,
        coding_score=68,
        speed_tps=20,
        description="Fast, capable. Good for most coding tasks.",
    ),
    ModelConfig(
        name="qwen2.5-coder-7b",
        display_name="Qwen 2.5 Coder 7B",
        huggingface_repo="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        size_gb=4.7,
        ram_required_gb=7.0,
        tier=ModelTier.MICRO,
        coding_score=72,
        speed_tps=22,
        description="Best coding model at 7B size. Beats Llama on code.",
        is_recommended=True,
    ),

    # SMALL TIER — 16GB RAM
    ModelConfig(
        name="qwen2.5-coder-14b",
        display_name="Qwen 2.5 Coder 14B",
        huggingface_repo="Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
        filename="qwen2.5-coder-14b-instruct-q4_k_m.gguf",
        size_gb=9.0,
        ram_required_gb=12.0,
        tier=ModelTier.SMALL,
        coding_score=79,
        speed_tps=12,
        description="Strong coding, runs on 16GB RAM laptops.",
        is_recommended=True,
    ),
    ModelConfig(
        name="deepseek-coder-15b",
        display_name="Deepseek Coder 15B",
        huggingface_repo="TheBloke/deepseek-coder-15B-instruct-GGUF",
        filename="deepseek-coder-15b-instruct.Q4_K_M.gguf",
        size_gb=9.3,
        ram_required_gb=12.0,
        tier=ModelTier.SMALL,
        coding_score=78,
        speed_tps=11,
        description="Excellent for code completion and debugging.",
    ),

    # MEDIUM TIER — 24GB RAM
    ModelConfig(
        name="qwen2.5-coder-32b",
        display_name="Qwen 2.5 Coder 32B",
        huggingface_repo="Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        filename="qwen2.5-coder-32b-instruct-q4_k_m.gguf",
        size_gb=19.4,
        ram_required_gb=24.0,
        tier=ModelTier.MEDIUM,
        coding_score=85,
        speed_tps=6,
        description="Near GPT-4 coding quality. Best at this tier.",
        is_recommended=True,
    ),

    # LARGE TIER — 48GB RAM
    ModelConfig(
        name="llama3.1-70b",
        display_name="Llama 3.1 70B",
        huggingface_repo="bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        filename="Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
        size_gb=40.0,
        ram_required_gb=48.0,
        tier=ModelTier.LARGE,
        coding_score=82,
        speed_tps=3,
        description="Meta's flagship. Strong reasoning and coding.",
    ),
    ModelConfig(
        name="qwen2.5-72b",
        display_name="Qwen 2.5 72B",
        huggingface_repo="Qwen/Qwen2.5-72B-Instruct-GGUF",
        filename="qwen2.5-72b-instruct-q4_k_m.gguf",
        size_gb=43.0,
        ram_required_gb=52.0,
        tier=ModelTier.LARGE,
        coding_score=87,
        speed_tps=2,
        description="Top tier open model. Excellent on all tasks.",
        is_recommended=True,
    ),
]


class ModelSelector:
    """
    Detects system specs and recommends the best model.
    With NeuralHive runtime, models punch above their weight class.
    """

    # NeuralHive runtime boost — our engine makes models run
    # as if system has ~1.5-2x more RAM
    RUNTIME_BOOST_FACTOR = 1.6

    def __init__(self):
        self.ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        self.effective_ram_gb = self.ram_gb * self.RUNTIME_BOOST_FACTOR

    def get_compatible_models(self) -> List[ModelConfig]:
        """Models that can run on this system with NeuralHive runtime."""
        return [
            m for m in AVAILABLE_MODELS
            if m.ram_required_gb <= self.effective_ram_gb
        ]

    def get_recommended(self) -> Optional[ModelConfig]:
        """Get the single best model for this system."""
        compatible = self.get_compatible_models()
        if not compatible:
            return None

        # Among compatible models, pick highest coding score
        # that is also marked as recommended
        recommended = [m for m in compatible if m.is_recommended]
        if recommended:
            return max(recommended, key=lambda m: m.coding_score)

        return max(compatible, key=lambda m: m.coding_score)

    def get_system_info(self) -> Dict:
        """Get system information relevant to model selection."""
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_count(logical=False)

        return {
            "ram_gb": round(self.ram_gb, 1),
            "effective_ram_gb_with_runtime": round(self.effective_ram_gb, 1),
            "cpu_cores": cpu,
            "platform": sys.platform,
            "compatible_model_count": len(self.get_compatible_models()),
        }


class ModelDownloader:
    """Download models from HuggingFace with resume support."""

    HF_BASE = "https://huggingface.co"

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model: ModelConfig) -> Path:
        return self.models_dir / model.filename

    def is_downloaded(self, model: ModelConfig) -> bool:
        path = self.get_model_path(model)
        if not path.exists():
            return False
        # Check file size roughly matches expected
        actual_gb = path.stat().st_size / (1024 ** 3)
        return actual_gb >= model.size_gb * 0.95

    def download(self, model: ModelConfig, progress_callback=None) -> Path:
        """
        Download model with resume support.
        Returns path to downloaded file.
        """
        output_path = self.get_model_path(model)
        url = f"{self.HF_BASE}/{model.huggingface_repo}/resolve/main/{model.filename}"

        # Check existing partial download
        existing_size = output_path.stat().st_size if output_path.exists() else 0
        headers = {}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"

        response = requests.get(url, headers=headers, stream=True, timeout=30)

        if response.status_code == 416:
            # Already fully downloaded
            return output_path

        if response.status_code not in (200, 206):
            raise RuntimeError(f"Download failed: HTTP {response.status_code}")

        total_size = int(response.headers.get('content-length', 0))
        if existing_size > 0:
            total_size += existing_size

        mode = 'ab' if existing_size > 0 else 'wb'
        downloaded = existing_size
        chunk_size = 1024 * 1024  # 1MB chunks

        with open(output_path, mode) as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total_size:
                        progress_callback(downloaded, total_size)

        return output_path

    def get_download_url(self, model: ModelConfig) -> str:
        return f"{self.HF_BASE}/{model.huggingface_repo}/resolve/main/{model.filename}"


class GGUFScanner:
    """
    Scans the neuralhive_models folder for ANY .gguf file —
    even ones not in our known list. Lets users manually download
    any model from HuggingFace and have it just work.
    """

    # Known model name patterns → display info
    # If filename contains any of these keys, we use that info
    KNOWN_PATTERNS = {
        "llama-3.1-8b":   {"display": "Llama 3.1 8B",         "score": 68, "tps": 20},
        "llama-3.1-70b":  {"display": "Llama 3.1 70B",        "score": 82, "tps": 3},
        "llama-3-8b":     {"display": "Llama 3 8B",           "score": 65, "tps": 20},
        "llama-3-70b":    {"display": "Llama 3 70B",          "score": 80, "tps": 3},
        "qwen2.5-coder-7b":  {"display": "Qwen 2.5 Coder 7B",  "score": 72, "tps": 22},
        "qwen2.5-coder-14b": {"display": "Qwen 2.5 Coder 14B", "score": 79, "tps": 12},
        "qwen2.5-coder-32b": {"display": "Qwen 2.5 Coder 32B", "score": 85, "tps": 6},
        "qwen2.5-72b":    {"display": "Qwen 2.5 72B",         "score": 87, "tps": 2},
        "deepseek-coder-15b": {"display": "Deepseek Coder 15B","score": 78, "tps": 11},
        "deepseek-coder-33b": {"display": "Deepseek Coder 33B","score": 82, "tps": 5},
        "mistral-7b":     {"display": "Mistral 7B",           "score": 65, "tps": 20},
        "mixtral-8x7b":   {"display": "Mixtral 8x7B",         "score": 75, "tps": 8},
        "codellama":      {"display": "Code Llama",           "score": 70, "tps": 15},
        "phi-3":          {"display": "Phi-3",                "score": 68, "tps": 25},
        "gemma-2":        {"display": "Gemma 2",              "score": 67, "tps": 20},
        "starcoder2":     {"display": "StarCoder2",           "score": 74, "tps": 12},
    }

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)

    def scan(self) -> List[Dict]:
        """
        Scan models folder for .gguf files.
        Returns list of dicts with model info — works for ANY gguf file.
        """
        if not self.models_dir.exists():
            return []

        found = []
        for f in sorted(self.models_dir.glob("*.gguf")):
            size_gb = f.stat().st_size / (1024 ** 3)
            info = self._identify(f.name, size_gb)
            found.append({
                "path": str(f),
                "filename": f.name,
                "size_gb": round(size_gb, 1),
                "display_name": info["display"],
                "coding_score": info["score"],
                "speed_tps": info["tps"],
                "ram_required_gb": self._estimate_ram(size_gb),
                "is_known": info["known"],
            })

        return found

    def _identify(self, filename: str, size_gb: float) -> Dict:
        """Match filename against known patterns. Falls back to generic info."""
        fname_lower = filename.lower()

        for pattern, info in self.KNOWN_PATTERNS.items():
            if pattern in fname_lower:
                return {**info, "known": True}

        # Unknown model — estimate quality from size
        if size_gb > 35:
            score, tps, label = 82, 3,  "Large Model (70B class)"
        elif size_gb > 15:
            score, tps, label = 78, 7,  "Medium-Large Model (32B class)"
        elif size_gb > 7:
            score, tps, label = 72, 12, "Medium Model (13-14B class)"
        else:
            score, tps, label = 65, 20, "Small Model (7B class)"

        # Try to extract a clean display name from filename
        display = filename.replace(".gguf", "").replace("-", " ").replace("_", " ")
        display = " ".join(w.capitalize() for w in display.split()[:6])

        return {"display": f"{display} ({label})", "score": score, "tps": tps, "known": False}

    def _estimate_ram(self, size_gb: float) -> float:
        """Estimate RAM needed from file size (quantized model)."""
        # Rough rule: GGUF size * 1.2 for overhead
        return round(size_gb * 1.2 + 1.0, 1)

    def get_best_local(self) -> Optional[Dict]:
        """Return the best locally available model by coding score."""
        models = self.scan()
        if not models:
            return None
        return max(models, key=lambda m: m["coding_score"])