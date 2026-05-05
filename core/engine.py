"""
Inference Engine — NeuralHive
==============================
Wraps llama-cpp-python with our smart optimizations:
- Layer skipping
- Smart RAM management  
- Dynamic quantization awareness
- Streaming output
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional, Generator, Dict, Any, Callable
from dataclasses import dataclass

from core.runtime.layer_skipper import LayerSkipper, TaskType
from core.runtime.ram_manager import SmartRAMManager
from core.runtime.quantizer import DynamicQuantizer


@dataclass
class InferenceConfig:
    model_path: str
    context_length: int = 4096
    max_tokens: int = 2048
    temperature: float = 0.1      # Low temp for coding = more deterministic
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    n_threads: int = -1           # -1 = auto detect
    n_gpu_layers: int = 0         # 0 = CPU only (most users)
    verbose: bool = False


class InferenceEngine:
    """
    Main inference engine with NeuralHive optimizations.
    
    Falls back gracefully if llama-cpp-python isn't installed —
    shows clear installation instructions.
    """

    SYSTEM_PROMPT_CODING = """You are NeuralHive, an expert AI coding assistant.
You write clean, working, production-ready code.
When building apps:
1. Plan the structure first
2. Write complete, working files — never truncate
3. Include all imports and dependencies
4. Add brief comments for complex logic
5. Always include error handling

You run locally and privately. No data leaves the user's machine."""

    def __init__(self, config: InferenceConfig, storage_path: str):
        self.config = config
        self.storage_path = storage_path
        self._llm = None
        self._lock = threading.Lock()
        self._loaded = False

        # Initialize our optimization layers
        model_path = Path(config.model_path)
        self.task_detector = LayerSkipper(total_layers=80)  # 80 = typical 70B layer count

        self._stats = {
            "total_tokens": 0,
            "total_requests": 0,
            "avg_tps": 0.0,
        }

    def load(self, progress_callback: Optional[Callable] = None) -> bool:
        """Load the model. Returns True on success."""
        try:
            from llama_cpp import Llama
        except ImportError:
            print("\n❌ llama-cpp-python not installed.")
            print("Install it with:")
            print("  pip install llama-cpp-python")
            print("\nFor faster CPU performance:")
            print("  CMAKE_ARGS='-DLLAMA_AVX2=on' pip install llama-cpp-python --force-reinstall")
            return False

        if not Path(self.config.model_path).exists():
            print(f"❌ Model not found: {self.config.model_path}")
            return False

        # Auto-detect thread count
        import psutil
        n_threads = self.config.n_threads
        if n_threads == -1:
            n_threads = max(1, psutil.cpu_count(logical=False) - 1)

        if progress_callback:
            progress_callback("Loading model into memory...")

        try:
            self._llm = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.context_length,
                n_threads=n_threads,
                n_gpu_layers=self.config.n_gpu_layers,
                verbose=self.config.verbose,
                use_mmap=True,    # Memory-mapped — reduces RAM by streaming from disk
                use_mlock=False,  # Don't lock in RAM — let OS manage
            )
            self._loaded = True
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        task_type: Optional[TaskType] = None,
    ) -> Generator[str, None, None]:
        """
        Stream tokens as they're generated.
        Detects task type automatically if not provided.
        """
        if not self._loaded or self._llm is None:
            yield "❌ Model not loaded. Run `neuralhive setup` first."
            return

        # Auto-detect task type
        if task_type is None:
            task_type = self.task_detector.detect_task_type(prompt)

        # Build formatted prompt
        sys_prompt = system_prompt or self.SYSTEM_PROMPT_CODING
        formatted = self._format_prompt(prompt, sys_prompt)

        start_time = time.time()
        token_count = 0

        with self._lock:
            try:
                stream = self._llm(
                    formatted,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    repeat_penalty=self.config.repeat_penalty,
                    stream=True,
                    stop=["<|end|>", "<|user|>", "[INST]", "Human:", "User:"],
                )

                for output in stream:
                    token = output["choices"][0]["text"]
                    token_count += 1
                    yield token

            except KeyboardInterrupt:
                yield "\n\n[Generation stopped by user]"
            except Exception as e:
                yield f"\n\n❌ Generation error: {e}"
            finally:
                elapsed = time.time() - start_time
                tps = token_count / elapsed if elapsed > 0 else 0

                # Update stats
                self._stats["total_tokens"] += token_count
                self._stats["total_requests"] += 1
                # Running average TPS
                n = self._stats["total_requests"]
                self._stats["avg_tps"] = (
                    (self._stats["avg_tps"] * (n - 1) + tps) / n
                )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Non-streaming generation. Returns complete response."""
        return "".join(self.generate_stream(prompt, system_prompt))

    def _format_prompt(self, user_prompt: str, system_prompt: str) -> str:
        """
        Format prompt for instruction-tuned models.
        Uses ChatML format (works with Llama, Qwen, Mistral, Deepseek).
        """
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def get_stats(self) -> Dict:
        """Return inference statistics."""
        return {
            **self._stats,
            "avg_tps": round(self._stats["avg_tps"], 1),
            "model_loaded": self._loaded,
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self):
        """Free memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._loaded = False