"""
Smart RAM Manager — NeuralHive
==============================
Keeps hot layers in RAM, cold layers on disk.
Preloads next likely layers before they're needed.

The key insight: If you can predict which layers are needed
1-2 tokens ahead, you can preload from SSD while the CPU
processes the current token. Net result: near-zero wait time
even when model is larger than available RAM.
"""

import os
import time
import threading
import psutil
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path


@dataclass
class LayerCacheEntry:
    layer_index: int
    data: Any  # actual tensor/weight data
    size_bytes: int
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    is_pinned: bool = False  # pinned = never evict

    def touch(self):
        self.last_accessed = time.time()
        self.access_count += 1


class SmartRAMManager:
    """
    Manages which model layers live in RAM vs disk.
    
    Strategy:
    - Always keep first 3 and last 3 layers in RAM (pinned)
    - LRU eviction for middle layers
    - Prefetch next layers in background thread
    - Monitor RAM pressure and evict proactively
    
    This lets a 16GB RAM system run a 40GB model
    by streaming layers on demand with minimal latency.
    """

    RAM_SAFETY_BUFFER_GB = 2.0  # Always keep 2GB free for OS
    PREFETCH_AHEAD = 3           # Preload 3 layers ahead

    def __init__(self, total_layers: int, storage_path: str, max_ram_gb: Optional[float] = None):
        self.total_layers = total_layers
        self.storage_path = Path(storage_path)
        self.cache: OrderedDict[int, LayerCacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_queue: List[int] = []
        self._stop_prefetch = threading.Event()

        # Determine max RAM we can use
        total_ram = psutil.virtual_memory().total / (1024 ** 3)
        safety = self.RAM_SAFETY_BUFFER_GB
        self.max_ram_gb = max_ram_gb or max(total_ram - safety, 2.0)
        self.max_ram_bytes = int(self.max_ram_gb * 1024 ** 3)

        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_loads = 0

        # Start prefetch thread
        self._start_prefetch_thread()

    def _start_prefetch_thread(self):
        """Background thread that preloads layers before they're needed."""
        def prefetch_worker():
            while not self._stop_prefetch.is_set():
                if self._prefetch_queue:
                    layer_idx = self._prefetch_queue.pop(0)
                    if layer_idx not in self.cache:
                        self._load_layer_from_disk(layer_idx)
                time.sleep(0.01)  # 10ms polling

        self._prefetch_thread = threading.Thread(
            target=prefetch_worker,
            daemon=True,
            name="neuralhive-prefetch"
        )
        self._prefetch_thread.start()

    def get_layer(self, layer_idx: int) -> Any:
        """
        Get layer data. Loads from disk if not in RAM.
        Also schedules prefetch of upcoming layers.
        """
        with self._lock:
            if layer_idx in self.cache:
                self.cache_hits += 1
                entry = self.cache[layer_idx]
                entry.touch()
                # Move to end (most recently used)
                self.cache.move_to_end(layer_idx)
                # Schedule prefetch of next layers
                self._schedule_prefetch(layer_idx)
                return entry.data
            else:
                self.cache_misses += 1
                return self._load_layer_from_disk(layer_idx)

    def _load_layer_from_disk(self, layer_idx: int) -> Any:
        """Load a layer from storage into RAM."""
        layer_path = self.storage_path / f"layer_{layer_idx:04d}.bin"

        if not layer_path.exists():
            # Layer file doesn't exist yet (model not downloaded)
            return None

        self.total_loads += 1
        start = time.time()

        with open(layer_path, 'rb') as f:
            data = f.read()

        load_time = time.time() - start
        size_bytes = len(data)

        # Evict if needed before adding
        self._evict_if_needed(size_bytes)

        entry = LayerCacheEntry(
            layer_index=layer_idx,
            data=data,
            size_bytes=size_bytes,
        )

        # Pin first and last few layers
        if layer_idx < 3 or layer_idx >= self.total_layers - 3:
            entry.is_pinned = True

        with self._lock:
            self.cache[layer_idx] = entry

        return data

    def _evict_if_needed(self, incoming_bytes: int):
        """Evict least recently used layers to make room."""
        with self._lock:
            current_usage = self._current_ram_usage()

            while (current_usage + incoming_bytes > self.max_ram_bytes
                   and len(self.cache) > 0):
                # Find least recently used non-pinned layer
                evict_idx = None
                for idx, entry in self.cache.items():
                    if not entry.is_pinned:
                        evict_idx = idx
                        break

                if evict_idx is None:
                    break  # All layers pinned, can't evict

                evicted = self.cache.pop(evict_idx)
                current_usage -= evicted.size_bytes

    def _current_ram_usage(self) -> int:
        """Total bytes currently in cache."""
        return sum(e.size_bytes for e in self.cache.values())

    def _schedule_prefetch(self, current_layer: int):
        """Queue next few layers for background preloading."""
        for i in range(1, self.PREFETCH_AHEAD + 1):
            next_idx = current_layer + i
            if (next_idx < self.total_layers
                    and next_idx not in self.cache
                    and next_idx not in self._prefetch_queue):
                self._prefetch_queue.append(next_idx)

    def pin_layers(self, indices: List[int]):
        """Pin specific layers — they'll never be evicted."""
        with self._lock:
            for idx in indices:
                if idx in self.cache:
                    self.cache[idx].is_pinned = True

    def get_stats(self) -> Dict:
        """Return cache performance statistics."""
        with self._lock:
            usage_gb = self._current_ram_usage() / (1024 ** 3)
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

            return {
                "cached_layers": len(self.cache),
                "ram_usage_gb": round(usage_gb, 2),
                "max_ram_gb": round(self.max_ram_gb, 2),
                "cache_hit_rate_pct": round(hit_rate * 100, 1),
                "total_disk_loads": self.total_loads,
                "prefetch_queue_size": len(self._prefetch_queue),
            }

    def get_system_ram_info(self) -> Dict:
        """Get real system RAM information."""
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024 ** 3), 1),
            "available_gb": round(mem.available / (1024 ** 3), 1),
            "used_pct": mem.percent,
            "neuralhive_budget_gb": round(self.max_ram_gb, 1),
        }

    def shutdown(self):
        """Clean shutdown."""
        self._stop_prefetch.set()
        if self._prefetch_thread:
            self._prefetch_thread.join(timeout=2.0)