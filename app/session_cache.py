"""LRU=1 session cache shared by image and video pipelines.

DirectML keeps each held ONNX Runtime session's VRAM live until the
session object is released. Caching multiple multi-hundred-MB models
simultaneously is what made the host freeze under model-switching: shared
GPU memory spilled into system RAM, then into the page file. Holding at
most one session caps that cost — switching pays a re-load, switching
back pays it again, but neither path can exhaust VRAM.
"""
from __future__ import annotations

import gc
import threading
from typing import Callable, Generic, Optional, TypeVar

T = TypeVar("T")


class SingleSessionCache(Generic[T]):
    """Thread-safe one-slot session cache.

    ``get(key)`` returns the cached session if its key matches, otherwise
    evicts the current entry (dropping its VRAM with ``gc.collect()``)
    and loads a fresh one for the new key.
    """

    def __init__(self, loader: Callable[[str], T]) -> None:
        self._loader = loader
        self._key: Optional[str] = None
        self._session: Optional[T] = None
        self._lock = threading.Lock()

    def get(self, key: str) -> T:
        with self._lock:
            if key != self._key:
                self._evict_locked()
                self._session = self._loader(key)
                self._key = key
            assert self._session is not None
            return self._session

    def loaded(self) -> list[str]:
        with self._lock:
            return [self._key] if self._key is not None else []

    def clear(self) -> None:
        with self._lock:
            self._evict_locked()

    def _evict_locked(self) -> None:
        if self._session is None and self._key is None:
            return
        self._session = None
        self._key = None
        gc.collect()
