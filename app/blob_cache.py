"""Bounded LRU cache for API result blobs (PNG bytes).

Capping the cache by count alone lets a handful of 4K PNGs balloon
past 2 GB resident before any eviction fires. Bound by bytes too:
oldest entries drop as soon as either limit is exceeded.
"""
from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from typing import Optional


class BlobCache:
    """Thread-safe FIFO/LRU cache with both an entry count and a byte limit."""

    def __init__(self, max_count: int, max_bytes: int) -> None:
        self._max_count = max_count
        self._max_bytes = max_bytes
        self._data: OrderedDict[str, bytes] = OrderedDict()
        self._total_bytes = 0
        self._lock = threading.Lock()

    def add(self, data: bytes) -> str:
        key = uuid.uuid4().hex[:12]
        with self._lock:
            self._data[key] = data
            self._total_bytes += len(data)
            self._evict_locked()
        return key

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            return self._data.get(key)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def _evict_locked(self) -> None:
        while (
            len(self._data) > self._max_count
            or self._total_bytes > self._max_bytes
        ):
            _, evicted = self._data.popitem(last=False)
            self._total_bytes -= len(evicted)
