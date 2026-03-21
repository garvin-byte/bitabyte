"""Data sources for the next-gen bit viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


class ByteDataSource:
    """Lazy byte loader that can wrap numpy arrays or memory-mapped files."""

    def __init__(self, bytes_per_row: int = 16) -> None:
        self._bytes_per_row = max(1, bytes_per_row)
        self._bits_per_row = 64
        self._data: Optional[np.memmap | np.ndarray] = None
        self._byte_length = 0
        self._path: Optional[Path] = None

    @property
    def bytes_per_row(self) -> int:
        return self._bytes_per_row

    @bytes_per_row.setter
    def bytes_per_row(self, value: int) -> None:
        self._bytes_per_row = max(1, value)

    @property
    def byte_length(self) -> int:
        return self._byte_length

    @property
    def row_count(self) -> int:
        if self._byte_length == 0:
            return 0
        return (self._byte_length + self._bytes_per_row - 1) // self._bytes_per_row

    @property
    def source_path(self) -> Optional[Path]:
        return self._path

    @property
    def bit_length(self) -> int:
        return self._byte_length * 8

    def load_from_bytes(self, data: Sequence[int] | bytes | bytearray | np.ndarray) -> None:
        """Attach in-memory data."""
        if isinstance(data, np.ndarray):
            arr = np.asarray(data, dtype=np.uint8).reshape(-1)
        elif isinstance(data, (bytes, bytearray, memoryview)):
            arr = np.frombuffer(data, dtype=np.uint8)
        else:
            arr = np.asarray(data, dtype=np.uint8).reshape(-1)
        self._data = arr
        self._byte_length = arr.size
        self._path = None

    def load_from_file(self, path: str | Path) -> None:
        """Memory-map a file so we never read it fully."""
        file_path = Path(path)
        mapped = np.memmap(file_path, dtype=np.uint8, mode="r")
        self._data = mapped
        self._byte_length = mapped.size
        self._path = file_path

    def slice_row(self, row_index: int) -> np.ndarray:
        """Return a view of the bytes for a logical row."""
        if self._data is None:
            return np.empty(0, dtype=np.uint8)

        start = row_index * self._bytes_per_row
        end = min(start + self._bytes_per_row, self._byte_length)
        if start >= self._byte_length:
                return np.empty(0, dtype=np.uint8)
        return self._data[start:end]

    def slice_bits(self, row_index: int, bits_per_row: int) -> np.ndarray:
        """Return bit range for a logical row without unpacking entire file."""
        if self._data is None or self._byte_length == 0:
            return np.empty(0, dtype=np.uint8)

        total_bits = self.bit_length
        start_bit = row_index * bits_per_row
        if start_bit >= total_bits:
            return np.empty(0, dtype=np.uint8)

        end_bit = min(start_bit + bits_per_row, total_bits)
        start_byte = start_bit // 8
        end_byte = (end_bit + 7) // 8

        chunk = self._data[start_byte:end_byte]
        if chunk.size == 0:
            return np.empty(0, dtype=np.uint8)

        bits = np.unpackbits(chunk)
        offset = start_bit % 8
        bits = bits[offset : offset + (end_byte - start_byte) * 8]
        return bits[: end_bit - start_bit]

    def slice_region(self, start: int, length: int) -> np.ndarray:
        if self._data is None or length <= 0:
            return np.empty(0, dtype=np.uint8)
        end = min(start + length, self._byte_length)
        if start >= end:
            return np.empty(0, dtype=np.uint8)
        return self._data[start:end]

    def raw_bytes(self) -> bytes:
        if self._data is None:
            return b""
        return self._data.tobytes()
