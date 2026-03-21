"""Qt models for the next-gen viewer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, QVariant
from PyQt6.QtGui import QColor

from .data import ByteDataSource
from .colors import COLOR_NAME_TO_QCOLOR


@dataclass
class HeaderBand:
    """Represents one logical header row (e.g., bytes / bits / custom labels)."""

    labels: List[str] = field(default_factory=list)
    height: int = 20
    spans: Optional[List[tuple[int, int, str]]] = None


@dataclass
class ColumnDefinition:
    start_byte: int
    end_byte: int
    label: str
    display_format: str
    color_name: str = "None"
    unit: str = "byte"
    start_bit: int = 0
    total_bits: int = 8

    def normalized_byte_span(self, width: int) -> tuple[int, int] | None:
        if self.unit != "byte" or width <= 0:
            return None
        start = max(0, self.start_byte)
        end = min(self.end_byte, width - 1)
        if end < start:
            return None
        return start, end


@dataclass
class HeaderModel:
    """Container for multi-row header definitions."""

    bands: List[HeaderBand] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return len(self.bands)

    def label_for(self, band_index: int, section: int) -> str:
        if band_index < 0 or band_index >= len(self.bands):
            return ""
        band = self.bands[band_index]
        if band.spans:
            for start, length, text in band.spans:
                if start <= section < start + length:
                    return text
        if not band.labels:
            return ""
        return band.labels[section % len(band.labels)]

    def height_for(self, band_index: int) -> int:
        if band_index < 0 or band_index >= len(self.bands):
            return 20
        return self.bands[band_index].height


class ByteTableModel(QAbstractTableModel):
    """Data model that serves bytes lazily to the table view."""
    def __init__(self, data_source: ByteDataSource):
        super().__init__()
        self._source = data_source
        self._bytes_per_row = data_source.bytes_per_row
        self._bits_per_row = 64
        self._display_mode = "byte"  # "byte" or "bit"
        self._frames: Optional[List[tuple[int, int]]] = None
        self._frame_max_length = 0
        self.column_definitions: List[ColumnDefinition] = []
        self._column_span_data: dict[int, tuple[int, ColumnDefinition]] = {}
        self._column_span_lookup: dict[int, int] = {}
        self._column_backgrounds: dict[int, QColor] = {}
        self._rebuild_column_spans()

    def set_bytes_per_row(self, value: int):
        if value <= 0 or value == self._bytes_per_row:
            return
        self.beginResetModel()
        self._bytes_per_row = value
        self._source.bytes_per_row = value
        self._rebuild_column_spans()
        self.endResetModel()

    def set_bits_per_row(self, value: int):
        if value <= 0 or value == self._bits_per_row:
            return
        self.beginResetModel()
        self._bits_per_row = value
        self.endResetModel()

    def set_display_mode(self, mode: str):
        if mode == self._display_mode:
            return
        self.beginResetModel()
        self._display_mode = mode
        self._rebuild_column_spans()
        self.endResetModel()

    @property
    def display_mode(self) -> str:
        return self._display_mode

    def set_frames(self, frames: Optional[List[tuple[int, int]]]):
        self.beginResetModel()
        self._frames = frames
        if frames:
            self._frame_max_length = max((length for _, length in frames), default=0)
        else:
            self._frame_max_length = 0
        self.endResetModel()

    def has_frames(self) -> bool:
        return bool(self._frames)

    @property
    def frame_max_length(self) -> int:
        return self._frame_max_length

    # Qt model overrides
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        if self._display_mode == "byte":
            if self._frames:
                return len(self._frames)
            return self._source.row_count
        total_bits = self._source.bit_length
        if total_bits == 0:
            return 0
        return (total_bits + self._bits_per_row - 1) // self._bits_per_row

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return self._bytes_per_row if self._display_mode == "byte" else self._bits_per_row

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if not index.isValid():
            return QVariant()

        row = index.row()
        col = index.column()
        if self._display_mode == "byte":
            if self._frames:
                start, length = self._frames[row]
                chunk = self._source.slice_region(start, length)
            else:
                chunk = self._source.slice_row(row)
            value = chunk[col] if col < chunk.size else None
            if role == Qt.ItemDataRole.BackgroundRole:
                color = self._background_for_column(col)
                if color:
                    return color
            if role == Qt.ItemDataRole.DisplayRole:
                return "--" if value is None else f"{int(value):02X}"
            if role == Qt.ItemDataRole.UserRole:
                return value
            return QVariant()
        # Bit mode
        bits = self._source.slice_bits(row, self._bits_per_row)
        bit_val = bits[col] if col < bits.size else None

        if role == Qt.ItemDataRole.DisplayRole:
            if bit_val is None:
                return ""
            return "1" if int(bit_val) else "0"
        if role == Qt.ItemDataRole.UserRole:
            return bit_val
        return QVariant()

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        if role != Qt.ItemDataRole.DisplayRole:
            return QVariant()
        if orientation == Qt.Orientation.Horizontal:
            if self._display_mode == "byte":
                return f"{section:02X}"
            return str(section)
        else:
            if self._display_mode == "byte":
                start = section * self._bytes_per_row
                return f"{start:08X}"
            start_bit = section * self._bits_per_row
            return f"{start_bit:08d}"

    def _background_for_column(self, column: int) -> Optional[QColor]:
        if self._display_mode != "byte":
            return None
        return self._column_backgrounds.get(column)

    def notify_column_definitions_changed(self):
        self._rebuild_column_spans()
        self.layoutChanged.emit()

    def _rebuild_column_spans(self):
        self._column_span_data.clear()
        self._column_span_lookup.clear()
        self._column_backgrounds.clear()
        if self._display_mode != "byte":
            return
        for col_def in self.column_definitions:
            span = col_def.normalized_byte_span(self._bytes_per_row)
            if span is None:
                continue
            start, end = span
            color = COLOR_NAME_TO_QCOLOR.get(col_def.color_name)
            for col in range(start, end + 1):
                if color is not None:
                    self._column_backgrounds[col] = color
            if not col_def.label:
                continue
            length = end - start + 1
            self._column_span_data[start] = (length, col_def)
            for col in range(start, end + 1):
                self._column_span_lookup[col] = start

    def span_for_column(self, column: int):
        if self._display_mode != "byte":
            return None
        start = self._column_span_lookup.get(column)
        if start is None:
            return None
        data = self._column_span_data.get(start)
        if not data:
            return None
        length, col_def = data
        is_start = (start == column)
        return start, length, col_def, is_start

    def get_row_bytes(self, row: int):
        if self._display_mode != "byte":
            return np.empty(0, dtype=np.uint8)
        if self._frames:
            start, length = self._frames[row]
            return self._source.slice_region(start, length)
        return self._source.slice_row(row)
