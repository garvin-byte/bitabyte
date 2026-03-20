"""Frame sync dialog and controller for next-gen viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from .data import ByteDataSource
from .models import ByteTableModel, ColumnDefinition
from .models import HeaderModel, HeaderBand
from .headers import MultiRowHeaderView
from .colors import populate_color_combo

if TYPE_CHECKING:  # pragma: no cover
    from .main_window import NextGenBitViewerWindow


class FrameSyncDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Frame Sync Pattern")
        self.setMinimumWidth(440)
        self.setStyleSheet("""
            QGroupBox {
                background-color: #C9FFD8;
                border: 2px solid #33cc7a;
                border-radius: 6px;
                margin-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
        """)

        layout = QVBoxLayout(self)

        format_group = QGroupBox("Input Format")
        format_layout = QVBoxLayout()
        self.hex_radio = QRadioButton("Hex (0x414B or 414B)")
        self.hex_radio.setChecked(True)
        self.bin_radio = QRadioButton("Binary (1/0 or X/O)")
        format_layout.addWidget(self.hex_radio)
        format_layout.addWidget(self.bin_radio)
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("Sync Pattern:"))
        self.pattern_input = QLineEdit()
        self.pattern_input.setPlaceholderText("0x1ACFFC1D or 11001010")
        pattern_layout.addWidget(self.pattern_input)
        layout.addLayout(pattern_layout)

        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("Column Label:"))
        self.label_input = QLineEdit("Sync")
        label_layout.addWidget(self.label_input)
        layout.addLayout(label_layout)

        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Highlight Color:"))
        self.color_combo = QComboBox()
        populate_color_combo(self.color_combo, "Sky")
        color_layout.addWidget(self.color_combo)
        layout.addLayout(color_layout)

        display_group = QGroupBox("Display Format")
        display_layout = QVBoxLayout()
        self.display_hex_radio = QRadioButton("Hex")
        self.display_hex_radio.setChecked(True)
        self.display_bin_radio = QRadioButton("Binary")
        self.display_ascii_radio = QRadioButton("ASCII")
        display_layout.addWidget(self.display_hex_radio)
        display_layout.addWidget(self.display_bin_radio)
        display_layout.addWidget(self.display_ascii_radio)
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.pattern_input.textChanged.connect(self._update_info)
        self.hex_radio.toggled.connect(self._update_info)
        self._update_info()

    def _update_info(self):
        pattern = self.pattern_input.text().strip()
        if not pattern:
            self.info_label.setText("")
            return
        bits = self._parse_bits(pattern)
        if bits is None:
            self.info_label.setText("Invalid pattern.")
        else:
            self.info_label.setText(f"Pattern length: {len(bits)} bits")

    def _parse_bits(self, text: str) -> Optional[np.ndarray]:
        if not text:
            return None
        if self.hex_radio.isChecked():
            cleaned = text.replace(" ", "")
            if cleaned.startswith("0x") or cleaned.startswith("0X"):
                cleaned = cleaned[2:]
            if len(cleaned) == 0 or len(cleaned) % 2 != 0:
                return None
            try:
                byte_values = bytes.fromhex(cleaned)
                return np.unpackbits(np.frombuffer(byte_values, dtype=np.uint8))
            except ValueError:
                return None
        else:
            cleaned = text.replace(" ", "").upper()
            cleaned = cleaned.replace("X", "1").replace("O", "0")
            if not cleaned or any(ch not in "10" for ch in cleaned):
                return None
            return np.array([int(ch) for ch in cleaned], dtype=np.uint8)

    def get_pattern_bits(self) -> Optional[np.ndarray]:
        return self._parse_bits(self.pattern_input.text().strip())

    def get_label(self) -> str:
        text = self.label_input.text().strip()
        return text if text else "Sync"

    def get_color(self) -> str:
        return self.color_combo.currentText()

    def get_display_format(self) -> str:
        if self.display_ascii_radio.isChecked():
            return "ascii"
        if self.display_bin_radio.isChecked():
            return "binary"
        return "hex"


class FrameSyncController:
    """Handles framing workflow for the next-gen viewer."""

    def __init__(self, parent_window: NextGenBitViewerWindow, data_source: ByteDataSource, table_model: ByteTableModel):
        self.window = parent_window
        self.data_source = data_source
        self.model = table_model

    def run_dialog(self):
        if self.data_source.byte_length == 0:
            QMessageBox.information(self.window, "Frame Sync", "Load data before running frame sync.")
            return

        dialog = FrameSyncDialog(self.window)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        bits = dialog.get_pattern_bits()
        if bits is None or len(bits) == 0:
            QMessageBox.warning(self.window, "Frame Sync", "Invalid pattern.")
            return

        pattern_bytes = np.packbits(bits)
        frames = self._find_frames(pattern_bytes)
        if not frames:
            QMessageBox.information(self.window, "Frame Sync", "Pattern not found.")
            return

        self.model.set_frames(frames)
        self.window.status_label.setText(f"Frames detected: {len(frames)}")
        column_def = self._make_sync_column(dialog, bits)
        self.model.column_definitions.append(column_def)
        self.window.frame_label_spans = [(0, max(1, len(pattern_bytes)), dialog.get_label())]
        self.window.on_column_definitions_changed()
        self.window.on_frames_changed()

    def clear_frames(self, update_status: bool = True):
        self.model.set_frames(None)
        self.window.frame_label_spans = []
        if update_status:
            self.window.status_label.setText("Frames cleared")
        self.window.on_frames_changed()

    def _find_frames(self, pattern: np.ndarray) -> List[tuple[int, int]]:
        raw = self.data_source.raw_bytes()
        pat_bytes = bytes(pattern.tolist())
        if len(pat_bytes) == 0:
            return []

        positions = []
        start = 0
        while True:
            idx = raw.find(pat_bytes, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + 1

        if not positions:
            return []

        frames: List[tuple[int, int]] = []
        for i, pos in enumerate(positions):
            next_pos = positions[i + 1] if i + 1 < len(positions) else len(raw)
            length = max(0, next_pos - pos)
            frames.append((pos, length))
        return frames

    def _make_sync_column(self, dialog: FrameSyncDialog, bits: np.ndarray) -> ColumnDefinition:
        total_bits = len(bits)
        label = dialog.get_label()
        display_format = dialog.get_display_format()
        color = dialog.get_color()
        if total_bits % 8 == 0:
            byte_len = total_bits // 8
            return ColumnDefinition(
                start_byte=0,
                end_byte=max(byte_len - 1, 0),
                label=label,
                display_format=display_format,
                color_name=color,
                unit="byte",
            )
        return ColumnDefinition(
            start_byte=0,
            end_byte=0,
            label=label,
            display_format=display_format,
            color_name=color,
            unit="bit",
            start_bit=0,
            total_bits=total_bits,
        )
