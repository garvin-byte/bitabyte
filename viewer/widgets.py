"""Supporting widgets: LogScaleSpinBox, FieldInspectorWidget, FieldStatisticsWidget, TextDisplayWidget."""

import math
import numpy as np
from collections import Counter
from common.bit_view_utils import (
    apply_bit_order,
    bits_to_ascii,
    bits_to_int,
    build_highlight_intervals,
    is_bit_highlighted,
)
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
                              QSizePolicy, QSpinBox, QDoubleSpinBox, QTextEdit, QTableWidget,
                              QTableWidgetItem, QHeaderView, QAbstractItemView, QApplication,
                              QButtonGroup, QRadioButton)
from PyQt6.QtCore import Qt, QRect, QSize
from PyQt6.QtGui import QColor, QFont, QPainter, QPen


class LogScaleSpinBox(QDoubleSpinBox):
    """Spin box whose arrows scale the value by powers of ten."""

    def __init__(self):
        super().__init__()
        self.setDecimals(6)
        self.setRange(0.000001, 1_000_000.0)
        self.setKeyboardTracking(False)

    def stepBy(self, steps: int) -> None:
        value = self.value()
        if steps > 0:
            for _ in range(steps):
                value *= 10
        elif steps < 0:
            for _ in range(-steps):
                value /= 10

        value = max(self.minimum(), min(self.maximum(), value))
        self.setValue(value)


class FieldInspectorWidget(QWidget):
    """Displays textual representations of selected bits for each frame."""

    def __init__(self):
        super().__init__()
        self.frames_bits = []
        self.bit_order = "msb"
        self.display_mode = "hex"
        self.scale_factor = 1.0
        self.parent_window = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(6)

        # Bit order controls
        order_layout = QHBoxLayout()
        order_layout.addWidget(QLabel("Bit Order:"))

        self.order_group = QButtonGroup(self)
        self.msb_radio = QRadioButton("MSBF")
        self.msb_radio.setChecked(True)
        self.order_group.addButton(self.msb_radio)
        order_layout.addWidget(self.msb_radio)

        self.lsb_radio = QRadioButton("LSBF")
        self.order_group.addButton(self.lsb_radio)
        order_layout.addWidget(self.lsb_radio)
        order_layout.addStretch()

        self.msb_radio.toggled.connect(self._on_bit_order_changed)
        self.lsb_radio.toggled.connect(self._on_bit_order_changed)

        layout.addLayout(order_layout)

        # Display mode controls
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("View As:"))

        self.mode_group = QButtonGroup(self)
        self.mode_buttons = []
        modes = [
            ("Hex", "hex"),
            ("Binary", "binary"),
            ("Decimal", "unsigned"),
            ("Two's Complement", "twos"),
            ("ASCII", "ascii"),
        ]

        for text, mode in modes:
            btn = QRadioButton(text)
            if mode == "hex":
                btn.setChecked(True)
            self.mode_group.addButton(btn)
            btn.mode_value = mode  # type: ignore[attr-defined]
            btn.toggled.connect(self._on_mode_changed)
            mode_layout.addWidget(btn)
            self.mode_buttons.append(btn)

        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # Scale control
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Scale:"))
        self.scale_spin = LogScaleSpinBox()
        self.scale_spin.setValue(1.0)
        self.scale_spin.valueChanged.connect(self._on_scale_changed)
        scale_layout.addWidget(self.scale_spin)
        scale_layout.addStretch()
        layout.addLayout(scale_layout)

        # Output area
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Consolas", 10))
        self.output.setPlaceholderText("Select column(s) in framed byte mode to inspect values.")
        layout.addWidget(self.output, stretch=1)

    def _on_bit_order_changed(self):
        self.bit_order = "msb" if self.msb_radio.isChecked() else "lsb"
        self._render_output()
        self._request_viewer_refresh()

    def _on_mode_changed(self):
        for btn in self.mode_buttons:
            if btn.isChecked():
                self.display_mode = btn.mode_value  # type: ignore[attr-defined]
                break
        self._render_output()
        self._request_viewer_refresh()

    def _on_scale_changed(self, value):
        self.scale_factor = value
        self._render_output()
        self._request_viewer_refresh()

    def _request_viewer_refresh(self):
        if self.parent_window and hasattr(self.parent_window, "byte_table"):
            self.parent_window.byte_table._update_live_bit_viewer()

    def set_frames_bits(self, frames_bits):
        """Accept list of numpy arrays (one per frame) and update the view."""
        self.frames_bits = frames_bits or []
        self._render_output()

    def _render_output(self):
        if not self.frames_bits:
            self.output.setPlainText("Select column(s) in framed byte mode to inspect values.")
            return

        lines = []
        for frame_idx, bits in enumerate(self.frames_bits):
            formatted = self._format_bits(bits)
            lines.append(f"F{frame_idx}: {formatted}")

        self.output.setPlainText("\n".join(lines))

    def _format_bits(self, bits):
        if bits is None or len(bits) == 0:
            return "(no data)"

        ordered_bits = self._apply_bit_order(bits)

        if self.display_mode == "binary":
            bit_chars = "".join(str(int(b)) for b in ordered_bits.tolist())
            grouped = " ".join(bit_chars[i:i + 8] for i in range(0, len(bit_chars), 8))
            return grouped

        if self.display_mode == "ascii":
            return self._bits_to_ascii(ordered_bits)

        value = self._bits_to_int(ordered_bits)
        if value is None:
            return "(no data)"

        if self.display_mode == "hex":
            width = max(1, (len(ordered_bits) + 3) // 4)
            return f"0x{value:0{width}X}"

        if self.display_mode == "twos":
            bit_len = len(ordered_bits)
            if bit_len == 0:
                return "0"
            max_val = 1 << bit_len
            signed = value - max_val if ordered_bits[0] == 1 else value
            return self._format_scaled_value(signed)

        # Default to unsigned decimal
        return self._format_scaled_value(value)

    def _apply_bit_order(self, bits):
        return apply_bit_order(bits, self.bit_order)

    def _bits_to_int(self, bits):
        return bits_to_int(bits)

    def _bits_to_ascii(self, bits):
        return bits_to_ascii(bits)

    def _format_scaled_value(self, value):
        if value is None:
            return "(no data)"

        if self.scale_factor == 1.0:
            return str(value)

        scaled = value * self.scale_factor
        # Use fixed-point with 4 decimals to mirror default spin box precision
        return f"{scaled:.4f}"


class FieldStatisticsWidget(QWidget):
    """Provides statistical summary for currently selected bits."""

    def __init__(self):
        super().__init__()
        self.frames_bits = []
        self.bit_order = "msb"
        self.scale_factor = 1.0
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(6)

        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Statistics Summary"))
        title_layout.addStretch()
        layout.addLayout(title_layout)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Consolas", 10))
        self.output.setPlaceholderText("Select column(s) to see statistics.")
        layout.addWidget(self.output, stretch=1)

    def set_frames_bits(self, frames_bits, bit_order="msb", scale_factor=1.0):
        self.frames_bits = frames_bits or []
        self.bit_order = bit_order
        self.scale_factor = scale_factor
        self._render_stats()

    def _apply_bit_order(self, bits):
        return apply_bit_order(bits, self.bit_order)

    def _bits_to_int(self, bits):
        return bits_to_int(bits)

    def _format_value(self, value):
        if value is None:
            return "(n/a)"

        if self.scale_factor == 1.0:
            return str(value)
        return f"{value * self.scale_factor:.4f}"

    def _render_stats(self):
        if not self.frames_bits:
            self.output.setPlainText("Select column(s) to see statistics.")
            return

        values = []
        bit_length = 0
        for bits in self.frames_bits:
            if bits is None or len(bits) == 0:
                continue
            ordered = self._apply_bit_order(bits)
            bit_length = max(bit_length, len(ordered))
            val = self._bits_to_int(ordered)
            if val is not None:
                values.append(val)

        if not values:
            self.output.setPlainText("No numeric data available for this selection.")
            return

        total = len(values)
        distinct = len(set(values))
        counts = Counter(values)
        min_val = min(values)
        max_val = max(values)

        lines = []
        lines.append(f"Samples        : {total}")
        lines.append(f"Distinct values: {distinct}")
        lines.append(f"Min / Max      : {self._format_value(min_val)} / {self._format_value(max_val)}")

        # Top N frequencies
        top_entries = counts.most_common(5)
        lines.append("Top values:")
        for value, count in top_entries:
            pct = (count / total) * 100
            display_val = self._format_value(value)
            lines.append(f"  {display_val:<12} {count:>5} ({pct:4.1f}%)")

        flags = self._classify_entropy(counts, total, bit_length, distinct)
        lines.append(f"Flags          : {', '.join(flags)}")

        self.output.setPlainText("\n".join(lines))

    def _classify_entropy(self, counts, total, bit_length, distinct):
        flags = []
        if distinct <= 1:
            flags.append("Constant")
            return flags

        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)

        if bit_length <= 0:
            bit_length = math.log2(distinct)

        normalized = entropy / max(1e-9, bit_length)

        if normalized < 0.15:
            flags.append("Very low entropy")
        elif normalized > 0.85:
            flags.append("High entropy")
        else:
            flags.append("Moderate entropy")

        if distinct <= 4:
            flags.append("Likely small enum/flags")
        if normalized > 0.9 and distinct > total * 0.5:
            flags.append("Likely payload/crypto")

        return flags


class TextDisplayWidget(QTextEdit):
    """Text-based display for Binary and Hex modes"""

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 11))
        self.bits = None
        self.width = 64
        self.display_mode = "binary"
        self.highlighted_positions = set()
        self.pattern_length = 0

        # OPTIMIZED: Pagination for large files
        self.max_display_rows = 1000  # Show max 1000 rows at a time
        self.current_page = 0
        self.highlight_intervals = []

    def set_bits(self, bits):
        self.bits = bits
        self.update_display()

    def set_highlights(self, positions, pattern_len):
        self.highlighted_positions = set(positions)
        self.pattern_length = pattern_len

        # OPTIMIZED: Build sorted intervals for fast lookup
        self._build_highlight_intervals()
        self.update_display()

    def _build_highlight_intervals(self):
        """Build sorted list of highlight intervals for O(log n) lookup"""
        self.highlight_intervals = build_highlight_intervals(
            self.highlighted_positions,
            self.pattern_length,
        )

    def _is_bit_highlighted(self, bit_index):
        """Fast O(log n) binary search to check if bit is highlighted"""
        return is_bit_highlighted(self.highlight_intervals, bit_index)

    def clear_highlights(self):
        self.highlighted_positions.clear()
        self.highlight_intervals = []
        self.update_display()

    def update_display(self):
        if self.bits is None:
            return

        self.clear()

        if self.display_mode == "binary":
            self.display_binary()
        elif self.display_mode == "hex":
            self.display_hex()

    def display_binary(self):
        """Display bits as binary (0s and 1s) - OPTIMIZED with pagination"""
        html_parts = []
        html_parts.append('<pre style="font-family: Consolas, Courier New, monospace; font-size: 11pt; margin: 0; padding: 8px;">')

        # OPTIMIZED: Calculate pagination
        total_rows = (len(self.bits) + self.width - 1) // self.width
        start_row = self.current_page * self.max_display_rows
        end_row = min(start_row + self.max_display_rows, total_rows)

        if total_rows > self.max_display_rows:
            html_parts.append(f'<div style="background: #ffffcc; padding: 5px;">Showing rows {start_row}-{end_row} of {total_rows} (Page {self.current_page + 1} of {(total_rows + self.max_display_rows - 1) // self.max_display_rows})</div>\n')

        start_bit = start_row * self.width
        end_bit = min(end_row * self.width, len(self.bits))

        for i in range(start_bit, end_bit, self.width):
            row_bits = self.bits[i:i + self.width]

            html_parts.append(f'<span style="color: #666;">{i:6d}: </span>')

            for j, bit in enumerate(row_bits):
                bit_position = i + j
                # OPTIMIZED: Use binary search instead of any()
                is_highlighted = self._is_bit_highlighted(bit_position)

                if is_highlighted:
                    color = '#ff0000' if bit == 1 else '#000000'
                    html_parts.append(
                        f'<span style="background-color: #ffff00; color: {color};">{bit}</span>'
                    )
                else:
                    html_parts.append(str(bit))

            html_parts.append('\n')

        html_parts.append('</pre>')
        self.setHtml(''.join(html_parts))

    def display_hex(self):
        """Display bits as hex (nibbles - 4 bits per character) - OPTIMIZED with pagination"""
        html_parts = []
        html_parts.append('<pre style="font-family: Consolas, Courier New, monospace; font-size: 11pt; margin: 0; padding: 8px;">')

        # OPTIMIZED: Calculate pagination
        total_rows = (len(self.bits) + self.width - 1) // self.width
        start_row = self.current_page * self.max_display_rows
        end_row = min(start_row + self.max_display_rows, total_rows)

        if total_rows > self.max_display_rows:
            html_parts.append(f'<div style="background: #ffffcc; padding: 5px;">Showing rows {start_row}-{end_row} of {total_rows} (Page {self.current_page + 1} of {(total_rows + self.max_display_rows - 1) // self.max_display_rows})</div>\n')

        start_bit = start_row * self.width
        end_bit = min(end_row * self.width, len(self.bits))

        for i in range(start_bit, end_bit, self.width):
            row_bits = self.bits[i:i + self.width]

            html_parts.append(f'<span style="color: #666;">{i:6d}: </span>')

            padded_bits = row_bits.copy()
            if len(padded_bits) % 4 != 0:
                padding_needed = 4 - (len(padded_bits) % 4)
                padded_bits = np.append(padded_bits, np.zeros(padding_needed, dtype=np.uint8))

            for nibble_idx in range(0, len(padded_bits), 4):
                nibble = padded_bits[nibble_idx:nibble_idx + 4]

                # OPTIMIZED: Use bit shifting instead of power operation
                hex_val = (nibble[0] << 3) | (nibble[1] << 2) | (nibble[2] << 1) | nibble[3]
                hex_char = f"{hex_val:X}"

                # OPTIMIZED: Check any bit in nibble with binary search
                is_highlighted = False
                for k in range(4):
                    bit_position = i + nibble_idx + k
                    if bit_position < i + len(row_bits):
                        if self._is_bit_highlighted(bit_position):
                            is_highlighted = True
                            break

                is_padded = (nibble_idx + 4) > len(row_bits)

                if is_highlighted:
                    html_parts.append(
                        f'<span style="background-color: #ffff00; color: #ff0000;">{hex_char}</span>'
                    )
                elif is_padded:
                    html_parts.append(
                        f'<span style="background-color: #ffc8c8; color: #808080;">{hex_char}</span>'
                    )
                else:
                    html_parts.append(hex_char)

            html_parts.append('\n')

        html_parts.append('</pre>')
        self.setHtml(''.join(html_parts))

