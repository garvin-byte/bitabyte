"""Frame-sync helpers for the main bit viewer window."""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from .colors import BIT_SYNC_HARD_LIMIT_BYTES, BIT_SYNC_WARNING_BYTES, _populate_color_combo
from .column import ColumnDefinition


class BitViewerWindowFramesMixin:
    def frame_sync(self, prefill_pattern=None):
        if self.bytes_data is None:
            QMessageBox.information(self, "Frame Sync", "No byte data loaded.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Frame Sync Pattern")
        dialog.setMinimumWidth(450)
        layout = QVBoxLayout(dialog)

        input_format_group = QWidget()
        input_format_layout = QVBoxLayout(input_format_group)
        input_hex_radio = QRadioButton("Hex (0x414B or 414B)")
        input_binary_radio = QRadioButton("Binary (1/0 or X/O)")
        input_hex_radio.setChecked(True)
        input_format_layout.addWidget(input_hex_radio)
        input_format_layout.addWidget(input_binary_radio)
        layout.addWidget(input_format_group)

        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("Sync Pattern:"))
        pattern_input = QLineEdit()
        pattern_input.setPlaceholderText("e.g. 0x414B or 414B")
        if prefill_pattern:
            pattern_input.setText(prefill_pattern)
            if prefill_pattern.startswith(("0x", "0X")):
                input_hex_radio.setChecked(True)
        pattern_layout.addWidget(pattern_input)
        pattern_widget = QWidget()
        pattern_widget.setLayout(pattern_layout)
        layout.addWidget(pattern_widget)

        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("Column Label:"))
        label_input = QLineEdit("Sync")
        label_layout.addWidget(label_input)
        label_widget = QWidget()
        label_widget.setLayout(label_layout)
        layout.addWidget(label_widget)

        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Highlight Color:"))
        color_combo = QComboBox()
        _populate_color_combo(color_combo, "Sky")
        color_layout.addWidget(color_combo)
        color_widget = QWidget()
        color_widget.setLayout(color_layout)
        layout.addWidget(color_widget)

        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        display_hex_radio = QRadioButton("Hex")
        display_binary_radio = QRadioButton("Binary")
        display_ascii_radio = QRadioButton("ASCII")
        display_hex_radio.setChecked(True)
        display_layout.addWidget(display_hex_radio)
        display_layout.addWidget(display_binary_radio)
        display_layout.addWidget(display_ascii_radio)
        layout.addWidget(display_widget)

        mode_widget = QWidget()
        mode_layout = QVBoxLayout(mode_widget)
        self.mode_bit_radio = QRadioButton("Bit-accurate (finds offset sync; slower)")
        self.mode_byte_radio = QRadioButton("Byte-aligned only (fast; requires byte pattern)")
        self.mode_bit_radio.setChecked(True)
        mode_layout.addWidget(self.mode_bit_radio)
        mode_layout.addWidget(self.mode_byte_radio)
        layout.addWidget(mode_widget)

        info_label = QLabel("")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        result_label = QLabel("")
        result_label.setWordWrap(True)
        result_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; }")
        layout.addWidget(result_label)

        button_box = QHBoxLayout()
        find_button = QPushButton("Find")
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_box.addWidget(find_button)
        button_box.addStretch()
        button_box.addWidget(ok_button)
        button_box.addWidget(cancel_button)
        layout.addLayout(button_box)

        warning_state = {"soft": False, "hard": False, "invalid": False, "bit_limit": False}

        def determine_search_mode(bit_sequence, show_dialog=True):
            byte_aligned = len(bit_sequence) % 8 == 0
            file_size = len(self.bytes_data) if self.bytes_data is not None else 0
            use_byte_mode = self.mode_byte_radio.isChecked()

            if use_byte_mode and not byte_aligned:
                if show_dialog and not warning_state["invalid"]:
                    QMessageBox.information(
                        dialog,
                        "Byte-aligned Frame Sync",
                        "Byte-aligned frame sync requires patterns with a length divisible by 8 bits.\n"
                        "Switching back to bit-accurate search for this pattern.",
                    )
                    warning_state["invalid"] = True
                self.mode_bit_radio.setChecked(True)
                use_byte_mode = False

            if byte_aligned and file_size >= BIT_SYNC_HARD_LIMIT_BYTES and not use_byte_mode:
                if show_dialog and not warning_state["hard"]:
                    QMessageBox.information(
                        dialog,
                        "Byte-aligned Required",
                        "Files that are 1 MB or larger require byte-aligned frame sync when the pattern is byte-aligned.\n"
                        "Switching to byte-aligned mode for performance.",
                    )
                    warning_state["hard"] = True
                self.mode_byte_radio.setChecked(True)
                use_byte_mode = True
            elif byte_aligned and file_size >= BIT_SYNC_WARNING_BYTES and not use_byte_mode:
                if show_dialog and not warning_state["soft"]:
                    QMessageBox.information(
                        dialog,
                        "Consider Byte-aligned Mode",
                        "Files over 256 KB run much faster with byte-aligned frame sync when your pattern is byte-aligned.",
                    )
                    warning_state["soft"] = True
            elif not byte_aligned and file_size >= BIT_SYNC_HARD_LIMIT_BYTES and not warning_state["bit_limit"]:
                if show_dialog:
                    reply = QMessageBox.warning(
                        dialog,
                        "Bit-accurate Frame Sync",
                        "This pattern is not byte-aligned, so the search must operate bit-by-bit.\n\n"
                        "Searching more than 1 MB of data in bit mode can be slow. Continue?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    )
                    warning_state["bit_limit"] = True
                    if reply == QMessageBox.StandardButton.No:
                        return use_byte_mode, False

            return use_byte_mode, True

        def parse_bits_from_dialog():
            pattern_text = pattern_input.text().strip()
            if not pattern_text:
                return None
            if input_binary_radio.isChecked():
                bits_str = pattern_text.replace(" ", "").upper().replace("X", "1").replace("O", "0")
                if bits_str and all(c in "10" for c in bits_str):
                    return [int(bit) for bit in bits_str]
                return None

            hex_str = pattern_text.replace(" ", "").upper()
            if hex_str.startswith("0X"):
                hex_str = hex_str[2:]
            if not hex_str or len(hex_str) % 2 != 0 or not all(c in "0123456789ABCDEF" for c in hex_str):
                return None
            try:
                return np.unpackbits(np.frombuffer(bytes.fromhex(hex_str), dtype=np.uint8)).tolist()
            except ValueError:
                return None

        def do_find():
            bits = parse_bits_from_dialog()
            if not bits:
                result_label.setText("Invalid pattern")
                return
            use_byte_mode, allowed = determine_search_mode(bits)
            if not allowed:
                result_label.setText("Search cancelled")
                return
            positions, max_len = self.byte_table.find_pattern_positions(bits, byte_aligned=use_byte_mode)
            if positions:
                mode_text = "byte-aligned" if use_byte_mode else "bit-accurate"
                result_label.setText(f"Found {len(positions):,} frames ({mode_text}, max length: {max_len:,} bytes)")
                self.byte_table.set_pattern_highlights(positions, len(bits))
            else:
                result_label.setText("Pattern not found")
                self.byte_table.clear_pattern_highlights()

        def update_display_options():
            bits = parse_bits_from_dialog()
            if bits is None:
                info_label.setText("Invalid pattern" if pattern_input.text().strip() else "")
                display_hex_radio.setEnabled(False)
                display_ascii_radio.setEnabled(False)
                return
            bit_count = len(bits)
            byte_count = bit_count // 8
            remainder = bit_count % 8
            divisible_by_4 = bit_count % 4 == 0
            divisible_by_8 = bit_count % 8 == 0
            display_binary_radio.setEnabled(True)
            display_hex_radio.setEnabled(divisible_by_4)
            display_ascii_radio.setEnabled(divisible_by_8)
            if not divisible_by_4 and display_hex_radio.isChecked():
                display_binary_radio.setChecked(True)
            if not divisible_by_8 and display_ascii_radio.isChecked():
                display_binary_radio.setChecked(True)
            info_text = f"Pattern length: {bit_count} bits"
            if divisible_by_8:
                info_text += f" ({byte_count} bytes)"
            elif remainder > 0:
                info_text += f" ({byte_count} bytes + {remainder} bits)"
            info_label.setText(info_text)

        find_button.clicked.connect(do_find)
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        pattern_input.textChanged.connect(update_display_options)
        input_binary_radio.toggled.connect(update_display_options)

        dialog_result = dialog.exec()
        self.byte_table.clear_pattern_highlights()
        if dialog_result != QDialog.DialogCode.Accepted:
            return

        bits = parse_bits_from_dialog()
        if not bits:
            QMessageBox.warning(self, "Frame Sync", "Invalid pattern.")
            return

        self.save_state()
        use_byte_mode, allowed = determine_search_mode(bits)
        if not allowed:
            return

        frame_count = self.byte_table.set_frame_pattern(None, pattern_bits=bits, force_byte_mode=use_byte_mode)
        if frame_count == 0:
            QMessageBox.information(self, "Frame Sync", "Pattern not found.")
            return

        frames = self.byte_table.frames
        if frames:
            self._apply_frame_row_size_lock()

        label_text = label_input.text().strip() or "Sync"
        color_name = color_combo.currentText()
        if color_name == "None":
            color_name = None
        bit_count = len(bits)
        if display_ascii_radio.isChecked() and bit_count % 8 == 0:
            display_fmt = "ascii_be"
        elif display_hex_radio.isChecked() and bit_count % 4 == 0:
            display_fmt = "hex_be"
        else:
            display_fmt = "binary"

        col_def = ColumnDefinition(0, 0, label_text, display_fmt, color_name, "bit", 0, bit_count)
        self.byte_table.add_column_definition(col_def)
        self.refresh_column_definitions_list()

        pattern_text = pattern_input.text().strip()
        mode_label = "Byte-aligned" if use_byte_mode else "Bit-accurate"
        op_text = f"Frame Sync ({mode_label}): {pattern_text}"
        self.operations.append(op_text)
        self.add_operation_to_list(op_text)
        if hasattr(self, "live_bit_viewer_dock"):
            self.live_bit_viewer_dock.show()

        mode_text = "byte-aligned" if use_byte_mode else "bit-accurate"
        if bit_count % 8 != 0:
            padding_bits = 8 - (bit_count % 8)
            self.status_label.setText(
                f"Framed into {frame_count} frames ({mode_text}, zero-padded with {padding_bits} bits per frame)"
            )
        else:
            self.status_label.setText(f"Framed into {frame_count} frames ({mode_text})")

        if hasattr(self, "const_on_radio") and self.const_on_radio.isChecked():
            self.highlight_constant_columns()

    def clear_frame_sync(self):
        if self.bytes_data is None:
            return
        self.save_state()
        self.byte_table.clear_frames()
        self.clear_column_definitions()
        self._set_row_size_control(value=16, enabled=True)
        if hasattr(self, "live_bit_viewer_dock"):
            self._sync_live_viewer_dock_visibility()
        if hasattr(self.byte_table, "_clear_live_bit_viewer"):
            self.byte_table._clear_live_bit_viewer()
        if hasattr(self, "const_on_radio") and self.const_on_radio.isChecked():
            self.clear_constant_highlights()
            self.const_off_radio.setChecked(True)
        self.operations.append("Clear Frames")
        self.add_operation_to_list("Clear Frames")
        self.status_label.setText("Frames cleared, column definitions cleared, row size reset to 16")

    def highlight_constant_columns(self):
        if self.bytes_data is None:
            return
        self.byte_table.highlight_constant_columns()
        self.status_label.setText("Constant columns highlighted")

    def clear_constant_highlights(self):
        self.byte_table.clear_constant_highlights()
        self.status_label.setText("Constant highlights cleared")

    def on_constant_highlight_changed(self):
        if self.const_on_radio.isChecked():
            self.highlight_constant_columns()
        else:
            self.clear_constant_highlights()
