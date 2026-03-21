"""Bit-mode search and operation helpers for the main bit viewer window."""

from __future__ import annotations

import re

import numpy as np
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from common.bit_utils import bits_from_binary_string, bits_from_hex_string


class BitViewerWindowBitOpsMixin:
    def apply_takeskip_internal(self, pattern):
        matches = re.findall(r"([tsir])(\d+)", pattern.lower())
        pattern_cycle_len = sum(int(num) for _, num in matches)
        num_cycles = len(self.bits) // pattern_cycle_len if pattern_cycle_len > 0 else 0

        if num_cycles > 1000 and all(op in ("t", "s") for op, _ in matches):
            take_ranges = []
            pos = 0
            for op, num_str in matches:
                num = int(num_str)
                if op == "t":
                    take_ranges.append((pos, pos + num))
                pos += num

            full_bits = len(self.bits)
            num_full_cycles = full_bits // pattern_cycle_len
            if len(take_ranges) == 1 and take_ranges[0][0] == 0:
                take_len = take_ranges[0][1]
                data_len = num_full_cycles * pattern_cycle_len
                reshaped = self.bits[:data_len].reshape(num_full_cycles, pattern_cycle_len)
                result = reshaped[:, :take_len].copy().ravel()
                remainder = full_bits - data_len
                if remainder > 0:
                    take_remainder = min(take_len, remainder)
                    if take_remainder > 0:
                        result = np.append(result, self.bits[data_len:data_len + take_remainder])
                self.bits = result
                return

            result_parts = []
            chunk_cycles = 5_000_000
            for cycle_start in range(0, num_full_cycles, chunk_cycles):
                cycle_end = min(cycle_start + chunk_cycles, num_full_cycles)
                n_cycles = cycle_end - cycle_start
                start_bit = cycle_start * pattern_cycle_len
                end_bit = cycle_end * pattern_cycle_len
                chunk = self.bits[start_bit:end_bit].reshape(n_cycles, pattern_cycle_len)
                for t_start, t_end in take_ranges:
                    result_parts.append(chunk[:, t_start:t_end].ravel())

            result = np.concatenate(result_parts)
            remainder = full_bits % pattern_cycle_len
            if remainder > 0:
                rem_start = num_full_cycles * pattern_cycle_len
                for t_start, t_end in take_ranges:
                    if t_start < remainder:
                        actual_end = min(t_end, remainder)
                        result = np.append(result, self.bits[rem_start + t_start:rem_start + actual_end])
            self.bits = result
            return

        operations = []
        pos = 0
        total_output = 0
        while pos < len(self.bits):
            for op, num_str in matches:
                num = int(num_str)
                if pos >= len(self.bits):
                    break
                end = min(pos + num, len(self.bits))
                if op in ("t", "r", "i"):
                    operations.append((op, pos, end))
                    total_output += end - pos
                pos = end
            if pos >= len(self.bits):
                break

        result = np.empty(total_output, dtype=np.uint8)
        result_idx = 0
        for op, start, end in operations:
            chunk_len = end - start
            if op == "t":
                result[result_idx:result_idx + chunk_len] = self.bits[start:end]
            elif op == "r":
                result[result_idx:result_idx + chunk_len] = self.bits[start:end][::-1]
            elif op == "i":
                result[result_idx:result_idx + chunk_len] = 1 - self.bits[start:end]
            result_idx += chunk_len
        self.bits = result

    def apply_delta_internal(self, window):
        if window == 1:
            self.bits = np.bitwise_xor(self.bits[1:], self.bits[:-1])
            return

        total_bits = len(self.bits)
        num_complete_windows = (total_bits - window) // window
        if num_complete_windows > 0:
            end_idx = window + num_complete_windows * window
            current = self.bits[window:end_idx].reshape(num_complete_windows, window)
            previous = self.bits[0:end_idx - window].reshape(num_complete_windows, window)
            result = np.bitwise_xor(current, previous).flatten()
            remainder_start = end_idx
            if remainder_start < total_bits:
                remainder_len = total_bits - remainder_start
                current_remainder = self.bits[remainder_start:total_bits]
                prev_remainder = self.bits[remainder_start - window:remainder_start - window + remainder_len]
                result = np.concatenate([result, np.bitwise_xor(current_remainder, prev_remainder)])
        else:
            result = np.empty(0, dtype=np.uint8)
        self.bits = result

    def apply_xor_internal(self, pattern):
        pattern_bits = self.hex_to_bits(pattern) if pattern.startswith(("0x", "0X")) else self.binary_to_bits(pattern)
        if pattern_bits is None:
            return
        pattern_len = len(pattern_bits)
        if pattern_len == 1:
            self.bits = np.bitwise_xor(self.bits, pattern_bits[0])
            return
        num_repeats = (len(self.bits) + pattern_len - 1) // pattern_len
        extended_pattern = np.tile(pattern_bits, num_repeats)[:len(self.bits)]
        self.bits = np.bitwise_xor(self.bits, extended_pattern)

    def apply_invert_internal(self):
        self.bits = 1 - self.bits

    def hex_to_bits(self, hex_string):
        return bits_from_hex_string(hex_string)

    def binary_to_bits(self, binary_string):
        return bits_from_binary_string(binary_string, ignored_characters={" ", "\t", "\n", "\r"})

    def find_pattern_positions(self, pattern_bits):
        if self.bits is None or pattern_bits is None:
            return []
        pattern_len = len(pattern_bits)
        error_percent = self.error_tolerance_spin.value()
        max_errors = int((error_percent / 100.0) * pattern_len)
        from numpy.lib.stride_tricks import sliding_window_view

        if len(self.bits) < pattern_len:
            return []
        windows = sliding_window_view(self.bits, pattern_len)
        if max_errors == 0:
            return np.where(np.all(windows == pattern_bits, axis=1))[0].tolist()
        return np.where(np.sum(windows != pattern_bits, axis=1) <= max_errors)[0].tolist()

    def highlight_pattern(self):
        if not self._ensure_bits_loaded():
            self.pattern_results_label.setText("No bit data")
            return
        pattern_str = self.pattern_input.text().strip()
        if not pattern_str:
            return
        pattern_bits = self.hex_to_bits(pattern_str) if pattern_str.startswith(("0x", "0X")) else self.binary_to_bits(pattern_str)
        if pattern_bits is None:
            self.pattern_results_label.setText("Invalid pattern")
            return
        positions = self.find_pattern_positions(pattern_bits)
        if positions:
            error_percent = self.error_tolerance_spin.value()
            self.pattern_results_label.setText(
                f"Found {len(positions)} (+/-{error_percent}%)" if error_percent > 0 else f"Found {len(positions)}"
            )
            self.canvas.set_highlights(positions, len(pattern_bits))
            self.text_display.set_highlights(positions, len(pattern_bits))
        else:
            self.pattern_results_label.setText("Not found")

    def clear_highlights(self):
        self.canvas.clear_highlights()
        self.text_display.clear_highlights()
        self.pattern_results_label.setText("")

    def apply_takeskip(self):
        if not self._ensure_bits_loaded():
            QMessageBox.warning(self, "Warning", "No bit data available")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Take/Skip Pattern")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)

        explanation = QLabel("Enter pattern (e.g., t4r3i8s1):\nt=take, r=reverse, i=invert, s=skip")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("Pattern:"))
        pattern_input = QLineEdit()
        pattern_layout.addWidget(pattern_input)
        layout.addLayout(pattern_layout)

        preview_label = QLabel("")
        preview_label.setStyleSheet("QLabel { color: blue; font-weight: bold; }")
        layout.addWidget(preview_label)

        def preview_pattern():
            pattern = pattern_input.text().strip()
            if not pattern:
                preview_label.setText("Enter a pattern first")
                return
            try:
                matches = re.findall(r"([tsir])(\d+)", pattern.lower())
                if not matches:
                    preview_label.setText("Invalid pattern format")
                    return
                pattern_cycle_len = sum(int(num) for _, num in matches)
                if pattern_cycle_len == 0:
                    preview_label.setText("Pattern has zero length")
                    return
                bits_per_cycle = sum(int(num) for op, num in matches if op in ("t", "r", "i"))
                num_cycles = len(self.bits) // pattern_cycle_len
                output_bits = num_cycles * bits_per_cycle
                remainder = len(self.bits) % pattern_cycle_len
                rem_pos = 0
                for op, num_str in matches:
                    num = int(num_str)
                    if rem_pos >= remainder:
                        break
                    if op in ("t", "r", "i"):
                        output_bits += min(num, remainder - rem_pos)
                    rem_pos += num
                input_bits = len(self.bits)
                reduction_pct = (1 - output_bits / input_bits) * 100 if input_bits > 0 else 0
                preview_label.setText(
                    f"Preview: {input_bits:,} bits -> {output_bits:,} bits\n"
                    f"({reduction_pct:.1f}% reduction, {num_cycles:,} cycles)"
                )
            except Exception as e:
                preview_label.setText(f"Error: {e}")

        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(preview_pattern)
        layout.addWidget(preview_btn)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            pattern = pattern_input.text().strip()
            if pattern:
                try:
                    self.save_state()
                    self.apply_takeskip_internal(pattern)
                    self.operations.append(f"Take/Skip: {pattern}")
                    self.add_operation_to_list(f"Take/Skip: {pattern}")
                    self.update_display()
                    self.status_label.setText(f"Applied {pattern}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", str(e))

    def apply_delta(self):
        if not self._ensure_bits_loaded():
            QMessageBox.warning(self, "Warning", "No bit data available")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Delta Encoding")
        layout = QVBoxLayout(dialog)
        explanation = QLabel("Delta encoding XORs each window with the previous window.")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("Window size:"))
        window_spin = QSpinBox()
        window_spin.setRange(1, 32)
        window_spin.setValue(1)
        window_layout.addWidget(window_spin)
        layout.addLayout(window_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                self.save_state()
                window = window_spin.value()
                self.apply_delta_internal(window)
                self.operations.append(f"Delta {window}")
                self.add_operation_to_list(f"Delta {window}")
                self.update_display()
                self.status_label.setText(f"Applied Delta {window}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def apply_xor(self):
        if not self._ensure_bits_loaded():
            QMessageBox.warning(self, "Warning", "No bit data available")
            return
        pattern, ok = QInputDialog.getText(self, "XOR Pattern", "Enter pattern (hex or binary):")
        if ok and pattern:
            self.save_state()
            self.apply_xor_internal(pattern)
            self.operations.append(f"XOR: {pattern}")
            self.add_operation_to_list(f"XOR: {pattern}")
            self.update_display()

    def apply_invert(self):
        if not self._ensure_bits_loaded():
            QMessageBox.warning(self, "Warning", "No bit data available")
            return
        self.save_state()
        self.apply_invert_internal()
        self.operations.append("Invert All")
        self.add_operation_to_list("Invert All")
        self.update_display()
