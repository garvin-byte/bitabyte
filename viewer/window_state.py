"""Undo/redo and operation history helpers for the main bit viewer window."""

from __future__ import annotations

import copy
import re

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QListWidgetItem, QPushButton, QWidget


class BitViewerWindowStateMixin:
    def _max_frame_row_size(self):
        frames = self.byte_table.frames
        if not frames:
            return None
        first_frame = frames[0]
        if isinstance(first_frame, tuple) and len(first_frame) == 2:
            return max((length_bits + 7) // 8 for _, length_bits in frames)
        return max(len(frame) for frame in frames)

    def _set_row_size_control(self, value=None, enabled=None):
        if value is not None:
            self.row_size_spin.blockSignals(True)
            self.row_size_spin.setValue(value)
            self.row_size_spin.blockSignals(False)
        if enabled is not None:
            self.row_size_spin.setEnabled(enabled)

    def _apply_frame_row_size_lock(self):
        max_row_size = self._max_frame_row_size()
        if max_row_size:
            self.byte_table.set_row_size(max_row_size)
            self._set_row_size_control(value=max_row_size, enabled=False)
        else:
            self._set_row_size_control(enabled=True)
        return max_row_size

    def _clear_active_frames(self):
        self.byte_table.frames = None
        self.byte_table.frame_pattern = None
        self.byte_table.frame_bit_offsets = None

    def _reset_saved_frame_restore_state(self):
        self._return_to_byte_framed = False
        self._saved_byte_frame_pattern = None
        self._saved_byte_frames = None
        self._restore_padded_byte_frames = False
        self._saved_padded_frame_width_bits = None

    def _refresh_byte_table_view(self):
        self.byte_table.update_display()
        self.byte_table._update_live_bit_viewer()
        if hasattr(self, "refresh_column_definitions_list"):
            self.refresh_column_definitions_list()

    def save_state(self):
        bits_copy = self.bits.copy() if self.bits is not None else None
        bytes_copy = self.bytes_data.copy() if self.bytes_data is not None else None
        ops_copy = self.operations.copy()
        frame_pat = copy.deepcopy(self.byte_table.frame_pattern) if self.byte_table.frame_pattern is not None else None
        self.undo_stack.append((bits_copy, bytes_copy, ops_copy, frame_pat))
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            self.status_label.setText("Nothing to undo")
            return

        current_state = (
            self.bits.copy() if self.bits is not None else None,
            self.bytes_data.copy() if self.bytes_data is not None else None,
            self.operations.copy(),
            copy.deepcopy(self.byte_table.frame_pattern) if self.byte_table.frame_pattern is not None else None,
        )
        self.redo_stack.append(current_state)
        bits, bytes_data, operations, frame_pattern = self.undo_stack.pop()
        self.bits = bits.copy() if bits is not None else None
        self.bytes_data = bytes_data.copy() if bytes_data is not None else None
        self.operations = operations.copy()

        if frame_pattern is not None:
            self.byte_table.set_frame_pattern(frame_pattern)
            self._apply_frame_row_size_lock()
        else:
            self.byte_table.clear_frames()
            self._set_row_size_control(enabled=True)

        self.rebuild_operations_list()
        self.update_display()
        self.status_label.setText("Undone")

    def redo(self):
        if not self.redo_stack:
            self.status_label.setText("Nothing to redo")
            return

        current_state = (
            self.bits.copy() if self.bits is not None else None,
            self.bytes_data.copy() if self.bytes_data is not None else None,
            self.operations.copy(),
            copy.deepcopy(self.byte_table.frame_pattern) if self.byte_table.frame_pattern is not None else None,
        )
        self.undo_stack.append(current_state)
        bits, bytes_data, operations, frame_pattern = self.redo_stack.pop()
        self.bits = bits.copy() if bits is not None else None
        self.bytes_data = bytes_data.copy() if bytes_data is not None else None
        self.operations = operations.copy()

        if frame_pattern is not None:
            self.byte_table.set_frame_pattern(frame_pattern)
            self._apply_frame_row_size_lock()
        else:
            self.byte_table.clear_frames()
            self._set_row_size_control(enabled=True)

        self.rebuild_operations_list()
        self.update_display()
        self.status_label.setText("Redone")

    def rebuild_operations_list(self):
        self.operations_list.clear()
        for op_text in self.operations:
            self.add_operation_to_list(op_text)

    def add_operation_to_list(self, operation_text):
        item_widget = QWidget()
        item_layout = QHBoxLayout(item_widget)
        item_layout.setContentsMargins(5, 2, 5, 2)

        label = QLabel(operation_text)
        label.setStyleSheet("QLabel { background: transparent; }")
        item_layout.addWidget(label, 1)

        delete_btn = QPushButton("x")
        delete_btn.setFixedSize(20, 20)
        delete_btn.setStyleSheet(
            "QPushButton { background-color: transparent; color: #888; border: none; font-weight: bold; font-size: 14px; }"
            "QPushButton:hover { background-color: #ffcccc; color: #cc0000; border-radius: 10px; }"
        )

        current_index = self.operations_list.count()
        delete_btn.clicked.connect(lambda: self.delete_operation(current_index))
        item_layout.addWidget(delete_btn)

        item = QListWidgetItem(self.operations_list)
        item.setSizeHint(item_widget.sizeHint())
        self.operations_list.addItem(item)
        self.operations_list.setItemWidget(item, item_widget)

    def delete_operation(self, index):
        if index < 0 or index >= len(self.operations):
            return
        self.save_state()
        self.operations.pop(index)
        self.rebuild_from_operations()
        self.status_label.setText(f"Deleted operation {index + 1}")

    def rebuild_from_operations(self):
        if self.data_mode == "bit":
            if self.original_bits is None:
                return
            self.bits = self.original_bits.copy()
        else:
            if self.original_bytes is None:
                return
            self.bytes_data = self.original_bytes.copy()
            self.byte_table.clear_frames()

        self.operations_list.clear()

        for op_text in self.operations:
            if self.data_mode == "bit":
                if op_text.startswith("Delta"):
                    match = re.search(r"Delta (\d+)", op_text)
                    if match:
                        self.apply_delta_internal(int(match.group(1)))
                elif op_text.startswith("Take/Skip"):
                    match = re.search(r"Take/Skip: (.+)", op_text)
                    if match:
                        self.apply_takeskip_internal(match.group(1))
                elif op_text.startswith("XOR"):
                    match = re.search(r"XOR: (.+)", op_text)
                    if match:
                        self.apply_xor_internal(match.group(1))
                elif op_text == "Invert All":
                    self.apply_invert_internal()
            else:
                if op_text.startswith("Frame Sync"):
                    match = re.match(r"Frame Sync(?: \(([^)]+)\))?:\s*(.+)", op_text)
                    pattern_str = match.group(2).strip() if match else op_text.split("Frame Sync:", 1)[1].strip()
                    prefer_byte = bool(match and match.group(1) and "byte" in match.group(1).lower())
                    pattern = self.parse_hex_pattern(pattern_str)
                    if pattern is not None:
                        self.byte_table.set_frame_pattern(pattern, force_byte_mode=prefer_byte)
                        self._apply_frame_row_size_lock()
                elif op_text == "Clear Frames":
                    self.byte_table.clear_frames()
                    self._set_row_size_control(enabled=True)

            self.add_operation_to_list(op_text)

        self.update_display()
