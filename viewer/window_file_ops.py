"""Display, file, and navigation helpers for the main bit viewer window."""

from __future__ import annotations

import copy

import numpy as np
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox


class BitViewerWindowFileOpsMixin:
    def _ask_mode_transition_question(self, title, message):
        previous_suspend_state = getattr(self.byte_table, "_suspend_focus_clear", False)
        self.byte_table._suspend_focus_clear = True
        try:
            return QMessageBox.question(
                self,
                title,
                message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
        finally:
            self.byte_table._suspend_focus_clear = previous_suspend_state

    def _sync_live_viewer_dock_visibility(self):
        if not hasattr(self, "live_bit_viewer_dock"):
            return
        if self.data_mode == "byte":
            self.live_bit_viewer_dock.show()
        else:
            self.live_bit_viewer_dock.hide()

    def _restore_byte_frames_from_bit_layout(self, frame_width_bits, total_frame_bits):
        if frame_width_bits <= 0:
            self._clear_active_frames()
            self._set_row_size_control(enabled=True)
            return
        restored_frames = []
        for start_bit in range(0, total_frame_bits, frame_width_bits):
            length_bits = min(frame_width_bits, total_frame_bits - start_bit)
            if length_bits > 0:
                restored_frames.append((start_bit, length_bits))
        self.byte_table.frames = restored_frames if restored_frames else None
        self.byte_table.frame_pattern = copy.deepcopy(self._saved_byte_frame_pattern)
        self.byte_table.frame_bit_offsets = None
        self._set_row_size_control(enabled=not bool(restored_frames))

    def _restore_byte_mode_from_saved_frames(self):
        bits_to_save = self.bits.copy()
        final_padding = (8 - (len(bits_to_save) % 8)) % 8
        if final_padding > 0:
            bits_to_save = np.append(bits_to_save, np.zeros(final_padding, dtype=np.uint8))

        self.bytes_data = np.packbits(bits_to_save)
        self.byte_table.bytes_data = self.bytes_data

        if self._restore_padded_byte_frames and self._saved_padded_frame_width_bits:
            bytes_per_row = self._saved_padded_frame_width_bits // 8
            self.byte_table.set_row_size(bytes_per_row)
            self._set_row_size_control(value=bytes_per_row)
            self._restore_byte_frames_from_bit_layout(self._saved_padded_frame_width_bits, len(bits_to_save))
        else:
            self._restore_saved_byte_frames()

    def _restore_saved_byte_frames(self):
        if not self._saved_byte_frames:
            self._clear_active_frames()
            self._set_row_size_control(enabled=True)
            return

        self.byte_table.frames = copy.deepcopy(self._saved_byte_frames)
        self.byte_table.frame_pattern = copy.deepcopy(self._saved_byte_frame_pattern)
        self.byte_table.frame_bit_offsets = None
        self._apply_frame_row_size_lock()

    def _refresh_byte_mode_view(self):
        if self.bytes_data is None:
            return
        self.scroll_area.hide()
        self.text_display.hide()
        self.byte_table.show()
        self.byte_table.bytes_data = self.bytes_data
        if hasattr(self.byte_table, "_invalidate_bit_cache"):
            self.byte_table._invalidate_bit_cache()
        self._refresh_byte_table_view()
        self._sync_live_viewer_dock_visibility()

    def on_mode_changed(self):
        previous_mode = self.data_mode
        new_mode = "bit" if self.bit_mode_radio.isChecked() else "byte"
        if new_mode == previous_mode:
            return

        if previous_mode == "bit" and new_mode == "byte":
            if self.bits is None:
                self.data_mode = new_mode
                self.update_left_panel_visibility()
                self.update_display()
                self._sync_live_viewer_dock_visibility()
                self.apply_theme()
                return

            if self._return_to_byte_framed:
                self._restore_byte_mode_from_saved_frames()
                if self.operations:
                    self.operations = []
                    self.operations_list.clear()
                    self.undo_stack.clear()
                    self.redo_stack.clear()
                    self.clear_highlights()
                self.data_mode = new_mode
                self.update_left_panel_visibility()
                self.update_display()
                self._sync_live_viewer_dock_visibility()
                self.apply_theme()
                return

            frame_width_bits = self.width_spin.value()
            total_bits = len(self.bits)
            num_complete_frames = total_bits // frame_width_bits
            remaining_bits = total_bits % frame_width_bits
            frame_padding_needed = (8 - (frame_width_bits % 8)) % 8

            use_frame_padding = False
            if frame_padding_needed > 0:
                padded_frame_width = frame_width_bits + frame_padding_needed
                reply = self._ask_mode_transition_question(
                    "Zero-Pad to Byte Boundary?",
                    f"Current frame width: {frame_width_bits:,} bits\n\n"
                    f"This is not divisible by 8.\n\n"
                    f"Zero-pad each frame by {frame_padding_needed} bit(s)?\n\n"
                    f"Yes = Pad frames to {padded_frame_width:,} bits and preserve frame alignment\n"
                    f"No = Keep all {total_bits:,} bits as-is (frame alignment will be lost in byte mode)",
                )
                use_frame_padding = reply == QMessageBox.StandardButton.Yes

            if use_frame_padding and frame_padding_needed > 0:
                padded_frame_width = frame_width_bits + frame_padding_needed
                result_bits = []
                for frame_index in range(num_complete_frames):
                    frame_start = frame_index * frame_width_bits
                    frame_end = frame_start + frame_width_bits
                    frame = self.bits[frame_start:frame_end]
                    result_bits.append(np.append(frame, np.zeros(frame_padding_needed, dtype=np.uint8)))
                if remaining_bits > 0:
                    last_frame = self.bits[num_complete_frames * frame_width_bits:]
                    last_padding = (8 - (len(last_frame) % 8)) % 8
                    if last_padding > 0:
                        last_frame = np.append(last_frame, np.zeros(last_padding, dtype=np.uint8))
                    result_bits.append(last_frame)
                bits_to_save = np.concatenate(result_bits) if result_bits else self.bits.copy()
            else:
                bits_to_save = self.bits.copy()
                final_padding = (8 - (len(bits_to_save) % 8)) % 8
                if final_padding > 0:
                    bits_to_save = np.append(bits_to_save, np.zeros(final_padding, dtype=np.uint8))

            self.bytes_data = np.packbits(bits_to_save)
            self.byte_table.bytes_data = self.bytes_data

            if use_frame_padding and frame_padding_needed > 0:
                bytes_per_row = (frame_width_bits + frame_padding_needed) // 8
            elif frame_padding_needed == 0:
                bytes_per_row = frame_width_bits // 8
            else:
                bytes_per_row = frame_width_bits // 8

            self.byte_table.set_row_size(bytes_per_row)
            self._set_row_size_control(value=bytes_per_row)

            if self._return_to_byte_framed and frame_width_bits > 0:
                if self._restore_padded_byte_frames:
                    self._restore_byte_frames_from_bit_layout(frame_width_bits, len(bits_to_save))
                else:
                    self._restore_saved_byte_frames()
            else:
                self._clear_active_frames()
                self._set_row_size_control(enabled=True)

            if self.operations:
                self.operations = []
                self.operations_list.clear()
                self.undo_stack.clear()
                self.redo_stack.clear()
                self.clear_highlights()

        elif previous_mode == "byte" and new_mode == "bit":
            self._reset_saved_frame_restore_state()
            if self.bytes_data is not None:
                if self.byte_table.frames is not None and len(self.byte_table.frames) > 0:
                    self._saved_byte_frames = copy.deepcopy(self.byte_table.frames)
                    self._saved_byte_frame_pattern = copy.deepcopy(self.byte_table.frame_pattern)
                    self._return_to_byte_framed = True
                    frame_count = len(self.byte_table.frames)
                    extracted_frames = []
                    for frame_index in range(frame_count):
                        frame_bytes = self.byte_table._get_frame_bytes(frame_index)
                        if len(frame_bytes) > 0:
                            extracted_frames.append(frame_bytes)

                    if not extracted_frames:
                        work_bytes = self.bytes_data
                    else:
                        frame_lengths = [len(frame_bytes) for frame_bytes in extracted_frames]
                        max_len = max(frame_lengths)
                        min_len = min(frame_lengths)
                        if min_len < max_len:
                            reply = self._ask_mode_transition_question(
                                "Zero-Pad Frames?",
                                f"You have {len(extracted_frames)} frames with varying lengths.\n\n"
                                f"Min length: {min_len:,} bytes\n"
                                f"Max length: {max_len:,} bytes\n\n"
                                f"Zero-pad all frames to max length ({max_len:,} bytes)?\n\n"
                                f"Yes = Pad frames and set width to {max_len * 8:,} bits\n"
                                f"No = Use original data without padding",
                            )
                            use_padding = reply == QMessageBox.StandardButton.Yes
                        else:
                            use_padding = True

                        if use_padding:
                            frame_width_bits = max_len * 8
                            if frame_width_bits > self.width_spin.maximum():
                                self.width_spin.setMaximum(frame_width_bits)
                            self.width_spin.blockSignals(True)
                            self.width_spin.setValue(frame_width_bits)
                            self.width_spin.blockSignals(False)
                            padded_frames = []
                            for frame_bytes in extracted_frames:
                                if len(frame_bytes) < max_len:
                                    padding = np.zeros(max_len - len(frame_bytes), dtype=np.uint8)
                                    padded_frames.append(np.concatenate([frame_bytes, padding]))
                                else:
                                    padded_frames.append(frame_bytes)
                            work_bytes = np.concatenate(padded_frames)
                            self._restore_padded_byte_frames = True
                            self._saved_padded_frame_width_bits = frame_width_bits
                        else:
                            work_bytes = self.bytes_data
                    self.bits = np.unpackbits(work_bytes)
                else:
                    self.bits = np.unpackbits(self.bytes_data)
                self.original_bits = self.bits

        self.data_mode = new_mode
        self.update_left_panel_visibility()
        self.update_display()
        self._sync_live_viewer_dock_visibility()
        self.apply_theme()

    def _ensure_bits_loaded(self):
        if self.bits is None and self.bytes_data is not None:
            self.status_label.setText("Unpacking bits from bytes...")
            QApplication.processEvents()
            self.bits = np.unpackbits(self.bytes_data)
            self.original_bits = self.bits
            self.status_label.setText("Bits unpacked")
            return True
        return self.bits is not None

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Binary File", "", "Binary files (*.bin);;All files (*.*)")
        if not filename:
            return
        try:
            with open(filename, "rb") as f:
                byte_data = np.frombuffer(f.read(), dtype=np.uint8)

            self.original_bytes = byte_data.copy()
            self.bytes_data = self.original_bytes
            self.original_bits = None
            self.bits = None
            self._reset_saved_frame_restore_state()
            self.filename = filename
            self.operations = []
            self.operations_list.clear()
            self.undo_stack = []
            self.redo_stack = []
            self.byte_table.clear_frames()
            self.clear_column_definitions()
            self._set_row_size_control(value=16, enabled=True)

            file_size = len(byte_data)
            bit_count = file_size * 8
            is_large_file = file_size >= 10_000_000

            if self.data_mode == "bit" and not is_large_file:
                self.bits = np.unpackbits(byte_data)
                self.original_bits = self.bits
                ones = np.sum(self.bits)
                ones_pct = ones / bit_count * 100
            elif self.data_mode == "bit" and is_large_file:
                ones = "N/A (calculating...)"
                ones_pct = "N/A"
            else:
                ones = "N/A"
                ones_pct = "N/A"

            self.file_info_label.setText(
                f"File: {filename.split('/')[-1]}\n"
                f"Size: {file_size} bytes ({bit_count} bits)\n"
                f"Ones: {ones} ({ones_pct}{'%' if ones_pct != 'N/A' else ''})"
            )
            self.update_display()
            self.status_label.setText(
                f"Loaded {file_size} bytes (large file - unpacking bits...)" if is_large_file and self.data_mode == "bit"
                else f"Loaded {file_size} bytes"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open file: {e}")

    def save_file(self):
        if self.data_mode == "bit" and self.bits is None:
            QMessageBox.warning(self, "Warning", "No data to save")
            return
        if self.data_mode == "byte" and self.bytes_data is None:
            QMessageBox.warning(self, "Warning", "No data to save")
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Save Processed Data", "", "Binary files (*.bin);;All files (*.*)")
        if not filename:
            return
        try:
            if self.data_mode == "bit":
                bits_to_save = self.bits.copy()
                padding = (8 - (len(bits_to_save) % 8)) % 8
                if padding:
                    bits_to_save = np.append(bits_to_save, np.zeros(padding, dtype=np.uint8))
                byte_data = np.packbits(bits_to_save)
            else:
                byte_data = self.bytes_data
            with open(filename, "wb") as f:
                f.write(byte_data.tobytes())
            QMessageBox.information(self, "Success", f"Saved {len(byte_data)} bytes")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file: {e}")

    def _queue_width_change(self, value):
        self._pending_width_value = value
        self._width_timer.start()

    def _apply_width_change(self):
        width = getattr(self, "_pending_width_value", self.width_spin.value())
        if self.data_mode != "bit" or self.bits is None:
            self.update_display()
            return
        mode = self.current_bit_display_mode()
        if mode in ["squares", "circles"]:
            self.canvas.set_bits_per_row(width)
        elif mode in ["binary", "hex"]:
            self.text_display.width = width
            self.text_display.update_display()

    def update_display(self):
        if self.data_mode == "bit":
            if self.bits is None:
                if self.bytes_data is not None:
                    self.status_label.setText("Unpacking bits from bytes (one-time operation)...")
                    QApplication.processEvents()
                    self.bits = np.unpackbits(self.bytes_data)
                    self.original_bits = self.bits
                    bit_count = len(self.bits)
                    if len(self.bytes_data) < 10_000_000:
                        ones = np.sum(self.bits)
                        ones_pct = ones / bit_count * 100
                    else:
                        ones = "N/A"
                        ones_pct = "N/A"
                    file_size = len(self.bytes_data)
                    self.file_info_label.setText(
                        f"File: {self.filename.split('/')[-1] if self.filename else 'N/A'}\n"
                        f"Size: {file_size} bytes ({bit_count} bits)\n"
                        f"Ones: {ones} ({ones_pct}{'%' if ones_pct != 'N/A' else ''})"
                    )
                    self.status_label.setText("Bits unpacked - ready to display")
                else:
                    self.scroll_area.hide()
                    self.text_display.hide()
                    self.byte_table.hide()
                    self.status_label.setText("No data loaded")
                    return

            mode = self.current_bit_display_mode()
            self.byte_table.hide()
            if mode in ["squares", "circles"]:
                self.scroll_area.show()
                self.text_display.hide()
                self.canvas.display_mode = mode
                self.canvas.bit_size = self.size_spin.value()
                self.canvas.set_bits(self.bits)
                self.canvas.set_bits_per_row(self.width_spin.value())
            elif mode in ["binary", "hex"]:
                self.scroll_area.hide()
                self.text_display.show()
                self.text_display.display_mode = mode
                self.text_display.width = self.width_spin.value()
                self.text_display.set_bits(self.bits)
        else:
            if self.bytes_data is None:
                return
            self._refresh_byte_mode_view()

    def reset_to_original(self):
        if self.data_mode == "bit" and self.original_bits is not None:
            self.save_state()
            self.bits = self.original_bits.copy()
            self.operations = []
            self.operations_list.clear()
            self.clear_highlights()
            self.update_display()
            self.status_label.setText("Reset")
        elif self.data_mode == "byte" and self.original_bytes is not None:
            self.save_state()
            self.bytes_data = self.original_bytes.copy()
            self.operations = []
            self.operations_list.clear()
            self.byte_table.clear_frames()
            self.update_display()
            self.status_label.setText("Reset")

    def scroll_up(self):
        if self.scroll_area.isVisible():
            scrollbar = self.scroll_area.verticalScrollBar()
            scrollbar.setValue(scrollbar.value() - self.canvas.bit_size)
        elif self.text_display.isVisible():
            scrollbar = self.text_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.value() - 100)

    def scroll_down(self):
        if self.scroll_area.isVisible():
            scrollbar = self.scroll_area.verticalScrollBar()
            scrollbar.setValue(scrollbar.value() + self.canvas.bit_size)
        elif self.text_display.isVisible():
            scrollbar = self.text_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.value() + 100)

    def go_to_start(self):
        if self.scroll_area.isVisible():
            self.scroll_area.verticalScrollBar().setValue(0)
        elif self.text_display.isVisible():
            self.text_display.verticalScrollBar().setValue(0)

    def go_to_end(self):
        if self.scroll_area.isVisible():
            self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())
        elif self.text_display.isVisible():
            self.text_display.verticalScrollBar().setValue(self.text_display.verticalScrollBar().maximum())
