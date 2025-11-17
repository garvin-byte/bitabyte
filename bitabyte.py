#!/usr/bin/env python3
"""
PyQt6 Bit Viewer - Fast rendering with all your operations
Enhanced byte mode with flexible bit/byte column definitions and combined headers
"""

import sys
import numpy as np
import re
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSpinBox,
                             QFileDialog, QScrollArea, QMessageBox, QComboBox,
                             QLineEdit, QListWidget, QGroupBox,
                             QSplitter, QTextEdit, QDialog, QDialogButtonBox,
                             QInputDialog, QListWidgetItem, QRadioButton, QButtonGroup,
                             QTableWidget, QTableWidgetItem)
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPalette, QFont


class BitCanvas(QWidget):
    """Fast bit rendering widget using QPainter"""

    def __init__(self):
        super().__init__()
        self.bits = None

        # How many bits to draw per row (what used to be self.width)
        self.bits_per_row = 64
        self.bit_size = 10

        self.start_position = 0
        self.highlighted_positions = set()
        self.pattern_length = 0
        self.display_mode = "squares"  # "squares" or "circles"

        # Set background
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(255, 255, 255))
        self.setPalette(palette)

    def set_bits(self, bits):
        self.bits = bits
        self.update_size()
        self.update()

    def update_size(self):
        """Update the widget's minimum size based on bits_per_row and bit_size."""
        if self.bits is None:
            return

        cols = max(1, self.bits_per_row)
        rows = (len(self.bits) + cols - 1) // cols
        height = rows * self.bit_size
        width = cols * self.bit_size + 60  # Extra space for row labels
        self.setMinimumSize(width, height)

    def set_highlights(self, positions, pattern_len):
        self.highlighted_positions = set(positions)
        self.pattern_length = pattern_len

        # OPTIMIZED: Build sorted intervals for fast lookup
        self._build_highlight_intervals()
        self.update()

    def _build_highlight_intervals(self):
        """Build sorted list of highlight intervals for O(log n) lookup"""
        if not self.highlighted_positions or self.pattern_length == 0:
            self.highlight_intervals = []
            return

        # Create list of (start, end) intervals
        intervals = [(pos, pos + self.pattern_length) for pos in self.highlighted_positions]
        # Sort by start position
        self.highlight_intervals = sorted(intervals)

    def _is_bit_highlighted(self, bit_index):
        """Fast O(log n) binary search to check if bit is highlighted"""
        if not hasattr(self, 'highlight_intervals') or not self.highlight_intervals:
            return False

        # Binary search for the interval that might contain bit_index
        left, right = 0, len(self.highlight_intervals) - 1

        while left <= right:
            mid = (left + right) // 2
            start, end = self.highlight_intervals[mid]

            if start <= bit_index < end:
                return True
            elif bit_index < start:
                right = mid - 1
            else:
                left = mid + 1

        return False

    def clear_highlights(self):
        self.highlighted_positions.clear()
        self.pattern_length = 0
        self.highlight_intervals = []
        self.update()

    def paintEvent(self, event):
        if self.bits is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cols = max(1, self.bits_per_row)
        rows = (len(self.bits) + cols - 1) // cols

        # OPTIMIZED: Cull both vertically AND horizontally to visible region only
        visible_rect = event.rect()
        start_row = max(0, visible_rect.top() // self.bit_size)
        end_row = min(rows, (visible_rect.bottom() // self.bit_size) + 1)

        # Horizontal culling - only draw columns that are visible
        label_width = 55
        start_col = max(0, (visible_rect.left() - label_width) // self.bit_size)
        end_col = min(cols, (visible_rect.right() - label_width) // self.bit_size + 1)

        for row in range(start_row, end_row):
            # Row label: starting bit index for this row
            bit_pos = row * cols
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(
                QRect(0, row * self.bit_size, 50, self.bit_size),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                str(bit_pos),
            )

            # OPTIMIZED: Only iterate through visible columns, not all columns
            for col in range(start_col, end_col):
                bit_index = row * cols + col
                if bit_index >= len(self.bits):
                    break

                bit = self.bits[bit_index]

                # OPTIMIZED: Use binary search instead of O(n) any() loop
                is_highlighted = self._is_bit_highlighted(bit_index)

                if is_highlighted:
                    if bit == 1:
                        fill_color = QColor(255, 0, 0)
                        outline_color = QColor(139, 0, 0)
                    else:
                        fill_color = QColor(255, 255, 0)
                        outline_color = QColor(255, 165, 0)
                else:
                    if bit == 1:
                        fill_color = QColor(0, 0, 0)
                        outline_color = QColor(100, 100, 100)
                    else:
                        fill_color = QColor(255, 255, 255)
                        outline_color = QColor(100, 100, 100)

                x = col * self.bit_size + 55
                y = row * self.bit_size

                painter.setPen(QPen(outline_color, 1))
                painter.setBrush(QBrush(fill_color))

                if self.display_mode == "circles":
                    painter.drawEllipse(x, y, self.bit_size - 1, self.bit_size - 1)
                else:
                    painter.drawRect(x, y, self.bit_size - 1, self.bit_size - 1)



class TextDisplayWidget(QTextEdit):
    """Text-based display for Binary and Hex modes"""

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setFont(QFont("Courier", 10))
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
        if not self.highlighted_positions or self.pattern_length == 0:
            self.highlight_intervals = []
            return
        intervals = [(pos, pos + self.pattern_length) for pos in self.highlighted_positions]
        self.highlight_intervals = sorted(intervals)

    def _is_bit_highlighted(self, bit_index):
        """Fast O(log n) binary search to check if bit is highlighted"""
        if not self.highlight_intervals:
            return False
        left, right = 0, len(self.highlight_intervals) - 1
        while left <= right:
            mid = (left + right) // 2
            start, end = self.highlight_intervals[mid]
            if start <= bit_index < end:
                return True
            elif bit_index < start:
                right = mid - 1
            else:
                left = mid + 1
        return False

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
        html_parts.append('<pre style="font-family: Courier; font-size: 10pt; margin: 0; padding: 5px;">')

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
        html_parts.append('<pre style="font-family: Courier; font-size: 10pt; margin: 0; padding: 5px;">')

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


class ColumnDefinition:
    """
    Represents a column definition in byte mode.

    unit = "byte": start_byte..end_byte are byte-column indices and can span
                   multiple bytes (combined value).
    unit = "bit" : Bit-based definition with total_bits length.
                   start_bit is the absolute bit position.
                   Endianness determined by display_format (dec_be, dec_le, etc.)
    """

    def __init__(
            self,
            start_byte,
            end_byte,
            label,
            display_format,
            color_name=None,
            unit="byte",
            start_bit=0,
            total_bits=8,
    ):
        self.label = label
        # "hex", "binary", "dec_be", "dec_le", "tc_be", "tc_le", "ascii"
        self.display_format = display_format
        # highlight colour name: None, "Yellow", "Cyan", "Magenta", "Green"
        self.color_name = color_name or "None"
        # "byte" or "bit"
        self.unit = unit

        # Calculate byte positions based on unit
        if self.unit == "bit":
            # start_bit is absolute bit position
            self.start_bit = start_bit
            self.total_bits = total_bits

            # Calculate which bytes this spans
            self.start_byte = start_bit // 8
            end_bit_position = start_bit + total_bits - 1
            self.end_byte = end_bit_position // 8
        else:
            self.start_byte = start_byte
            self.end_byte = end_byte
            self.start_bit = 0
            self.total_bits = (end_byte - start_byte + 1) * 8


class ByteStructuredTableWidget(QTableWidget):
    """
    Hex grid with two-row header: column definitions on top, byte numbers below.
    - Continuous mode: columns are 0..row_size-1, data is linear bytes
    - Framed mode: one frame per row, columns are 0..max_frame_len-1 relative to sync
    - get_effective_length() returns length with trailing 0x00 ignored (for info only)
    """

    def __init__(self):
        super().__init__()
        self.bytes_data = None
        self.row_size = 16  # bytes per row in continuous mode
        self.column_definitions = []  # list[ColumnDefinition]

        # Framing
        self.frames = None  # list of numpy views/slices
        self.frame_pattern = None  # stored pattern (list of ints)
        self.frame_bit_offsets = None  # list of bit offsets within first byte of each frame

        # Constant column highlighting
        self.constant_columns = set()
        self._all_columns_info = None

        # Pattern highlighting (for frame sync search preview)
        self.pattern_highlight_positions = []  # List of bit positions
        self.pattern_highlight_length = 0  # Length of pattern in bits

        # Parent window reference (set by parent)
        self.parent_window = None

        # OPTIMIZED: Add row and column limits for large datasets
        self.max_display_rows = 1000  # CRITICAL: Limit frames displayed
        self.max_display_cols = 1000  # CRITICAL: Limit columns (bytes per frame) displayed

        self.setFont(QFont("Courier", 9))
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectItems)

        # Connect cell double-click for header editing
        self.cellDoubleClicked.connect(self._on_cell_double_clicked)

    def _on_cell_double_clicked(self, row, col):
        """
        Handle double-click on header labels:
        - If this column belongs to an existing ColumnDefinition -> open edit dialog
        - If this column is an undefined '?' region -> open a 'define bits' dialog
        """
        # Only react on the top label row
        if row != 0:
            return

        if self._all_columns_info is None:
            return
        if col < 0 or col >= len(self._all_columns_info):
            return

        byte_idx, bit_start, bit_end, col_def, is_undef = self._all_columns_info[col]

        # Existing definition: open normal edit
        if col_def is not None:
            if self.parent_window is None:
                return
            try:
                idx = self.column_definitions.index(col_def)
            except ValueError:
                return
            if 0 <= idx < self.parent_window.columns_list.count():
                item = self.parent_window.columns_list.item(idx)
                self.parent_window.edit_column_definition(item)
            return

        # Undefined '?' column: create a new bit-based definition prefilled
        if is_undef and self.parent_window is not None:
            abs_start_bit = byte_idx * 8 + bit_start
            total_bits = bit_end - bit_start + 1
            self.parent_window.add_definition_from_undefined(abs_start_bit, total_bits)

    def get_effective_length(self) -> int:
        """
        Return length in bytes excluding trailing 0x00 (padding).
        This does NOT affect what is displayed, only what we report.
        """
        if self.bytes_data is None:
            return 0
        data = self.bytes_data
        end = len(data)
        while end > 0 and data[end - 1] == 0:
            end -= 1
        return end

    def set_row_size(self, size: int):
        """Set number of bytes per row (continuous mode only)."""
        self.row_size = max(1, int(size))
        if self.frames is None:
            self.update_display()

    def set_bytes(self, byte_array):
        """Set byte data and refresh the view."""
        self.bytes_data = byte_array
        # CRITICAL: Clear frames when loading new data
        self.frames = None
        self.frame_pattern = None
        self.frame_bit_offsets = None
        self.update_display()

    def add_column_definition(self, col_def: ColumnDefinition):
        """Add a column definition."""
        self.column_definitions.append(col_def)
        self.update_display()

    def remove_column_definition(self, index: int):
        """Remove a column definition by index."""
        if 0 <= index < len(self.column_definitions):
            self.column_definitions.pop(index)
            self.update_display()

    def clear_column_definitions(self):
        """Clear all column definitions."""
        self.column_definitions.clear()
        self.update_display()

    def _get_def_for_column(self, col_index: int):
        """
        Return the last ColumnDefinition that covers this column, or None.
        Later definitions override earlier ones.
        """
        result = None
        for col_def in self.column_definitions:
            if col_def.start_byte <= col_index <= col_def.end_byte:
                result = col_def
        return result

    def clear_frames(self):
        """Disable framing and return to continuous mode display."""
        self.frames = None
        self.frame_pattern = None
        self.frame_bit_offsets = None
        self.update_display()

    def set_frame_pattern(self, pattern_list, pattern_bits=None):
        """
        Enable framing based on a sync pattern.
        pattern_list: list of bytes (for byte-aligned patterns)
        pattern_bits: list of bits (for bit-level patterns)
        Returns number of frames found.
        """

        if self.bytes_data is None or len(self.bytes_data) == 0:
            self.frames = None
            self.frame_pattern = None
            self.update_display()
            return 0

        if not pattern_list and not pattern_bits:
            self.frames = None
            self.frame_pattern = None
            self.update_display()
            return 0

        data = self.bytes_data
        starts = []
        starts_bit_positions = []  # Track actual bit positions for bit-aligned patterns

        # ALWAYS USE BIT-LEVEL SEARCH to find patterns at any bit position
        if not pattern_bits:
            # If called with byte pattern, convert to bits
            if pattern_list:
                pattern_bits = []
                for byte_val in pattern_list:
                    for bit in range(7, -1, -1):
                        pattern_bits.append((byte_val >> bit) & 1)
            else:
                self.frames = None
                self.frame_pattern = None
                self.update_display()
                return 0

        # Unpack bytes to bits for searching
        data_bits = np.unpackbits(data)
        pattern_bits_array = np.array(pattern_bits, dtype=np.uint8)
        pattern_len = len(pattern_bits_array)

        # DEBUG: Print pattern for verification
        pattern_str = ''.join(str(b) for b in pattern_bits_array)
        print(f"DEBUG: Searching for {pattern_len}-bit pattern: {pattern_str}")

        if pattern_len > len(data_bits):
            self.frames = None
            self.frame_pattern = None
            self.update_display()
            return 0

        # Sliding window search in bit space
        max_frames = 50_000
        prev_bit_pos = -1000  # Track previous match to avoid overlaps
        for bit_pos in range(len(data_bits) - pattern_len + 1):
            if np.array_equal(data_bits[bit_pos:bit_pos + pattern_len], pattern_bits_array):
                # Only add if this match doesn't overlap with previous
                if bit_pos >= prev_bit_pos + pattern_len:
                    # DEBUG: Print first few matches
                    if len(starts_bit_positions) < 5:
                        found_bits = ''.join(str(b) for b in data_bits[bit_pos:bit_pos + pattern_len])
                        byte_start = bit_pos // 8
                        byte_end = (bit_pos + pattern_len + 7) // 8
                        found_bytes = ' '.join(f'{int(b):02X}' for b in data[byte_start:byte_end])
                        print(f"DEBUG: Match {len(starts_bit_positions)} at bit {bit_pos}: bits={found_bits}, bytes={found_bytes}")

                    # Store the bit position
                    starts_bit_positions.append(bit_pos)
                    # Also store byte position (start of the byte containing this bit)
                    byte_pos = bit_pos // 8
                    starts.append(byte_pos)
                    prev_bit_pos = bit_pos

                    if len(starts) >= max_frames:
                        QMessageBox.information(
                            None,
                            "Frame Limit",
                            f"Found {max_frames:,} frames. Stopped to avoid memory issues."
                        )
                        break

        if not starts:
            self.frames = None
            self.frame_pattern = None
            self.update_display()
            return 0

        # Check if first match isn't at bit 0 - this means we're discarding data
        first_bit_offset = starts_bit_positions[0]
        if first_bit_offset > 0:
            discarded_bits = first_bit_offset
            discarded_bytes = discarded_bits // 8
            remaining_bits = discarded_bits % 8

            if remaining_bits > 0:
                discard_msg = f"{discarded_bytes} bytes and {remaining_bits} bits"
            else:
                discard_msg = f"{discarded_bytes} bytes"

            reply = QMessageBox.warning(
                None,
                "Frame Sync Offset",
                f"First sync pattern found at bit {first_bit_offset} (not at start of file).\n\n"
                f"This will discard the first {discard_msg} of data before the first frame.\n\n"
                f"Continue with frame sync?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                self.frames = None
                self.frame_pattern = None
                return 0

        # CRITICAL: Limit frame creation to prevent crashes
        # Don't even create frame objects for more than we'll display
        total_frames = len(starts)
        safe_limit = 1000  # Match max_display_rows

        if total_frames > safe_limit:
            # Warn user BEFORE creating frames
            reply = QMessageBox.warning(
                None,
                "Large Frame Count",
                f"Found {total_frames:,} frame sync matches.\n\n"
                f"Displaying only first {safe_limit:,} frames.\n"
                f"Consider using a more specific sync pattern.\n\n"
                f"Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                self.frames = None
                self.frame_pattern = None
                return 0

        # Only create the limited number of frames
        frames = []
        frame_bit_offsets = []
        create_count = min(total_frames, safe_limit)

        # Always use bit-level frame boundaries with proper bit shifting
        data_bits = np.unpackbits(data)

        for idx in range(create_count):
            start_bit = starts_bit_positions[idx]
            # Frame goes from this pattern start to the next pattern start (or end of data)
            if idx + 1 < len(starts_bit_positions):
                end_bit = starts_bit_positions[idx + 1]
            else:
                end_bit = len(data) * 8  # End of data in bits

            # Extract the bits for this frame
            frame_bits = data_bits[start_bit:end_bit]

            # Pad to byte boundary if needed
            if len(frame_bits) % 8 != 0:
                padding_bits = 8 - (len(frame_bits) % 8)
                frame_bits = np.concatenate([frame_bits, np.zeros(padding_bits, dtype=np.uint8)])

            # Convert bits back to bytes
            frame_bytes = np.packbits(frame_bits)

            frames.append(frame_bytes)
            # Frames now always start at bit 0 (no offset needed)
            frame_bit_offsets.append(0)


        self.frames = frames
        self.frame_pattern = pattern_list
        self.frame_bit_offsets = frame_bit_offsets

        # Note: status_label belongs to parent window, not this widget
        QApplication.processEvents()

        # Now safe to update display
        try:
            self.update_display()
        except Exception as e:
            # If still crashes, try with even fewer
            QMessageBox.critical(
                None,
                "Display Error",
                f"Error displaying {len(frames)} frames: {e}\n"
                f"Trying with only 100 frames..."
            )
            # Recreate with only 100
            frames = []
            frame_bit_offsets = []
            for idx in range(min(100, len(starts_bit_positions))):
                start_bit = starts_bit_positions[idx]
                if idx + 1 < len(starts_bit_positions):
                    end_bit = starts_bit_positions[idx + 1]
                else:
                    end_bit = len(data) * 8

                # Extract and shift bits
                frame_bits = data_bits[start_bit:end_bit]
                if len(frame_bits) % 8 != 0:
                    padding_bits = 8 - (len(frame_bits) % 8)
                    frame_bits = np.concatenate([frame_bits, np.zeros(padding_bits, dtype=np.uint8)])
                frame_bytes = np.packbits(frame_bits)

                frames.append(frame_bytes)
                frame_bit_offsets.append(0)
            self.frames = frames
            self.frame_bit_offsets = frame_bit_offsets
            self.update_display()

        return total_frames

    def find_pattern_positions(self, pattern_bits):
        """
        Find all positions of a bit pattern without creating frames.
        Returns (list of bit positions, max_frame_length_bytes).
        Used for highlighting patterns before framing.
        """
        if self.bytes_data is None or len(self.bytes_data) == 0:
            return [], 0

        if not pattern_bits:
            return [], 0

        data = self.bytes_data
        positions = []

        # Always use bit-level search
        data_bits = np.unpackbits(data)
        pattern_bits_array = np.array(pattern_bits, dtype=np.uint8)
        pattern_len = len(pattern_bits_array)

        if pattern_len > len(data_bits):
            return [], 0

        # Search for pattern with overlap prevention
        prev_bit_pos = -1000
        for bit_pos in range(len(data_bits) - pattern_len + 1):
            if np.array_equal(data_bits[bit_pos:bit_pos + pattern_len], pattern_bits_array):
                # Only add if this match doesn't overlap with previous
                if bit_pos >= prev_bit_pos + pattern_len:
                    positions.append(bit_pos)
                    prev_bit_pos = bit_pos

                    # Limit for performance
                    if len(positions) >= 50000:
                        break

        # Calculate max frame length
        max_frame_len = 0
        if positions:
            for i in range(len(positions)):
                if i + 1 < len(positions):
                    # Frame length from this position to next
                    frame_bits = positions[i + 1] - positions[i]
                    frame_bytes = (frame_bits + 7) // 8  # Round up
                    max_frame_len = max(max_frame_len, frame_bytes)
                else:
                    # Last frame: from this position to end of data
                    frame_bits = len(data) * 8 - positions[i]
                    frame_bytes = (frame_bits + 7) // 8
                    max_frame_len = max(max_frame_len, frame_bytes)

        return positions, max_frame_len

    def set_pattern_highlights(self, bit_positions, pattern_length):
        """
        Set pattern highlights for preview display.
        bit_positions: list of bit positions where pattern was found
        pattern_length: length of pattern in bits
        """
        self.pattern_highlight_positions = bit_positions
        self.pattern_highlight_length = pattern_length
        self.update_display()

    def clear_pattern_highlights(self):
        """Clear pattern highlights."""
        self.pattern_highlight_positions = []
        self.pattern_highlight_length = 0
        self.update_display()

    def highlight_constant_columns(self):
        """
        Mark columns where the value is constant across all rows/frames.
        Works with both byte-level and bit-level columns.
        Empty cells (out-of-range) are ignored.
        """
        self.constant_columns.clear()

        # Build column structure to get table column indices
        if self.frames is not None and len(self.frames) > 0:
            frames = self.frames
            max_len = max(len(f) for f in frames)
            _, _, _, _, all_columns_info = self._build_headers(max_len)

            # Check each table column for constant values
            for table_col_idx, (byte_idx, bit_start, bit_end, col_def, is_undef) in enumerate(all_columns_info):
                val_text = None
                constant = True

                for frame in frames:
                    if byte_idx >= len(frame):
                        continue

                    # Extract value for this column from this frame
                    if col_def is not None and col_def.unit == "bit":
                        # Bit-based column - extract bit range
                        start_byte = col_def.start_bit // 8
                        if start_byte < len(frame):
                            end_byte = min((col_def.start_bit + col_def.total_bits - 1) // 8 + 1, len(frame))
                            byte_slice = frame[start_byte:end_byte]
                            text = self._format_bit_range(byte_slice, col_def)
                        else:
                            continue
                    elif is_undef:
                        # Undefined bit range
                        if byte_idx < len(frame):
                            abs_start_bit = byte_idx * 8 + bit_start
                            num_bits = bit_end - bit_start + 1
                            byte_slice = frame[byte_idx:byte_idx + 1]
                            text = self._format_undefined_bits(byte_slice, abs_start_bit, num_bits)
                        else:
                            continue
                    else:
                        # Regular byte column
                        text = "{:02X}".format(int(frame[byte_idx]))

                    if val_text is None:
                        val_text = text
                    elif text != val_text:
                        constant = False
                        break

                if constant and val_text is not None and val_text != "":
                    self.constant_columns.add(table_col_idx)

        # Continuous mode (similar logic)
        elif self.bytes_data is not None:
            data = self.bytes_data
            num_bytes = len(data)
            if num_bytes == 0:
                self.update_display()
                return

            num_rows = (num_bytes + self.row_size - 1) // self.row_size
            _, _, _, _, all_columns_info = self._build_headers(self.row_size)

            # Check each table column for constant values
            for table_col_idx, (byte_offset, bit_start, bit_end, col_def, is_undef) in enumerate(all_columns_info):
                val_text = None
                constant = True

                for row in range(num_rows):
                    absolute_byte_idx = row * self.row_size + byte_offset

                    if absolute_byte_idx >= num_bytes:
                        continue

                    # Extract value for this column from this row
                    if col_def is not None and col_def.unit == "bit":
                        # Bit-based column
                        start_byte = col_def.start_bit // 8 + row * self.row_size
                        if start_byte < num_bytes:
                            end_byte = min((col_def.start_bit + col_def.total_bits - 1) // 8 + 1 + row * self.row_size, num_bytes)
                            byte_slice = data[start_byte:end_byte]
                            text = self._format_bit_range(byte_slice, col_def)
                        else:
                            continue
                    elif is_undef:
                        # Undefined bit range
                        abs_start_bit = absolute_byte_idx * 8 + bit_start
                        num_bits = bit_end - bit_start + 1
                        byte_slice = data[absolute_byte_idx:absolute_byte_idx + 1]
                        text = self._format_undefined_bits(byte_slice, abs_start_bit, num_bits)
                    else:
                        # Regular byte column
                        text = "{:02X}".format(int(data[absolute_byte_idx]))

                    if val_text is None:
                        val_text = text
                    elif text != val_text:
                        constant = False
                        break

                if constant and val_text is not None and val_text != "":
                    self.constant_columns.add(table_col_idx)

        self.update_display()

    def clear_constant_highlights(self):
        """Clear constant column highlighting."""
        self.constant_columns.clear()
        self.update_display()

    def _color_from_name(self, name: str):
        """Map a simple color name to a QColor (for column highlighting)."""
        if name == "Yellow":
            return QColor(255, 255, 180)
        if name == "Cyan":
            return QColor(180, 255, 255)
        if name == "Magenta":
            return QColor(255, 180, 255)
        if name == "Green":
            return QColor(200, 255, 200)
        return None  # "None" or unknown

    def _is_byte_in_pattern_highlight(self, byte_index):
        """
        Check if a byte at the given index overlaps with any pattern highlight.
        Returns True if the byte should be highlighted.
        """
        if not self.pattern_highlight_positions or self.pattern_highlight_length == 0:
            return False

        byte_start_bit = byte_index * 8
        byte_end_bit = byte_start_bit + 7

        # Check if this byte overlaps with any pattern match
        for pattern_bit_pos in self.pattern_highlight_positions:
            pattern_end_bit = pattern_bit_pos + self.pattern_highlight_length - 1

            # Check for overlap
            if not (byte_end_bit < pattern_bit_pos or byte_start_bit > pattern_end_bit):
                return True

        return False

    def _build_headers(self, num_cols: int):
        """
        Build column headers that split bytes when needed.
        If a byte has both defined and undefined bits, it gets multiple columns.

        Returns (col_map, label_spans, byte_labels, subbit_labels, all_columns_info) where:
        - col_map: dict mapping original byte index to list of
                   (table_col, (bit_start_in_byte, bit_end_in_byte), col_def_or_none, is_undefined)
        - label_spans: list of (table_col, span_width, label_text, color_name)
        - byte_labels: list of byte numbers for display (one per table column)
        - subbit_labels: list of bit ranges within the byte (e.g. "0-5", "6-7") per table column
        - all_columns_info: list of (byte_idx, bit_start_in_byte, bit_end_in_byte, col_def_or_none, is_undefined)
        """
        # First, for each byte, determine all the bit ranges that need separate columns
        byte_ranges = {}  # byte_idx -> list of (bit_start_in_byte, bit_end_in_byte, col_def_or_none, is_undefined)

        for byte_idx in range(num_cols):
            byte_start_bit = byte_idx * 8
            byte_end_bit = byte_start_bit + 7

            # Find all definitions that touch this byte
            ranges_in_byte = []

            for col_def in self.column_definitions:
                if col_def.unit == "bit":
                    def_start_bit = col_def.start_bit
                    def_end_bit = def_start_bit + col_def.total_bits - 1

                    # Check if this definition overlaps with current byte
                    if def_start_bit <= byte_end_bit and def_end_bit >= byte_start_bit:
                        # Calculate overlap
                        overlap_start_bit = max(def_start_bit, byte_start_bit)
                        overlap_end_bit = min(def_end_bit, byte_end_bit)

                        # Convert to bit positions within the byte (0-7)
                        start_in_byte = overlap_start_bit - byte_start_bit
                        end_in_byte = overlap_end_bit - byte_start_bit

                        ranges_in_byte.append((start_in_byte, end_in_byte, col_def, False))

                elif col_def.start_byte <= byte_idx <= col_def.end_byte:
                    # Byte-level definition covers entire byte
                    ranges_in_byte.append((0, 7, col_def, False))
                    break  # No need to check further, entire byte is covered

            # If we have bit-level definitions, find undefined gaps
            if ranges_in_byte and not (
                    len(ranges_in_byte) == 1 and ranges_in_byte[0][0] == 0 and ranges_in_byte[0][1] == 7
            ):
                # Sort by starting bit
                ranges_in_byte.sort(key=lambda x: x[0])

                # Find gaps (undefined bits)
                all_ranges = []
                current_bit = 0

                for start_in_byte, end_in_byte, col_def, is_undef in ranges_in_byte:
                    # Add undefined range before this definition if there's a gap
                    if current_bit < start_in_byte:
                        all_ranges.append((current_bit, start_in_byte - 1, None, True))

                    # Add the defined range
                    all_ranges.append((start_in_byte, end_in_byte, col_def, False))
                    current_bit = end_in_byte + 1

                # Add undefined range at the end if needed
                if current_bit <= 7:
                    all_ranges.append((current_bit, 7, None, True))

                byte_ranges[byte_idx] = all_ranges
            elif ranges_in_byte:
                # Single range covering the whole byte
                byte_ranges[byte_idx] = ranges_in_byte
            else:
                # No definitions, entire byte is default hex
                byte_ranges[byte_idx] = [(0, 7, None, False)]

        # Now build the actual column structure
        all_columns_info = []
        col_map = {}
        table_col = 0

        for byte_idx in range(num_cols):
            if byte_idx not in byte_ranges:
                # Default: single column for this byte
                col_map[byte_idx] = [(table_col, (0, 7), None, False)]
                all_columns_info.append((byte_idx, 0, 7, None, False))
                table_col += 1
            else:
                col_map[byte_idx] = []
                for start_in_byte, end_in_byte, col_def, is_undef in byte_ranges[byte_idx]:
                    col_map[byte_idx].append((table_col, (start_in_byte, end_in_byte), col_def, is_undef))
                    all_columns_info.append((byte_idx, start_in_byte, end_in_byte, col_def, is_undef))
                    table_col += 1

        # Build label spans, byte labels, and sub-bit labels
        label_spans = []
        byte_labels = []
        subbit_labels = []

        current_label = None
        current_color = "None"
        current_start = None
        current_len = 0

        for idx, (byte_idx, bit_start, bit_end, col_def, is_undef) in enumerate(all_columns_info):
            # Decide what label (if any) this column should have
            if col_def is not None and col_def.label:
                label_text = col_def.label
                color_name = col_def.color_name
            elif is_undef:
                label_text = "?"
                color_name = "None"
            else:
                label_text = None
                color_name = "None"

            # Merge contiguous columns with the same label
            if label_text is None:
                # Flush any open span
                if current_label is not None:
                    label_spans.append((current_start, current_len, current_label, current_color))
                    current_label = None
                    current_start = None
                    current_len = 0
            else:
                if (
                        current_label == label_text
                        and current_color == color_name
                        and current_start is not None
                        and current_start + current_len == idx
                ):
                    # Extend current span
                    current_len += 1
                else:
                    # Flush previous span
                    if current_label is not None:
                        label_spans.append((current_start, current_len, current_label, current_color))
                    # Start new span
                    current_label = label_text
                    current_color = color_name
                    current_start = idx
                    current_len = 1

            # Byte label: show byte index
            byte_labels.append(f"{byte_idx}")

            # Sub-bit label: show bit range within that byte
            subbit_labels.append(f"{bit_start}-{bit_end}")

        # Flush trailing span
        if current_label is not None:
            label_spans.append((current_start, current_len, current_label, current_color))

        return col_map, label_spans, byte_labels, subbit_labels, all_columns_info

    def _extract_bits_from_bytes(self, byte_slice, start_bit, total_bits):
        """
        Extract a specific bit range from a byte array.
        Always uses MSB-first bit ordering (bit 0 = MSB).

        Args:
            byte_slice: numpy array of bytes
            start_bit: absolute starting bit position (e.g., bit 11 = byte 1, bit 3)
            total_bits: total number of bits to extract

        Returns:
            Integer value of the extracted bits
        """
        if len(byte_slice) == 0:
            return 0

        # Calculate bit position within the first byte
        bit_offset_in_first_byte = start_bit % 8

        # Convert bytes to one big integer (big-endian)
        full_value = 0
        for b in byte_slice:
            full_value = (full_value << 8) | int(b)

        # Calculate how many bits we have total
        total_available_bits = len(byte_slice) * 8

        # MSB-first: bit 0 is leftmost (most significant)
        # Shift right to discard bits after our range
        bits_after = total_available_bits - bit_offset_in_first_byte - total_bits
        if bits_after > 0:
            full_value >>= bits_after

        # Mask to keep only our bits
        mask = (1 << total_bits) - 1
        return full_value & mask

    def _get_undefined_bit_ranges(self, num_cols):
        """
        Find bit ranges not covered by any column definition.
        Returns list of (start_bit, end_bit, start_byte, end_byte) for undefined ranges.
        """
        # Create a set of all defined bit positions
        defined_bits = set()

        for col_def in self.column_definitions:
            if col_def.unit == "bit":
                for bit_pos in range(col_def.start_bit, col_def.start_bit + col_def.total_bits):
                    defined_bits.add(bit_pos)
            else:
                # Byte definitions cover all bits in their byte range
                for byte_idx in range(col_def.start_byte, col_def.end_byte + 1):
                    for bit_in_byte in range(8):
                        bit_pos = byte_idx * 8 + bit_in_byte
                        defined_bits.add(bit_pos)

        # Find undefined ranges within our column range
        max_bit = num_cols * 8
        undefined_ranges = []

        current_start = None
        for bit_pos in range(max_bit):
            if bit_pos not in defined_bits:
                if current_start is None:
                    current_start = bit_pos
            else:
                if current_start is not None:
                    # End of undefined range
                    start_byte = current_start // 8
                    end_byte = (bit_pos - 1) // 8
                    undefined_ranges.append((current_start, bit_pos - 1, start_byte, end_byte))
                    current_start = None

        # Handle trailing undefined range
        if current_start is not None:
            start_byte = current_start // 8
            end_byte = (max_bit - 1) // 8
            undefined_ranges.append((current_start, max_bit - 1, start_byte, end_byte))

        return undefined_ranges

    def _format_multi_byte_value(self, byte_slice, col_def: ColumnDefinition):
        """
        Format a slice of bytes according to the display_format in col_def.
        """
        if col_def is None:
            # Default: hex bytes
            return " ".join("{:02X}".format(int(b)) for b in byte_slice)

        fmt = col_def.display_format
        # Backwards compatibility
        if fmt == "hex":
            fmt = "hex_be"
        if fmt == "ascii":
            fmt = "ascii_be"

        raw = bytes(int(b) for b in byte_slice)

        # Hex: MSBF/LSBF both render as bytes in table order
        if fmt in ["hex_be", "hex_le"]:
            return " ".join("{:02X}".format(int(b)) for b in byte_slice)

        if fmt == "binary":
            return " ".join("{:08b}".format(int(b)) for b in byte_slice)

        if fmt == "dec_be":
            val = int.from_bytes(raw, byteorder="big", signed=False)
            return str(val)

        if fmt == "dec_le":
            val = int.from_bytes(raw, byteorder="little", signed=False)
            return str(val)

        if fmt == "tc_be":
            val = int.from_bytes(raw, byteorder="big", signed=True)
            return str(val)

        if fmt == "tc_le":
            val = int.from_bytes(raw, byteorder="little", signed=True)
            return str(val)

        if fmt in ["ascii_be", "ascii_le"]:
            chars = []
            for b in byte_slice:
                v = int(b)
                if 32 <= v <= 126:
                    chars.append(chr(v))
                else:
                    chars.append(".")
            return "".join(chars)

        # Fallback
        return " ".join("{:02X}".format(int(b)) for b in byte_slice)

    def _format_undefined_bits(self, byte_slice, start_bit, num_bits):
        """
        Format undefined bits - as hex if 4 bits, otherwise binary.
        """
        if len(byte_slice) == 0 or num_bits == 0:
            return ""

        # Extract the bits (MSB-first) using existing helper
        bit_value = self._extract_bits_from_bytes(byte_slice, start_bit, num_bits)

        # Format as hex if exactly 4 bits (1 nibble), otherwise binary
        if num_bits == 4:
            return f"0x{bit_value:X}"
        else:
            return f"{bit_value:0{num_bits}b}"

    def _format_bit_range(self, row_bytes, col_def: ColumnDefinition):
        """
        Format a bit-based field (unit='bit') according to display_format.
        MSBF/LSBF here mean Most/Least Significant Bit First in the bit field,
        not byte ordering. Works for any bit length (e.g. 11 bits).

        Frames are bit-shifted during creation so patterns always start at bit 0.
        """
        total_bits = col_def.total_bits
        if total_bits <= 0 or row_bytes is None or len(row_bytes) == 0:
            return ""

        # Build bits for the given byte slice in MSB-first order
        bits = []
        for b in row_bytes:
            v = int(b)
            for bit in range(7, -1, -1):
                bits.append((v >> bit) & 1)

        # col_def.start_bit is relative to the frame start
        # Frames are now bit-shifted so patterns always start at bit 0
        # Account for which byte we're starting from (col_def.start_byte)
        local_start = col_def.start_bit - (col_def.start_byte * 8)
        if local_start < 0:
            local_start = 0

        if local_start >= len(bits):
            return ""

        end = local_start + total_bits
        if end > len(bits):
            end = len(bits)
        field_bits = bits[local_start:end]
        if not field_bits:
            return ""

        # DEBUG: Print extraction details for sync column
        if col_def.label == "Sync":
            bits_str = ''.join(str(b) for b in field_bits)
            bytes_hex = ' '.join(f'{int(b):02X}' for b in row_bytes)
            print(f"DEBUG: Extracting '{col_def.label}': bytes=[{bytes_hex}], local_start={local_start}, bits={bits_str}")

        fmt = col_def.display_format

        # --- ASCII (only valid when byte-aligned & multiple of 8; dialog enforces that) ---
        if fmt in ("ascii_be", "ascii_le"):
            msbf = (fmt == "ascii_be")
            fb = field_bits[:]
            # pad to full bytes if somehow not aligned
            if len(fb) % 8 != 0:
                pad = 8 - (len(fb) % 8)
                fb.extend([0] * pad)

            chars = []
            for i in range(0, len(fb), 8):
                byte_bits = fb[i:i + 8]
                if not msbf:
                    byte_bits = list(reversed(byte_bits))
                val = 0
                for b in byte_bits:
                    val = (val << 1) | b
                if 32 <= val <= 126:
                    chars.append(chr(val))
                else:
                    chars.append(".")
            return "".join(chars)

        # --- Binary (MSBF) ---
        if fmt == "binary":
            return "".join(str(b) for b in field_bits)

        # Helper to turn bit list into integer
        def bits_to_int(b_list):
            val = 0
            for b in b_list:
                val = (val << 1) | b
            return val

        # --- Hex / Decimal / Twos complement with MSBF vs LSBF ---
        if fmt in ("hex_be", "hex_le", "dec_be", "dec_le", "tc_be", "tc_le"):
            be = fmt.endswith("_be")
            signed = fmt.startswith("tc")

            if be:
                ordered_bits = field_bits
            else:
                ordered_bits = list(reversed(field_bits))

            val = bits_to_int(ordered_bits)

            if fmt.startswith("hex"):
                hex_digits = (len(ordered_bits) + 3) // 4
                return f"0x{val:0{hex_digits}X}"

            if signed:
                # Two's complement with given bit width
                bit_width = len(ordered_bits)
                sign_bit = 1 << (bit_width - 1)
                if val & sign_bit:
                    val -= 1 << bit_width

            return str(val)

        # Fallback: just show binary if we somehow get an unknown format
        return "".join(str(b) for b in field_bits)

    def update_display(self):
        """Render table either in continuous mode or framed mode with three header rows:
           row 0: labels, row 1: byte index, row 2: sub-byte bit ranges.
           For any ColumnDefinition that spans multiple bytes, both the header
           and the data cell are merged so you only see one value.
        """

        # If nothing loaded, clear table cleanly
        if self.bytes_data is None:
            self.setRowCount(0)
            self.setColumnCount(0)
            return

        # CRITICAL: Disable updates during table construction to prevent crashes
        self.setUpdatesEnabled(False)

        # Clear existing spans for a fresh layout
        self.clearSpans()

        # =====================================================
        # ================  FRAMED MODE  ======================
        # =====================================================
        if self.frames is not None and len(self.frames) > 0:
            frames = self.frames

            # OPTIMIZED: Limit displayed rows for performance
            total_data_rows = len(frames)
            num_data_rows = min(total_data_rows, self.max_display_rows)

            if num_data_rows < total_data_rows:
                frames = frames[:num_data_rows]

            max_len = max(len(f) for f in frames)

            # CRITICAL: Limit columns to prevent freeze (only for files >= 5MB)
            file_size = len(self.bytes_data) if self.bytes_data is not None else 0
            is_large_file = file_size >= 5_000_000  # 5MB threshold

            if is_large_file and max_len > self.max_display_cols:
                QMessageBox.information(
                    None,
                    "Wide Frames",
                    f"Large file with frames {max_len:,} bytes wide.\n\n"
                    f"Displaying only first {self.max_display_cols:,} bytes per frame."
                )
                max_len = self.max_display_cols

            # Build the column structure
            col_map, label_spans, byte_labels, subbit_labels, all_columns_info = self._build_headers(max_len)
            # Store for click handling
            self._all_columns_info = all_columns_info

            # Decide which table column "owns" the value for each definition
            first_col_for_def = {}
            for idx, (_, _, _, col_def, is_undef) in enumerate(all_columns_info):
                if col_def is not None and col_def not in first_col_for_def:
                    first_col_for_def[col_def] = idx

            # Compute contiguous column spans for each definition (bit OR byte)
            field_spans = []
            current_def = None
            current_start = None
            current_len = 0

            for idx, (_, _, _, col_def, is_undef) in enumerate(all_columns_info):
                if col_def is not None:
                    if current_def is col_def:
                        current_len += 1
                    else:
                        if current_def is not None:
                            field_spans.append((current_start, current_len, current_def))
                        current_def = col_def
                        current_start = idx
                        current_len = 1
                else:
                    if current_def is not None:
                        field_spans.append((current_start, current_len, current_def))
                        current_def = None
                        current_start = None
                        current_len = 0

            if current_def is not None:
                field_spans.append((current_start, current_len, current_def))

            # Set up table with split columns
            num_table_cols = len(all_columns_info)
            self.setColumnCount(num_table_cols)
            # 3 header rows (labels, bytes, sub-bits) + data rows
            self.setRowCount(num_data_rows + 3)

            # === Header row 0: definition labels ===
            for col_idx in range(num_table_cols):
                item = QTableWidgetItem("")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setBackground(QColor(220, 220, 220))
                self.setItem(0, col_idx, item)

            for table_col, span_width, label_text, color_name in label_spans:
                item = QTableWidgetItem(label_text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

                if label_text == "?":
                    # Gray background for undefined bits
                    item.setBackground(QColor(200, 200, 200))
                    item.setFont(QFont("Arial", 9, QFont.Weight.Normal))
                else:
                    # Apply definition color
                    if color_name and color_name != "None":
                        bg_color = self._color_from_name(color_name)
                        if bg_color:
                            item.setBackground(bg_color)
                        else:
                            item.setBackground(QColor(200, 220, 255))
                    else:
                        item.setBackground(QColor(200, 220, 255))
                    item.setFont(QFont("Arial", 9, QFont.Weight.Bold))

                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.setItem(0, table_col, item)

                if span_width > 1:
                    # This makes the label appear once across multiple byte columns
                    self.setSpan(0, table_col, 1, span_width)

            # === Header row 1: byte numbers ===
            for col_idx, byte_label in enumerate(byte_labels):
                item = QTableWidgetItem(byte_label)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setBackground(QColor(240, 240, 240))
                item.setFont(QFont("Courier", 8))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.setItem(1, col_idx, item)

            # === Header row 2: sub-byte bit ranges ===
            for col_idx, sub_label in enumerate(subbit_labels):
                item = QTableWidgetItem(sub_label)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setBackground(QColor(250, 250, 250))
                item.setFont(QFont("Courier", 8))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.setItem(2, col_idx, item)

            # === Data rows ===
            row_labels = ["Labels", "Bytes", "Bits"]

            # OPTIMIZED: Show warning if rows were truncated
            if num_data_rows < total_data_rows:
                row_labels[0] += f" (showing {num_data_rows}/{total_data_rows})"


            for data_row_idx in range(num_data_rows):
                table_row = data_row_idx + 3
                frame = frames[data_row_idx]
                flen = len(frame)

                # Show truncation indicator in row label if frame is wider than display
                if flen > max_len:
                    row_labels.append(f"{data_row_idx} ({flen}→{max_len})")
                else:
                    row_labels.append(f"{data_row_idx} ({flen})")

                # Merge columns that belong to the same field so the value spans them
                for start_col, span_len, col_def in field_spans:
                    if span_len > 1:
                        self.setSpan(table_row, start_col, 1, span_len)

                for table_col_idx, (byte_idx, bit_start, bit_end, col_def, is_undef) in enumerate(all_columns_info):
                    if byte_idx >= flen:
                        item = QTableWidgetItem("")
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        self.setItem(table_row, table_col_idx, item)
                        continue

                    byte_val = int(frame[byte_idx])
                    text = ""
                    tooltip = ""

                    if col_def is not None:
                        # Defined field
                        if col_def.unit == "bit":
                            # Only show one combined value in the first column of that field
                            if table_col_idx == first_col_for_def.get(col_def, table_col_idx):
                                start_byte = col_def.start_byte
                                end_byte = col_def.end_byte
                                if start_byte < flen:
                                    slice_end = min(end_byte + 1, flen)
                                    byte_slice = frame[start_byte:slice_end]
                                    text = self._format_bit_range(byte_slice, col_def)
                                    tooltip = "{} (bit {}-{}, {} bits, {})".format(
                                        col_def.label,
                                        col_def.start_bit,
                                        col_def.start_bit + col_def.total_bits - 1,
                                        col_def.total_bits,
                                        col_def.display_format
                                    )
                        elif col_def.unit == "byte":
                            # Multi-byte field: again, only show in first column
                            if table_col_idx == first_col_for_def.get(col_def, table_col_idx):
                                start_byte = col_def.start_byte
                                end_byte = col_def.end_byte
                                if start_byte < flen:
                                    slice_end = min(end_byte + 1, flen)
                                    byte_slice = frame[start_byte:slice_end]
                                    text = self._format_multi_byte_value(byte_slice, col_def)
                                    tooltip = "{} ({})".format(
                                        col_def.label,
                                        col_def.display_format
                                    ) if col_def.label else ""

                    elif is_undef:
                        # Undefined bits - show actual bits
                        abs_start_bit = byte_idx * 8 + bit_start
                        num_bits = bit_end - bit_start + 1
                        byte_slice = frame[byte_idx:byte_idx + 1]
                        text = self._format_undefined_bits(byte_slice, abs_start_bit, num_bits)
                        tooltip = f"{num_bits} undefined bits (bit {abs_start_bit}-{abs_start_bit + num_bits - 1})"

                    else:
                        # Default: single byte hex
                        text = "{:02X}".format(byte_val)

                    item = QTableWidgetItem(text)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    if tooltip and text:
                        item.setToolTip(tooltip)

                    # Coloring
                    if table_col_idx in self.constant_columns and text != "":
                        item.setBackground(QColor(255, 255, 180))
                    else:
                        if col_def is not None:
                            col_color = self._color_from_name(col_def.color_name)
                            if col_color is not None and text != "":
                                item.setBackground(col_color)

                    self.setItem(table_row, table_col_idx, item)

            self.setVerticalHeaderLabels(row_labels)
            self.resizeColumnsToContents()
            self.setUpdatesEnabled(True)  # Re-enable updates before returning
            return  # IMPORTANT: end framed mode here

        # =====================================================
        # ==============  CONTINUOUS MODE  ====================
        # =====================================================
        data = self.bytes_data
        num_bytes = len(data)

        if num_bytes == 0:
            self.setRowCount(0)
            self.setColumnCount(0)
            return

        # Build the column structure for one row
        col_map, label_spans, byte_labels, subbit_labels, all_columns_info = self._build_headers(self.row_size)
        # Store for click handling
        self._all_columns_info = all_columns_info

        # Decide which table column "owns" the value for each definition
        first_col_for_def = {}
        for idx, (_, _, _, col_def, is_undef) in enumerate(all_columns_info):
            if col_def is not None and col_def not in first_col_for_def:
                first_col_for_def[col_def] = idx

        # Compute contiguous column spans for each definition (bit OR byte)
        field_spans = []
        current_def = None
        current_start = None
        current_len = 0

        for idx, (_, _, _, col_def, is_undef) in enumerate(all_columns_info):
            if col_def is not None:
                if current_def is col_def:
                    current_len += 1
                else:
                    if current_def is not None:
                        field_spans.append((current_start, current_len, current_def))
                    current_def = col_def
                    current_start = idx
                    current_len = 1
            else:
                if current_def is not None:
                    field_spans.append((current_start, current_len, current_def))
                    current_def = None
                    current_start = None
                    current_len = 0

        if current_def is not None:
            field_spans.append((current_start, current_len, current_def))

        # Set up table with split columns
        num_table_cols = len(all_columns_info)
        self.setColumnCount(num_table_cols)

        # OPTIMIZED: Limit displayed rows for performance
        total_data_rows = (num_bytes + self.row_size - 1) // self.row_size
        num_data_rows = min(total_data_rows, self.max_display_rows)

        # 3 header rows + data rows
        self.setRowCount(num_data_rows + 3)

        # === Header row 0: labels ===
        for col_idx in range(num_table_cols):
            item = QTableWidgetItem("")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setBackground(QColor(220, 220, 220))
            self.setItem(0, col_idx, item)

        for table_col, span_width, label_text, color_name in label_spans:
            item = QTableWidgetItem(label_text)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

            if label_text == "?":
                item.setBackground(QColor(200, 200, 200))
                item.setFont(QFont("Arial", 9, QFont.Weight.Normal))
            else:
                if color_name and color_name != "None":
                    bg_color = self._color_from_name(color_name)
                    if bg_color:
                        item.setBackground(bg_color)
                    else:
                        item.setBackground(QColor(200, 220, 255))
                else:
                    item.setBackground(QColor(200, 220, 255))
                item.setFont(QFont("Arial", 9, QFont.Weight.Bold))

            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(0, table_col, item)

            if span_width > 1:
                # merged header cell (e.g. "Sync" over bytes 1–2)
                self.setSpan(0, table_col, 1, span_width)

        # === Header row 1: byte numbers ===
        for col_idx, byte_label in enumerate(byte_labels):
            item = QTableWidgetItem(byte_label)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setBackground(QColor(240, 240, 240))
            item.setFont(QFont("Courier", 8))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(1, col_idx, item)

        # === Header row 2: bit ranges ===
        for col_idx, sub_label in enumerate(subbit_labels):
            item = QTableWidgetItem(sub_label)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setBackground(QColor(250, 250, 250))
            item.setFont(QFont("Courier", 8))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(2, col_idx, item)

        # === Data rows ===
        row_labels = ["Labels", "Bytes", "Bits"]

        # OPTIMIZED: Show warning if rows were truncated
        if num_data_rows < total_data_rows:
            row_labels[0] += f" (showing {num_data_rows}/{total_data_rows})"

        for data_row_idx in range(num_data_rows):
            table_row = data_row_idx + 3
            row_labels.append(str(data_row_idx))
            base_byte_idx = data_row_idx * self.row_size

            # Merge columns for the same field so one wide cell shows the value
            for start_col, span_len, col_def in field_spans:
                if span_len > 1:
                    self.setSpan(table_row, start_col, 1, span_len)

            for table_col_idx, (byte_offset, bit_start, bit_end, col_def, is_undef) in enumerate(all_columns_info):
                absolute_byte_idx = base_byte_idx + byte_offset

                if absolute_byte_idx >= num_bytes:
                    item = QTableWidgetItem("")
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.setItem(table_row, table_col_idx, item)
                    continue

                byte_val = int(data[absolute_byte_idx])
                text = ""
                tooltip = ""

                if col_def is not None:
                    if col_def.unit == "bit":
                        # Single combined value for the whole bit field
                        if table_col_idx == first_col_for_def.get(col_def, table_col_idx):
                            start_byte = col_def.start_byte + base_byte_idx
                            end_byte = col_def.end_byte + base_byte_idx
                            if start_byte < num_bytes:
                                slice_end = min(end_byte + 1, num_bytes)
                                byte_slice = data[start_byte:slice_end]
                                text = self._format_bit_range(byte_slice, col_def)
                                tooltip = "{} (bit {}-{}, {} bits, {})".format(
                                    col_def.label,
                                    col_def.start_bit,
                                    col_def.start_bit + col_def.total_bits - 1,
                                    col_def.total_bits,
                                    col_def.display_format
                                )
                    elif col_def.unit == "byte":
                        # Single combined value for multi-byte field
                        if table_col_idx == first_col_for_def.get(col_def, table_col_idx):
                            start_byte = col_def.start_byte + base_byte_idx
                            end_byte = col_def.end_byte + base_byte_idx
                            if start_byte < num_bytes:
                                slice_end = min(end_byte + 1, num_bytes)
                                byte_slice = data[start_byte:slice_end]
                                text = self._format_multi_byte_value(byte_slice, col_def)
                                tooltip = "{} ({})".format(
                                    col_def.label,
                                    col_def.display_format
                                ) if col_def.label else ""

                elif is_undef:
                    abs_start_bit = absolute_byte_idx * 8 + bit_start
                    num_bits = bit_end - bit_start + 1
                    byte_slice = data[absolute_byte_idx:absolute_byte_idx + 1]
                    text = self._format_undefined_bits(byte_slice, abs_start_bit, num_bits)
                    tooltip = f"{num_bits} undefined bits (bit {abs_start_bit}-{abs_start_bit + num_bits - 1})"

                else:
                    text = "{:02X}".format(byte_val)

                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if tooltip and text:
                    item.setToolTip(tooltip)

                # Check if this cell should be pattern-highlighted
                is_pattern_highlighted = self._is_byte_in_pattern_highlight(absolute_byte_idx)

                if is_pattern_highlighted and text != "":
                    # Pattern highlight takes priority (bright yellow/orange)
                    item.setBackground(QColor(255, 200, 0))  # Orange highlight
                elif table_col_idx in self.constant_columns and text != "":
                    item.setBackground(QColor(255, 255, 180))
                else:
                    if col_def is not None:
                        col_color = self._color_from_name(col_def.color_name)
                        if col_color is not None and text != "":
                            item.setBackground(col_color)

                self.setItem(table_row, table_col_idx, item)

        self.setVerticalHeaderLabels(row_labels)
        self.resizeColumnsToContents()
        self.setUpdatesEnabled(True)  # Re-enable updates before returning

        return

        # ==== CONTINUOUS MODE ====
        data = self.bytes_data
        num_bytes = len(data)

        if num_bytes == 0:
            self.setRowCount(0)
            self.setColumnCount(0)
            return

        # Build the column structure for one row
        col_map, label_spans, byte_labels, subbit_labels, all_columns_info = self._build_headers(self.row_size)

        # Decide which table column "owns" the value for each definition
        first_col_for_def = {}
        for idx, (_, _, _, col_def, is_undef) in enumerate(all_columns_info):
            if col_def is not None and col_def not in first_col_for_def:
                first_col_for_def[col_def] = idx

        # Compute contiguous column spans for each definition (bit OR byte)
        field_spans = []
        current_def = None
        current_start = None
        current_len = 0

        for idx, (_, _, _, col_def, is_undef) in enumerate(all_columns_info):
            if col_def is not None:
                if current_def is col_def:
                    current_len += 1
                else:
                    if current_def is not None:
                        field_spans.append((current_start, current_len, current_def))
                    current_def = col_def
                    current_start = idx
                    current_len = 1
            else:
                if current_def is not None:
                    field_spans.append((current_start, current_len, current_def))
                    current_def = None
                    current_start = None
                    current_len = 0

        if current_def is not None:
            field_spans.append((current_start, current_len, current_def))

        # Set up table with split columns
        num_table_cols = len(all_columns_info)
        self.setColumnCount(num_table_cols)

        num_data_rows = (num_bytes + self.row_size - 1) // self.row_size
        # 3 header rows + data rows
        self.setRowCount(num_data_rows + 3)

        # === Header row 0: labels ===
        for col_idx in range(num_table_cols):
            item = QTableWidgetItem("")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setBackground(QColor(220, 220, 220))
            self.setItem(0, col_idx, item)

        for table_col, span_width, label_text, color_name in label_spans:
            item = QTableWidgetItem(label_text)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

            if label_text == "?":
                item.setBackground(QColor(200, 200, 200))
                item.setFont(QFont("Arial", 9, QFont.Weight.Normal))
            else:
                if color_name and color_name != "None":
                    bg_color = self._color_from_name(color_name)
                    if bg_color:
                        item.setBackground(bg_color)
                    else:
                        item.setBackground(QColor(200, 220, 255))
                else:
                    item.setBackground(QColor(200, 220, 255))
                item.setFont(QFont("Arial", 9, QFont.Weight.Bold))

            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(0, table_col, item)

            if span_width > 1:
                # "test" across 1–5 becomes one merged header cell
                self.setSpan(0, table_col, 1, span_width)

        # === Header row 1: byte numbers ===
        for col_idx, byte_label in enumerate(byte_labels):
            item = QTableWidgetItem(byte_label)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setBackground(QColor(240, 240, 240))
            item.setFont(QFont("Courier", 8))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(1, col_idx, item)

        # === Header row 2: bit ranges ===
        for col_idx, sub_label in enumerate(subbit_labels):
            item = QTableWidgetItem(sub_label)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setBackground(QColor(250, 250, 250))
            item.setFont(QFont("Courier", 8))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(2, col_idx, item)

        # === Data rows ===
        row_labels = ["Labels", "Bytes", "Bits"]

        # OPTIMIZED: Show warning if rows were truncated
        if num_data_rows < total_data_rows:
            row_labels[0] += f" (showing {num_data_rows}/{total_data_rows})"

        for data_row_idx in range(num_data_rows):
            table_row = data_row_idx + 3
            row_labels.append(str(data_row_idx))
            base_byte_idx = data_row_idx * self.row_size

            # Merge columns for the same field so one wide cell shows the value
            for start_col, span_len, col_def in field_spans:
                if span_len > 1:
                    self.setSpan(table_row, start_col, 1, span_len)

            for table_col_idx, (byte_offset, bit_start, bit_end, col_def, is_undef) in enumerate(all_columns_info):
                absolute_byte_idx = base_byte_idx + byte_offset

                if absolute_byte_idx >= num_bytes:
                    item = QTableWidgetItem("")
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.setItem(table_row, table_col_idx, item)
                    continue

                byte_val = int(data[absolute_byte_idx])
                text = ""
                tooltip = ""

                if col_def is not None:
                    if col_def.unit == "bit":
                        # Single combined value for the whole bit field
                        if table_col_idx == first_col_for_def.get(col_def, table_col_idx):
                            start_byte = col_def.start_byte + base_byte_idx
                            end_byte = col_def.end_byte + base_byte_idx
                            if start_byte < num_bytes:
                                slice_end = min(end_byte + 1, num_bytes)
                                byte_slice = data[start_byte:slice_end]
                                text = self._format_bit_range(byte_slice, col_def)
                                tooltip = "{} (bit {}-{}, {} bits, {})".format(
                                    col_def.label,
                                    col_def.start_bit,
                                    col_def.start_bit + col_def.total_bits - 1,
                                    col_def.total_bits,
                                    col_def.display_format
                                )
                    elif col_def.unit == "byte":
                        # Single combined value for multi-byte field
                        if table_col_idx == first_col_for_def.get(col_def, table_col_idx):
                            start_byte = col_def.start_byte + base_byte_idx
                            end_byte = col_def.end_byte + base_byte_idx
                            if start_byte < num_bytes:
                                slice_end = min(end_byte + 1, num_bytes)
                                byte_slice = data[start_byte:slice_end]
                                text = self._format_multi_byte_value(byte_slice, col_def)
                                tooltip = "{} ({})".format(
                                    col_def.label,
                                    col_def.display_format
                                ) if col_def.label else ""

                elif is_undef:
                    abs_start_bit = absolute_byte_idx * 8 + bit_start
                    num_bits = bit_end - bit_start + 1
                    byte_slice = data[absolute_byte_idx:absolute_byte_idx + 1]
                    text = self._format_undefined_bits(byte_slice, abs_start_bit, num_bits)
                    tooltip = f"{num_bits} undefined bits (bit {abs_start_bit}-{abs_start_bit + num_bits - 1})"

                else:
                    text = "{:02X}".format(byte_val)

                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if tooltip and text:
                    item.setToolTip(tooltip)

                # Check if this cell should be pattern-highlighted
                is_pattern_highlighted = self._is_byte_in_pattern_highlight(absolute_byte_idx)

                if is_pattern_highlighted and text != "":
                    # Pattern highlight takes priority (bright yellow/orange)
                    item.setBackground(QColor(255, 200, 0))  # Orange highlight
                elif table_col_idx in self.constant_columns and text != "":
                    item.setBackground(QColor(255, 255, 180))
                else:
                    if col_def is not None:
                        col_color = self._color_from_name(col_def.color_name)
                        if col_color is not None and text != "":
                            item.setBackground(col_color)

                self.setItem(table_row, table_col_idx, item)

        self.setVerticalHeaderLabels(row_labels)
        self.resizeColumnsToContents()
        self.setUpdatesEnabled(True)  # Re-enable updates before returning

        return

        # ==== CONTINUOUS MODE ====
        data = self.bytes_data
        num_bytes = len(data)

        if num_bytes == 0:
            self.setRowCount(0)
            self.setColumnCount(0)
            return

        # Build the column structure for one row
        col_map, label_spans, byte_labels, subbit_labels, all_columns_info = self._build_headers(self.row_size)
        # Determine which table column should display each bit-field value
        first_col_for_def = {}
        for idx, (_, _, _, col_def, is_undef) in enumerate(all_columns_info):
            if col_def is not None and col_def.unit == "bit":
                if col_def not in first_col_for_def:
                    first_col_for_def[col_def] = idx

        # Compute contiguous column spans for each bit-field
        field_spans = []
        current_def = None
        current_start = None
        current_len = 0

        for idx, (_, _, _, col_def, is_undef) in enumerate(all_columns_info):
            if col_def is not None and col_def.unit == "bit":
                if current_def is col_def:
                    current_len += 1
                else:
                    if current_def is not None:
                        field_spans.append((current_start, current_len, current_def))
                    current_def = col_def
                    current_start = idx
                    current_len = 1
            else:
                if current_def is not None:
                    field_spans.append((current_start, current_len, current_def))
                    current_def = None
                    current_start = None
                    current_len = 0

        if current_def is not None:
            field_spans.append((current_start, current_len, current_def))

        # Determine which table column should display each bit-field value
        first_col_for_def = {}
        for idx, (_, _, _, col_def, is_undef) in enumerate(all_columns_info):
            if col_def is not None and col_def.unit == "bit":
                if col_def not in first_col_for_def:
                    first_col_for_def[col_def] = idx

        # Set up table with split columns
        num_table_cols = len(all_columns_info)
        self.setColumnCount(num_table_cols)

        num_data_rows = (num_bytes + self.row_size - 1) // self.row_size
        # 3 header rows + data rows
        self.setRowCount(num_data_rows + 3)

        # First header row: Column definition labels
        for col_idx in range(num_table_cols):
            item = QTableWidgetItem("")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setBackground(QColor(220, 220, 220))
            self.setItem(0, col_idx, item)

        # Apply labels
        for table_col, span_width, label_text, color_name in label_spans:
            item = QTableWidgetItem(label_text)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

            # Apply color
            if label_text == "?":
                # Gray background for undefined bits
                item.setBackground(QColor(200, 200, 200))
                item.setFont(QFont("Arial", 9, QFont.Weight.Normal))
            else:
                # Apply definition color
                if color_name and color_name != "None":
                    bg_color = self._color_from_name(color_name)
                    if bg_color:
                        item.setBackground(bg_color)
                    else:
                        item.setBackground(QColor(200, 220, 255))
                else:
                    item.setBackground(QColor(200, 220, 255))
                item.setFont(QFont("Arial", 9, QFont.Weight.Bold))

            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(0, table_col, item)

            # Span across all columns for this label if needed
            if span_width > 1:
                self.setSpan(0, table_col, 1, span_width)

        # Second header row: Byte numbers (row 1)
        for col_idx, byte_label in enumerate(byte_labels):
            item = QTableWidgetItem(byte_label)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setBackground(QColor(240, 240, 240))
            item.setFont(QFont("Courier", 8))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(1, col_idx, item)

        # Third header row: sub-byte bit ranges (row 2)
        for col_idx, sub_label in enumerate(subbit_labels):
            item = QTableWidgetItem(sub_label)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setBackground(QColor(250, 250, 250))
            item.setFont(QFont("Courier", 8))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(2, col_idx, item)

        # Data rows
        row_labels = ["Labels", "Bytes", "Bits"]
        for data_row_idx in range(num_data_rows):
            table_row = data_row_idx + 3
            row_labels.append(str(data_row_idx))
            base_byte_idx = data_row_idx * self.row_size

            # Merge columns that belong to the same bit-field so the value spans them
            for start_col, span_len, col_def in field_spans:
                if span_len > 1:
                    self.setSpan(table_row, start_col, 1, span_len)

            for table_col_idx, (byte_offset, bit_start, bit_end, col_def, is_undef) in enumerate(all_columns_info):
                absolute_byte_idx = base_byte_idx + byte_offset

                if absolute_byte_idx >= num_bytes:
                    text = ""
                    tooltip = ""
                    item = QTableWidgetItem(text)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.setItem(table_row, table_col_idx, item)
                    continue

                byte_val = int(data[absolute_byte_idx])

                if col_def is not None:
                    # This is a defined field
                    if col_def.unit == "bit":
                        # Only show the combined value in the first table column for this field
                        if table_col_idx != first_col_for_def.get(col_def, table_col_idx):
                            text = ""
                            tooltip = ""
                        else:
                            # Extract the specific bits for this column (full field value)
                            abs_start_bit = absolute_byte_idx * 8 + bit_start
                            num_bits = bit_end - bit_start + 1

                            # Get the bytes we need (might span multiple rows)
                            start_byte = col_def.start_byte + base_byte_idx
                            end_byte = col_def.end_byte + base_byte_idx
                            if start_byte >= num_bytes:
                                text = ""
                                tooltip = ""
                            else:
                                slice_end = min(end_byte + 1, num_bytes)
                                byte_slice = data[start_byte:slice_end]
                                text = self._format_bit_range(byte_slice, col_def)
                                tooltip = "{} (bit {}-{}, {} bits, {})".format(
                                    col_def.label,
                                    col_def.start_bit,
                                    col_def.start_bit + col_def.total_bits - 1,
                                    col_def.total_bits,
                                    col_def.display_format
                                )

                    else:
                        # Byte-level field
                        start_byte = col_def.start_byte + base_byte_idx
                        end_byte = col_def.end_byte + base_byte_idx
                        if start_byte >= num_bytes:
                            text = ""
                            tooltip = ""
                        else:
                            slice_end = min(end_byte + 1, num_bytes)
                            byte_slice = data[start_byte:slice_end]
                            text = self._format_multi_byte_value(byte_slice, col_def)
                            tooltip = "{} ({})".format(col_def.label,
                                                       col_def.display_format) if col_def.label else ""

                elif is_undef:
                    # Undefined bits - show in binary
                    abs_start_bit = absolute_byte_idx * 8 + bit_start
                    num_bits = bit_end - bit_start + 1
                    byte_slice = data[absolute_byte_idx:absolute_byte_idx + 1]
                    text = self._format_undefined_bits(byte_slice, abs_start_bit, num_bits)
                    tooltip = f"{num_bits} undefined bits (bit {abs_start_bit}-{abs_start_bit + num_bits - 1})"

                else:
                    # Default hex display
                    text = "{:02X}".format(byte_val)
                    tooltip = ""

                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if tooltip and text:
                    item.setToolTip(tooltip)

                # Apply coloring
                # Check if this cell should be pattern-highlighted
                is_pattern_highlighted = self._is_byte_in_pattern_highlight(absolute_byte_idx)

                if is_pattern_highlighted and text != "":
                    # Pattern highlight takes priority (bright yellow/orange)
                    item.setBackground(QColor(255, 200, 0))  # Orange highlight
                elif table_col_idx in self.constant_columns and text != "":
                    item.setBackground(QColor(255, 255, 180))
                else:
                    if col_def is not None:
                        col_color = self._color_from_name(col_def.color_name)
                        if col_color is not None and text != "":
                            item.setBackground(col_color)

                self.setItem(table_row, table_col_idx, item)

        self.setVerticalHeaderLabels(row_labels)
        self.resizeColumnsToContents()


class AddColumnDialog(QDialog):
    """Dialog for adding a column definition with flexible bit/byte selection"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Column Definition")
        self.setModal(True)
        self.resize(450, 400)

        layout = QVBoxLayout(self)

        # Label
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("Label:"))
        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("e.g., Sync, Timestamp, Data")
        label_layout.addWidget(self.label_input)
        layout.addLayout(label_layout)

        # Unit: bytes vs bits
        unit_group_box = QGroupBox("Definition Unit")
        unit_layout = QVBoxLayout(unit_group_box)

        self.byte_radio = QRadioButton("Bytes")
        self.bit_radio = QRadioButton("Bits")
        self.byte_radio.setChecked(True)

        unit_layout.addWidget(self.byte_radio)
        unit_layout.addWidget(self.bit_radio)

        layout.addWidget(unit_group_box)

        # === BYTE MODE CONTROLS ===
        self.byte_controls = QWidget()
        byte_layout = QVBoxLayout(self.byte_controls)
        byte_layout.setContentsMargins(0, 0, 0, 0)

        # Start byte
        start_byte_layout = QHBoxLayout()
        start_byte_layout.addWidget(QLabel("Start Byte:"))
        self.start_byte_spin = QSpinBox()
        self.start_byte_spin.setRange(0, 1000)
        self.start_byte_spin.setValue(0)
        start_byte_layout.addWidget(self.start_byte_spin)
        byte_layout.addLayout(start_byte_layout)

        # End byte
        end_byte_layout = QHBoxLayout()
        end_byte_layout.addWidget(QLabel("End Byte:"))
        self.end_byte_spin = QSpinBox()
        self.end_byte_spin.setRange(0, 1000)
        self.end_byte_spin.setValue(0)
        end_byte_layout.addWidget(self.end_byte_spin)
        byte_layout.addLayout(end_byte_layout)

        layout.addWidget(self.byte_controls)

        # === BIT MODE CONTROLS ===
        self.bit_controls = QWidget()
        bit_layout = QVBoxLayout(self.bit_controls)
        bit_layout.setContentsMargins(0, 0, 0, 0)

        # Bit position input method
        bit_input_group = QGroupBox("Bit Position Input")
        bit_input_layout = QVBoxLayout(bit_input_group)

        self.bit_input_button_group = QButtonGroup()
        self.bit_pos_radio = QRadioButton("Bit Position (absolute)")
        self.byte_pos_radio = QRadioButton("Byte Position (byte + offset)")
        self.bit_pos_radio.setChecked(True)

        self.bit_input_button_group.addButton(self.bit_pos_radio)
        self.bit_input_button_group.addButton(self.byte_pos_radio)

        bit_input_layout.addWidget(self.bit_pos_radio)
        bit_input_layout.addWidget(self.byte_pos_radio)
        bit_layout.addWidget(bit_input_group)

        # Absolute bit position input
        self.absolute_bit_widget = QWidget()
        abs_bit_layout = QVBoxLayout(self.absolute_bit_widget)
        abs_bit_layout.setContentsMargins(0, 0, 0, 0)

        abs_pos_layout = QHBoxLayout()
        abs_pos_layout.addWidget(QLabel("Starting Bit Position:"))
        self.abs_bit_position_spin = QSpinBox()
        self.abs_bit_position_spin.setRange(0, 10000)
        self.abs_bit_position_spin.setValue(0)
        abs_pos_layout.addWidget(self.abs_bit_position_spin)
        abs_bit_layout.addLayout(abs_pos_layout)

        bit_layout.addWidget(self.absolute_bit_widget)

        # Byte + offset position input
        self.byte_offset_widget = QWidget()
        byte_off_layout = QVBoxLayout(self.byte_offset_widget)
        byte_off_layout.setContentsMargins(0, 0, 0, 0)

        byte_pos_layout = QHBoxLayout()
        byte_pos_layout.addWidget(QLabel("Starting Byte:"))
        self.bit_start_byte_spin = QSpinBox()
        self.bit_start_byte_spin.setRange(0, 1000)
        self.bit_start_byte_spin.setValue(0)
        byte_pos_layout.addWidget(self.bit_start_byte_spin)
        byte_off_layout.addLayout(byte_pos_layout)

        bit_offset_layout = QHBoxLayout()
        bit_offset_layout.addWidget(QLabel("Bit Offset in Byte:"))
        self.bit_offset_spin = QSpinBox()
        self.bit_offset_spin.setRange(0, 7)
        self.bit_offset_spin.setValue(0)
        bit_offset_layout.addWidget(self.bit_offset_spin)
        byte_off_layout.addLayout(bit_offset_layout)

        bit_layout.addWidget(self.byte_offset_widget)
        self.byte_offset_widget.hide()

        # Total bits
        total_bits_layout = QHBoxLayout()
        total_bits_layout.addWidget(QLabel("Total Bits:"))
        self.total_bits_spin = QSpinBox()
        self.total_bits_spin.setRange(1, 128)
        self.total_bits_spin.setValue(8)
        total_bits_layout.addWidget(self.total_bits_spin)
        bit_layout.addLayout(total_bits_layout)

        # Example label
        self.bit_example_label = QLabel()
        self.bit_example_label.setWordWrap(True)
        self.bit_example_label.setStyleSheet("color: #666; font-style: italic;")
        bit_layout.addWidget(self.bit_example_label)

        layout.addWidget(self.bit_controls)
        self.bit_controls.hide()

        # Display format
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Display Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems([
            "Hex (MSBF)",  # hex_be
            "Hex (LSBF)",  # hex_le
            "Binary",  # binary (no endianness)
            "Decimal (MSBF)",  # dec_be
            "Decimal (LSBF)",  # dec_le
            "Twos complement (MSBF)",  # tc_be
            "Twos complement (LSBF)",  # tc_le
            "ASCII (MSBF)",  # ascii_be
            "ASCII (LSBF)",  # ascii_le
        ])
        format_layout.addWidget(self.format_combo)
        layout.addLayout(format_layout)

        # Color choice
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Highlight Color:"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(["None", "Yellow", "Cyan", "Magenta", "Green"])
        color_layout.addWidget(self.color_combo)
        layout.addLayout(color_layout)

        # Connect radio buttons
        # Connect radio buttons
        def update_format_enabled():
            """
            Enable / disable formats based on alignment:

            ASCII:
              - Byte mode: always allowed
              - Bit mode: only if start_bit is byte-aligned AND total_bits is multiple of 8

            Hex:
              - Byte mode: always allowed
              - Bit mode: only if start_bit is nibble-aligned AND total_bits is multiple of 4
            """
            # Defaults
            ascii_valid = False
            hex_valid = False

            # Byte mode: everything is nicely aligned
            if self.byte_radio.isChecked():
                ascii_valid = True
                hex_valid = True
            else:
                # Bit mode: compute absolute start_bit
                if self.bit_pos_radio.isChecked():
                    start_bit = self.abs_bit_position_spin.value()
                else:
                    start_byte_val = self.bit_start_byte_spin.value()
                    bit_offset = self.bit_offset_spin.value()
                    start_bit = start_byte_val * 8 + bit_offset

                total_bits = self.total_bits_spin.value()

                # ASCII -> full bytes only
                ascii_valid = ((start_bit % 8) == 0) and ((total_bits % 8) == 0)
                # Hex -> nibble aligned
                hex_valid = ((start_bit % 4) == 0) and ((total_bits % 4) == 0)

            # Enable/disable ASCII entries
            for label in ["ASCII (MSBF)", "ASCII (LSBF)"]:
                idx = self.format_combo.findText(label)
                if idx >= 0:
                    item = self.format_combo.model().item(idx)
                    if item:
                        item.setEnabled(ascii_valid)

            # Enable/disable Hex entries
            for label in ["Hex (MSBF)", "Hex (LSBF)"]:
                idx = self.format_combo.findText(label)
                if idx >= 0:
                    item = self.format_combo.model().item(idx)
                    if item:
                        item.setEnabled(hex_valid)

            # Fix current selection if it became invalid
            current_text = self.format_combo.currentText()

            # If ASCII currently selected but no longer valid
            if current_text.startswith("ASCII") and not ascii_valid:
                # Prefer Hex if that's valid, otherwise fall back to Binary
                fallback = "Hex (MSBF)" if hex_valid else "Binary"
                idx = self.format_combo.findText(fallback)
                if idx >= 0:
                    self.format_combo.setCurrentIndex(idx)

            # If Hex currently selected but no longer valid
            elif current_text.startswith("Hex") and not hex_valid:
                # Prefer ASCII if that's valid, otherwise fall back to Binary
                fallback = "ASCII (MSBF)" if ascii_valid else "Binary"
                idx = self.format_combo.findText(fallback)
                if idx >= 0:
                    self.format_combo.setCurrentIndex(idx)

        def update_mode():
            if self.byte_radio.isChecked():
                self.byte_controls.show()
                self.bit_controls.hide()
            else:
                self.byte_controls.hide()
                self.bit_controls.show()
                update_bit_input_mode()
                update_bit_example()
            update_format_enabled()

        self.byte_radio.toggled.connect(update_mode)
        self.bit_radio.toggled.connect(update_mode)

        # Connect bit input mode toggle
        def update_bit_input_mode():
            if self.bit_pos_radio.isChecked():
                self.absolute_bit_widget.show()
                self.byte_offset_widget.hide()
            else:
                self.absolute_bit_widget.hide()
                self.byte_offset_widget.show()
            update_bit_example()
            update_format_enabled()

        self.bit_pos_radio.toggled.connect(update_bit_input_mode)
        self.byte_pos_radio.toggled.connect(update_bit_input_mode)

        # Update bit example
        def update_bit_example():
            if self.bit_radio.isChecked():
                if self.bit_pos_radio.isChecked():
                    # Absolute bit position
                    start_bit = self.abs_bit_position_spin.value()
                    total_bits = self.total_bits_spin.value()

                    start_byte = start_bit // 8
                    end_bit = start_bit + total_bits - 1
                    end_byte = end_bit // 8

                    if end_byte == start_byte:
                        self.bit_example_label.setText(
                            f"Example: Extracts bits {start_bit}-{end_bit} (all in byte {start_byte})"
                        )
                    else:
                        self.bit_example_label.setText(
                            f"Example: Extracts {total_bits} bits starting at bit {start_bit}\n"
                            f"(spans bytes {start_byte}-{end_byte})"
                        )
                else:
                    # Byte + offset
                    start_byte = self.bit_start_byte_spin.value()
                    bit_offset = self.bit_offset_spin.value()
                    total_bits = self.total_bits_spin.value()

                    start_bit = start_byte * 8 + bit_offset
                    end_bit = start_bit + total_bits - 1
                    end_byte = end_bit // 8

                    if end_byte == start_byte:
                        self.bit_example_label.setText(
                            f"Example: Extracts bits {bit_offset}-{bit_offset + total_bits - 1} from byte {start_byte}\n"
                            f"(absolute bit position {start_bit}-{end_bit})"
                        )
                    else:
                        self.bit_example_label.setText(
                            f"Example: Extracts {total_bits} bits from byte {start_byte}, offset {bit_offset}\n"
                            f"(absolute bit {start_bit}-{end_bit}, spans bytes {start_byte}-{end_byte})"
                        )
            update_format_enabled()

        self.abs_bit_position_spin.valueChanged.connect(update_bit_example)
        self.bit_start_byte_spin.valueChanged.connect(update_bit_example)
        self.bit_offset_spin.valueChanged.connect(update_bit_example)
        self.total_bits_spin.valueChanged.connect(update_bit_example)

        update_mode()

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_column_definition(self):
        """Return the column definition from the dialog"""
        label = self.label_input.text().strip()

        if self.byte_radio.isChecked():
            unit = "byte"
            start_byte = self.start_byte_spin.value()
            end_byte = self.end_byte_spin.value()
            start_bit = 0
            total_bits = 8
        else:
            unit = "bit"

            # Calculate absolute bit position
            if self.bit_pos_radio.isChecked():
                start_bit = self.abs_bit_position_spin.value()
            else:
                start_byte_val = self.bit_start_byte_spin.value()
                bit_offset = self.bit_offset_spin.value()
                start_bit = start_byte_val * 8 + bit_offset

            total_bits = self.total_bits_spin.value()

            # These will be calculated by ColumnDefinition
            start_byte = 0
            end_byte = 0

        fmt_label = self.format_combo.currentText()
        fmt_map = {
            "Hex (MSBF)": "hex_be",
            "Hex (LSBF)": "hex_le",
            "Binary": "binary",
            "Decimal (MSBF)": "dec_be",
            "Decimal (LSBF)": "dec_le",
            "Twos complement (MSBF)": "tc_be",
            "Twos complement (LSBF)": "tc_le",
            "ASCII (MSBF)": "ascii_be",
            "ASCII (LSBF)": "ascii_le",
        }
        display_format = fmt_map.get(fmt_label, "hex_be")

        color_name = self.color_combo.currentText()

        return ColumnDefinition(
            start_byte,
            end_byte,
            label,
            display_format,
            color_name=color_name,
            unit=unit,
            start_bit=start_bit,
            total_bits=total_bits
        )


class BitViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Bit Viewer - Enhanced")
        self.setGeometry(100, 100, 1400, 900)

        # Data
        self.bits = None
        self.bytes_data = None
        self.original_bits = None
        self.original_bytes = None
        self.filename = ""
        self.operations = []  # List of operation text descriptions

        # Mode
        self.data_mode = "bit"  # "bit" or "byte"

        # Undo/Redo stacks
        self.undo_stack = []
        self.redo_stack = []

        self.init_ui()
        self.apply_theme()

    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Mode selector at the top
        mode_widget = QWidget()
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(10, 5, 10, 5)

        mode_label = QLabel("Data Mode:")
        mode_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        mode_layout.addWidget(mode_label)

        self.mode_button_group = QButtonGroup()

        self.bit_mode_radio = QRadioButton("Bit Mode")
        self.bit_mode_radio.setChecked(True)
        self.bit_mode_radio.toggled.connect(self.on_mode_changed)
        self.mode_button_group.addButton(self.bit_mode_radio)
        mode_layout.addWidget(self.bit_mode_radio)

        self.byte_mode_radio = QRadioButton("Byte Mode")
        self.byte_mode_radio.toggled.connect(self.on_mode_changed)
        self.mode_button_group.addButton(self.byte_mode_radio)
        mode_layout.addWidget(self.byte_mode_radio)

        mode_layout.addStretch()
        main_layout.addWidget(mode_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        main_layout.setStretch(0, 0)  # mode_widget (fixed-ish height)
        main_layout.setStretch(1, 1)  # splitter (expands)

        # Left panel - Controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel - Display
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([400, 1000])

    def on_mode_changed(self):
        """Handle mode change between Bit and Byte."""
        previous_mode = self.data_mode
        new_mode = "bit" if self.bit_mode_radio.isChecked() else "byte"

        if new_mode == previous_mode:
            return

        # ===== BIT -> BYTE =====
        if previous_mode == "bit" and new_mode == "byte":
            if self.bits is None:
                self.data_mode = new_mode
                self.update_left_panel_visibility()
                self.update_display()
                self.apply_theme()
                return

            # Get frame width from bit mode (width spinner value)
            frame_width_bits = self.width_spin.value()
            total_bits = len(self.bits)

            # Calculate how many complete frames we have at this width
            num_complete_frames = total_bits // frame_width_bits
            remaining_bits = total_bits % frame_width_bits

            # Calculate padding needed to make frame width divisible by 8
            frame_padding_needed = (8 - (frame_width_bits % 8)) % 8

            # Ask user about zero-padding if frame width isn't divisible by 8
            use_frame_padding = False
            if frame_padding_needed > 0:
                padded_frame_width = frame_width_bits + frame_padding_needed

                reply = QMessageBox.question(
                    self,
                    "Zero-Pad to Byte Boundary?",
                    f"Current frame width: {frame_width_bits:,} bits\n\n"
                    f"This is not divisible by 8 (not a whole number of bytes).\n\n"
                    f"Zero-pad each frame by {frame_padding_needed} bit(s)?\n\n"
                    f"Yes = Pad frames to {padded_frame_width:,} bits and preserve frame alignment\n"
                    f"No = Keep all {total_bits:,} bits as-is (frame alignment will be lost in byte mode)",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                use_frame_padding = (reply == QMessageBox.StandardButton.Yes)

            # Process bits based on user choice
            if use_frame_padding and frame_padding_needed > 0:
                # Pad each frame to make it divisible by 8
                padded_frame_width = frame_width_bits + frame_padding_needed
                result_bits = []

                for i in range(num_complete_frames):
                    frame_start = i * frame_width_bits
                    frame_end = frame_start + frame_width_bits
                    frame = self.bits[frame_start:frame_end]
                    # Add padding to this frame
                    padded_frame = np.append(frame, np.zeros(frame_padding_needed, dtype=np.uint8))
                    result_bits.append(padded_frame)

                # Handle remaining bits (if any)
                if remaining_bits > 0:
                    last_frame = self.bits[num_complete_frames * frame_width_bits:]
                    # Pad to nearest byte
                    last_padding = (8 - (len(last_frame) % 8)) % 8
                    if last_padding > 0:
                        last_frame = np.append(last_frame, np.zeros(last_padding, dtype=np.uint8))
                    result_bits.append(last_frame)

                bits_to_save = np.concatenate(result_bits) if result_bits else self.bits.copy()
            else:
                # Keep ALL bits - just pad the very end to make it divisible by 8 for packing
                bits_to_save = self.bits.copy()
                # Only pad the final partial byte if needed
                final_padding = (8 - (len(bits_to_save) % 8)) % 8
                if final_padding > 0:
                    bits_to_save = np.append(bits_to_save, np.zeros(final_padding, dtype=np.uint8))

            # Convert to bytes
            self.bytes_data = np.packbits(bits_to_save)
            self.original_bytes = self.bytes_data.copy()

            # Set byte table row size based on bit mode frame width
            if use_frame_padding and frame_padding_needed > 0:
                # User chose to pad - use padded frame width
                bytes_per_row = padded_frame_width // 8
            elif frame_padding_needed == 0:
                # Frame width was already divisible by 8
                bytes_per_row = frame_width_bits // 8
            else:
                # User chose not to pad - use original frame width rounded down
                bytes_per_row = frame_width_bits // 8

            # Update byte table row size AND the spinner in the UI
            self.byte_table.set_row_size(bytes_per_row)
            self.row_size_spin.blockSignals(True)
            self.row_size_spin.setValue(bytes_per_row)
            self.row_size_spin.blockSignals(False)

            # Clear bit operations when switching to byte mode
            if self.operations:
                self.operations = []
                self.operations_list.clear()
                self.undo_stack.clear()
                self.redo_stack.clear()
                self.clear_highlights()

        # BYTE -> BIT
        elif previous_mode == "byte" and new_mode == "bit":
            if self.bytes_data is not None:
                # If framed, ask user if they want to zero-pad
                if self.byte_table.frames is not None and len(self.byte_table.frames) > 0:
                    frames = self.byte_table.frames
                    max_len = max(len(f) for f in frames)
                    min_len = min(len(f) for f in frames)

                    # Only ask if frames have different lengths
                    if min_len < max_len:
                        reply = QMessageBox.question(
                            self,
                            "Zero-Pad Frames?",
                            f"You have {len(frames)} frames with varying lengths.\n\n"
                            f"Min length: {min_len:,} bytes\n"
                            f"Max length: {max_len:,} bytes\n\n"
                            f"Zero-pad all frames to max length ({max_len:,} bytes)?\n\n"
                            f"Yes = Pad frames and set width to {max_len * 8:,} bits\n"
                            f"No = Use original data without padding",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                        )
                        use_padding = (reply == QMessageBox.StandardButton.Yes)
                    else:
                        # All frames same length, no need to ask
                        use_padding = True

                    if use_padding:
                        # Set bit-mode width so one frame = one row (max_len bytes * 8 bits)
                        frame_width_bits = max_len * 8
                        max_spin = self.width_spin.maximum()
                        if frame_width_bits > max_spin:
                            self.width_spin.setMaximum(frame_width_bits)
                        self.width_spin.blockSignals(True)
                        self.width_spin.setValue(frame_width_bits)
                        self.width_spin.blockSignals(False)

                        # Zero-pad each frame to max_len
                        padded_frames = []
                        for f in frames:
                            if len(f) < max_len:
                                pad = np.zeros(max_len - len(f), dtype=np.uint8)
                                padded_f = np.concatenate([f, pad])
                            else:
                                padded_f = f
                            padded_frames.append(padded_f)

                        # Concatenate all padded frames into one byte array for bit mode
                        work_bytes = np.concatenate(padded_frames)
                    else:
                        # No padding - just concatenate frames as-is
                        work_bytes = np.concatenate(frames)
                else:
                    # No framing: just use the raw bytes
                    work_bytes = self.bytes_data

                # Unpack to bits for Bit mode
                self.bits = np.unpackbits(work_bytes)
                self.original_bits = self.bits.copy()

                # Reset bit-mode operations & highlights
                self.operations = []
                self.operations_list.clear()
                self.undo_stack.clear()
                self.redo_stack.clear()
                self.clear_highlights()
                # We do NOT touch self.byte_table.frames or frame_pattern here,
                # so the frame info is still there when you go back to Byte mode.

        self.data_mode = new_mode
        self.update_left_panel_visibility()
        self.update_display()
        self.apply_theme()

    def apply_theme(self):
        """Apply color theme based on mode"""
        if self.data_mode == "bit":
            self.setStyleSheet("""
                QMainWindow { background-color: #e6f2ff; }
                QGroupBox {
                    background-color: #cce5ff;
                    border: 2px solid #3399ff;
                    border-radius: 5px;
                    margin-top: 10px;
                    font-weight: bold;
                }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
                QPushButton { background-color: #3399ff; color: white; border: none; padding: 5px; border-radius: 3px; }
                QPushButton:hover { background-color: #66b3ff; }
                QRadioButton { font-size: 11pt; }
                QRadioButton::indicator:checked { background-color: #3399ff; border: 2px solid #0066cc; }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow { background-color: #e6ffe6; }
                QGroupBox {
                    background-color: #ccffcc;
                    border: 2px solid #33cc33;
                    border-radius: 5px;
                    margin-top: 10px;
                    font-weight: bold;
                }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
                QPushButton { background-color: #33cc33; color: white; border: none; padding: 5px; border-radius: 3px; }
                QPushButton:hover { background-color: #66ff66; }
                QRadioButton { font-size: 11pt; }
                QRadioButton::indicator:checked { background-color: #33cc33; border: 2px solid #009900; }
            """)

    def create_left_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # File controls
        file_group = QGroupBox("File")
        file_layout = QVBoxLayout()

        btn_open = QPushButton("📂 Open File")
        btn_open.clicked.connect(self.open_file)
        file_layout.addWidget(btn_open)

        btn_save = QPushButton("💾 Save Processed")
        btn_save.clicked.connect(self.save_file)
        file_layout.addWidget(btn_save)

        self.file_info_label = QLabel("No file loaded")
        self.file_info_label.setWordWrap(True)
        file_layout.addWidget(self.file_info_label)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # === BIT MODE CONTROLS ===
        self.bit_mode_widget = QWidget()
        bit_mode_layout = QVBoxLayout(self.bit_mode_widget)
        bit_mode_layout.setContentsMargins(0, 0, 0, 0)

        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout()

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Squares", "Circles", "Binary", "Hex"])
        self.mode_combo.currentTextChanged.connect(self.update_display)
        mode_layout.addWidget(self.mode_combo)
        display_layout.addLayout(mode_layout)

        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("Width:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(8, 1_000_000)
        self.width_spin.setValue(64)
        self.width_spin.valueChanged.connect(self.update_display)
        width_layout.addWidget(self.width_spin)
        display_layout.addLayout(width_layout)

        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        self.size_spin = QSpinBox()
        self.size_spin.setRange(4, 30)
        self.size_spin.setValue(10)
        self.size_spin.valueChanged.connect(self.update_display)
        size_layout.addWidget(self.size_spin)
        display_layout.addLayout(size_layout)

        display_group.setLayout(display_layout)
        bit_mode_layout.addWidget(display_group)

        pattern_group = QGroupBox("Pattern Search")
        pattern_layout = QVBoxLayout()

        self.pattern_input = QLineEdit()
        self.pattern_input.setPlaceholderText("0x1ACFFC1D or 11010...")
        pattern_layout.addWidget(self.pattern_input)

        self.error_tolerance_spin = QSpinBox()
        self.error_tolerance_spin.setRange(0, 100)
        self.error_tolerance_spin.setPrefix("Error ±")
        self.error_tolerance_spin.setSuffix("%")
        pattern_layout.addWidget(self.error_tolerance_spin)

        btn_highlight = QPushButton("🔍 Highlight Pattern")
        btn_highlight.clicked.connect(self.highlight_pattern)
        pattern_layout.addWidget(btn_highlight)

        btn_clear = QPushButton("Clear Highlights")
        btn_clear.clicked.connect(self.clear_highlights)
        pattern_layout.addWidget(btn_clear)

        self.pattern_results_label = QLabel("")
        pattern_layout.addWidget(self.pattern_results_label)

        pattern_group.setLayout(pattern_layout)
        bit_mode_layout.addWidget(pattern_group)

        bit_ops_group = QGroupBox("Bit Operations")
        bit_ops_layout = QVBoxLayout()

        btn_takeskip = QPushButton("Take/Skip")
        btn_takeskip.clicked.connect(self.apply_takeskip)
        bit_ops_layout.addWidget(btn_takeskip)

        btn_delta = QPushButton("Delta")
        btn_delta.clicked.connect(self.apply_delta)
        bit_ops_layout.addWidget(btn_delta)

        btn_xor = QPushButton("XOR Pattern")
        btn_xor.clicked.connect(self.apply_xor)
        bit_ops_layout.addWidget(btn_xor)

        btn_invert = QPushButton("Invert All")
        btn_invert.clicked.connect(self.apply_invert)
        bit_ops_layout.addWidget(btn_invert)

        bit_ops_group.setLayout(bit_ops_layout)
        bit_mode_layout.addWidget(bit_ops_group)

        layout.addWidget(self.bit_mode_widget)

        # === BYTE MODE CONTROLS ===
        self.byte_mode_widget = QWidget()
        byte_mode_layout = QVBoxLayout(self.byte_mode_widget)
        byte_mode_layout.setContentsMargins(0, 0, 0, 0)

        row_size_group = QGroupBox("Row Settings")
        row_size_layout = QVBoxLayout()

        row_layout = QHBoxLayout()
        row_layout.addWidget(QLabel("Bytes per row:"))
        self.row_size_spin = QSpinBox()
        self.row_size_spin.setRange(1, 1_000_000)
        self.row_size_spin.setValue(16)
        self.row_size_spin.valueChanged.connect(self.update_row_size)
        row_layout.addWidget(self.row_size_spin)
        row_size_layout.addLayout(row_layout)

        row_size_group.setLayout(row_size_layout)
        byte_mode_layout.addWidget(row_size_group)

        columns_group = QGroupBox("Column Definitions")
        columns_layout = QVBoxLayout()

        self.columns_list = QListWidget()
        self.columns_list.itemDoubleClicked.connect(self.edit_column_definition)
        columns_layout.addWidget(self.columns_list)

        btn_add_column = QPushButton("➕ Add Column")
        btn_add_column.clicked.connect(self.add_column_definition)
        columns_layout.addWidget(btn_add_column)

        btn_remove_column = QPushButton("➖ Remove Selected")
        btn_remove_column.clicked.connect(self.remove_column_definition)
        columns_layout.addWidget(btn_remove_column)

        btn_clear_columns = QPushButton("🗑 Clear All")
        btn_clear_columns.clicked.connect(self.clear_column_definitions)
        columns_layout.addWidget(btn_clear_columns)

        columns_group.setLayout(columns_layout)
        byte_mode_layout.addWidget(columns_group)

        framing_group = QGroupBox("Framing")
        framing_layout = QVBoxLayout()

        btn_frame_sync = QPushButton("Frame Sync")
        btn_frame_sync.clicked.connect(self.frame_sync)
        framing_layout.addWidget(btn_frame_sync)

        btn_clear_frames = QPushButton("Clear Frames")
        btn_clear_frames.clicked.connect(self.clear_frame_sync)
        framing_layout.addWidget(btn_clear_frames)

        btn_const_cols = QPushButton("Highlight Constant Columns")
        btn_const_cols.clicked.connect(self.highlight_constant_columns)
        framing_layout.addWidget(btn_const_cols)

        btn_clear_const = QPushButton("Clear Constant Highlights")
        btn_clear_const.clicked.connect(self.clear_constant_highlights)
        framing_layout.addWidget(btn_clear_const)

        framing_group.setLayout(framing_layout)
        byte_mode_layout.addWidget(framing_group)

        layout.addWidget(self.byte_mode_widget)
        self.byte_mode_widget.hide()

        # History
        history_group = QGroupBox("Applied Operations")
        history_layout = QVBoxLayout()

        self.operations_list = QListWidget()
        history_layout.addWidget(self.operations_list)

        btn_undo = QPushButton("⬅ Undo")
        btn_undo.clicked.connect(self.undo)
        history_layout.addWidget(btn_undo)

        btn_redo = QPushButton("➡ Redo")
        btn_redo.clicked.connect(self.redo)
        history_layout.addWidget(btn_redo)

        btn_reset = QPushButton("🔄 Reset")
        btn_reset.clicked.connect(self.reset_to_original)
        history_layout.addWidget(btn_reset)

        history_group.setLayout(history_layout)
        layout.addWidget(history_group)

        layout.addStretch()
        return widget

    def update_left_panel_visibility(self):
        """Show/hide panels based on mode"""
        if self.data_mode == "bit":
            self.bit_mode_widget.show()
            self.byte_mode_widget.hide()
        else:
            self.bit_mode_widget.hide()
            self.byte_mode_widget.show()

    def create_right_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        nav_layout = QHBoxLayout()
        btn_up = QPushButton("▲ Up")
        btn_up.clicked.connect(self.scroll_up)
        nav_layout.addWidget(btn_up)

        btn_down = QPushButton("▼ Down")
        btn_down.clicked.connect(self.scroll_down)
        nav_layout.addWidget(btn_down)

        btn_start = QPushButton("◄◄ Start")
        btn_start.clicked.connect(self.go_to_start)
        nav_layout.addWidget(btn_start)

        btn_end = QPushButton("End ►►")
        btn_end.clicked.connect(self.go_to_end)
        nav_layout.addWidget(btn_end)

        nav_layout.addStretch()
        layout.addLayout(nav_layout)

        self.display_container = QWidget()
        self.display_layout = QVBoxLayout(self.display_container)
        self.display_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

        self.canvas = BitCanvas()
        self.scroll_area.setWidget(self.canvas)
        self.display_layout.addWidget(self.scroll_area)

        self.text_display = TextDisplayWidget()
        self.display_layout.addWidget(self.text_display)
        self.text_display.hide()

        self.byte_table = ByteStructuredTableWidget()
        self.byte_table.parent_window = self  # Set reference for header editing
        self.display_layout.addWidget(self.byte_table)
        self.byte_table.hide()

        layout.addWidget(self.display_container)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        return widget

    def update_row_size(self):
        self.byte_table.set_row_size(self.row_size_spin.value())

    def add_column_definition(self):
        dialog = AddColumnDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            col_def = dialog.get_column_definition()
            self.byte_table.add_column_definition(col_def)

            if col_def.unit == "byte":
                desc = f"{col_def.label if col_def.label else '(unnamed)'}: Bytes [{col_def.start_byte}-{col_def.end_byte}] as {col_def.display_format} ({col_def.color_name})"
            else:
                desc = f"{col_def.label if col_def.label else '(unnamed)'}: Bit {col_def.start_bit}+{col_def.total_bits} as {col_def.display_format} ({col_def.color_name})"

            item = QListWidgetItem(desc)

            # Set background color to match the column definition color
            if col_def.color_name != "None":
                color = self.byte_table._color_from_name(col_def.color_name)
                if color:
                    item.setBackground(color)

            self.columns_list.addItem(item)

    def edit_column_definition(self, item):
        """Edit a column definition when double-clicked"""
        row = self.columns_list.row(item)
        if row < 0 or row >= len(self.byte_table.column_definitions):
            return

        # Get the existing definition
        old_def = self.byte_table.column_definitions[row]

        # Create dialog pre-filled with existing values
        dialog = AddColumnDialog(self)
        dialog.setWindowTitle("Edit Column Definition")

        # Pre-fill the dialog with existing values
        dialog.label_input.setText(old_def.label)

        if old_def.unit == "byte":
            dialog.byte_radio.setChecked(True)
            dialog.start_byte_spin.setValue(old_def.start_byte)
            dialog.end_byte_spin.setValue(old_def.end_byte)
        else:
            dialog.bit_radio.setChecked(True)
            # Default to absolute bit position mode
            dialog.bit_pos_radio.setChecked(True)
            dialog.abs_bit_position_spin.setValue(old_def.start_bit)
            dialog.total_bits_spin.setValue(old_def.total_bits)

        # Set format
        format_map = {
            "hex_be": "Hex (MSBF)",
            "hex_le": "Hex (LSBF)",
            "hex": "Hex (MSBF)",  # old saved configs fallback
            "binary": "Binary",
            "dec_be": "Decimal (MSBF)",
            "dec_le": "Decimal (LSBF)",
            "tc_be": "Twos complement (MSBF)",
            "tc_le": "Twos complement (LSBF)",
            "ascii_be": "ASCII (MSBF)",
            "ascii_le": "ASCII (LSBF)",
            "ascii": "ASCII (MSBF)",  # old fallback
        }

        format_text = format_map.get(old_def.display_format, "Hex")
        index = dialog.format_combo.findText(format_text)
        if index >= 0:
            dialog.format_combo.setCurrentIndex(index)

        # Set color
        color_index = dialog.color_combo.findText(old_def.color_name)
        if color_index >= 0:
            dialog.color_combo.setCurrentIndex(color_index)

        # Show dialog
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Remove old definition
            self.byte_table.remove_column_definition(row)
            self.columns_list.takeItem(row)

            # Add new definition at the same position
            col_def = dialog.get_column_definition()
            self.byte_table.column_definitions.insert(row, col_def)

            if col_def.unit == "byte":
                desc = f"{col_def.label if col_def.label else '(unnamed)'}: Bytes [{col_def.start_byte}-{col_def.end_byte}] as {col_def.display_format} ({col_def.color_name})"
            else:
                desc = f"{col_def.label if col_def.label else '(unnamed)'}: Bit {col_def.start_bit}+{col_def.total_bits} as {col_def.display_format} ({col_def.color_name})"

            new_item = QListWidgetItem(desc)

            # Set background color to match the column definition color
            if col_def.color_name != "None":
                color = self.byte_table._color_from_name(col_def.color_name)
                if color:
                    new_item.setBackground(color)

            self.columns_list.insertItem(row, new_item)
            self.byte_table.update_display()

    def remove_column_definition(self):
        current_row = self.columns_list.currentRow()
        if current_row >= 0:
            self.byte_table.remove_column_definition(current_row)
            self.columns_list.takeItem(current_row)

    def clear_column_definitions(self):
        self.byte_table.clear_column_definitions()
        self.columns_list.clear()
    def add_definition_from_undefined(self, start_bit, total_bits):
        """
        Called by ByteStructuredTableWidget when user double-clicks a '?' column.
        Creates a new real bit-field definition pre-filled with the undefined bit range.
        """
        dialog = AddColumnDialog(self)
        dialog.setWindowTitle("Define Field from Undefined Bits")

        # Bit mode
        dialog.bit_radio.setChecked(True)
        dialog.byte_radio.setChecked(False)

        # Absolute bit mode
        dialog.bit_pos_radio.setChecked(True)
        dialog.byte_pos_radio.setChecked(False)

        dialog.abs_bit_position_spin.setValue(start_bit)
        dialog.total_bits_spin.setValue(total_bits)

        # Format: Binary
        idx_fmt = dialog.format_combo.findText("Binary")
        if idx_fmt >= 0:
            dialog.format_combo.setCurrentIndex(idx_fmt)

        # Color: None
        idx_color = dialog.color_combo.findText("None")
        if idx_color >= 0:
            dialog.color_combo.setCurrentIndex(idx_color)

        # Keep label as '?'
        dialog.label_input.setText("?")

        # User accepted
        if dialog.exec() == QDialog.DialogCode.Accepted:
            col_def = dialog.get_column_definition()
            self.byte_table.add_column_definition(col_def)

            # Add into the left-hand list
            if col_def.unit == "byte":
                desc = f"{col_def.label if col_def.label else '(unnamed)'}: Bytes [{col_def.start_byte}-{col_def.end_byte}] as {col_def.display_format} ({col_def.color_name})"
            else:
                desc = f"{col_def.label if col_def.label else '(unnamed)'}: Bit {col_def.start_bit}+{col_def.total_bits} as {col_def.display_format} ({col_def.color_name})"

            item = QListWidgetItem(desc)

            if col_def.color_name != "None":
                color = self.byte_table._color_from_name(col_def.color_name)
                if color:
                    item.setBackground(color)

            self.columns_list.addItem(item)

    def parse_hex_pattern(self, text):
        s = text.strip().lower()
        if not s:
            return None
        if s.startswith("0x"):
            s = s[2:]
        s = s.replace(" ", "")
        if len(s) == 0 or len(s) % 2 != 0:
            return None
        try:
            return [int(s[i:i + 2], 16) for i in range(0, len(s), 2)]
        except ValueError:
            return None

    def frame_sync(self):
        if self.bytes_data is None:
            QMessageBox.information(self, "Frame Sync", "No byte data loaded.")
            return

        # Create custom dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Frame Sync Pattern")
        dialog.setMinimumWidth(450)
        layout = QVBoxLayout(dialog)

        # Input format selection
        input_format_group = QGroupBox("Input Format")
        input_format_layout = QVBoxLayout()
        input_binary_radio = QRadioButton("Binary (1/0 or X/O)")
        input_hex_radio = QRadioButton("Hex (0x414B or 414B)")
        input_hex_radio.setChecked(True)
        input_format_layout.addWidget(input_binary_radio)
        input_format_layout.addWidget(input_hex_radio)
        input_format_group.setLayout(input_format_layout)
        layout.addWidget(input_format_group)

        # Pattern input
        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("Sync Pattern:"))
        pattern_input = QLineEdit()
        pattern_input.setPlaceholderText("e.g. 0x414B or 01000001")
        pattern_layout.addWidget(pattern_input)
        layout.addWidget(QWidget())
        layout.itemAt(layout.count() - 1).widget().setLayout(pattern_layout)

        # Label input
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("Column Label:"))
        label_input = QLineEdit()
        label_input.setText("Sync")  # Pre-filled with "Sync"
        label_input.setPlaceholderText("e.g., Sync, Header")
        label_layout.addWidget(label_input)
        layout.addWidget(QWidget())
        layout.itemAt(layout.count() - 1).widget().setLayout(label_layout)

        # Color choice
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Highlight Color:"))
        color_combo = QComboBox()
        color_combo.addItems(["None", "Yellow", "Cyan", "Magenta", "Green", "lightblue", "lightgray"])
        color_combo.setCurrentText("Cyan")
        color_layout.addWidget(color_combo)
        layout.addWidget(QWidget())
        layout.itemAt(layout.count() - 1).widget().setLayout(color_layout)

        # Display format selection (will be updated based on pattern length)
        display_format_group = QGroupBox("Display Format")
        display_format_layout = QVBoxLayout()
        display_binary_radio = QRadioButton("Binary")
        display_hex_radio = QRadioButton("Hex")
        display_ascii_radio = QRadioButton("ASCII")
        display_binary_radio.setChecked(True)
        display_format_layout.addWidget(display_binary_radio)
        display_format_layout.addWidget(display_hex_radio)
        display_format_layout.addWidget(display_ascii_radio)
        display_format_group.setLayout(display_format_layout)
        layout.addWidget(display_format_group)

        # Info label
        info_label = QLabel("")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Result display (shows find results)
        result_label = QLabel("")
        result_label.setWordWrap(True)
        result_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; }")
        layout.addWidget(result_label)

        # Buttons
        button_box = QHBoxLayout()
        find_button = QPushButton("Find")
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        # OK is always enabled - user doesn't have to Find first
        button_box.addWidget(find_button)
        button_box.addStretch()
        button_box.addWidget(ok_button)
        button_box.addWidget(cancel_button)
        layout.addLayout(button_box)

        # Store found count for later
        found_info = {'count': 0, 'max_len': 0, 'bits': None, 'positions': None}

        def do_find():
            pattern_text = pattern_input.text().strip()
            if not pattern_text:
                result_label.setText("Enter a pattern first")
                return

            # Parse pattern to bits
            bits = None
            if input_binary_radio.isChecked():
                bits_str = pattern_text.upper().replace('X', '1').replace('O', '0')
                if all(c in '10' for c in bits_str):
                    bits = [int(b) for b in bits_str]
            else:
                hex_str = pattern_text.upper()
                if hex_str.startswith('0X'):
                    hex_str = hex_str[2:]
                if all(c in '0123456789ABCDEF' for c in hex_str):
                    bits = []
                    for hex_char in hex_str:
                        val = int(hex_char, 16)
                        bits.extend([int(b) for b in format(val, '04b')])

            if not bits:
                result_label.setText("Invalid pattern")
                return

            # Store for later
            found_info['bits'] = bits

            # Find pattern positions WITHOUT framing (just for preview)
            positions, max_len = self.byte_table.find_pattern_positions(bits)
            found_info['positions'] = positions
            found_info['count'] = len(positions)
            found_info['max_len'] = max_len

            if len(positions) > 0:
                result_label.setText(f"✓ Found {len(positions):,} frames (max length: {max_len:,} bytes)")
                # Highlight the patterns in the display
                self.byte_table.set_pattern_highlights(positions, len(bits))
            else:
                result_label.setText("✗ Pattern not found")
                self.byte_table.clear_pattern_highlights()

        find_button.clicked.connect(do_find)
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        # Function to update display options based on pattern
        def update_display_options():
            pattern_text = pattern_input.text().strip()
            if not pattern_text:
                info_label.setText("")
                return

            # Parse pattern to bits
            bits = None
            if input_binary_radio.isChecked():
                # Binary input: 1/0 or X/O
                bits_str = pattern_text.upper().replace('X', '1').replace('O', '0')
                if all(c in '10' for c in bits_str):
                    bits = [int(b) for b in bits_str]
            else:
                # Hex input
                hex_str = pattern_text.upper()
                if hex_str.startswith('0X'):
                    hex_str = hex_str[2:]
                if all(c in '0123456789ABCDEF' for c in hex_str):
                    bits = []
                    for hex_char in hex_str:
                        val = int(hex_char, 16)
                        bits.extend([int(b) for b in format(val, '04b')])

            if bits is None:
                info_label.setText("Invalid pattern")
                display_hex_radio.setEnabled(False)
                display_ascii_radio.setEnabled(False)
                return

            bit_count = len(bits)
            byte_count = bit_count // 8
            remainder = bit_count % 8

            # Enable/disable display options based on divisibility
            divisible_by_4 = (bit_count % 4 == 0)
            divisible_by_8 = (bit_count % 8 == 0)

            display_binary_radio.setEnabled(True)
            display_hex_radio.setEnabled(divisible_by_4)
            display_ascii_radio.setEnabled(divisible_by_8)

            # Update selection if current choice is disabled
            if not divisible_by_4 and display_hex_radio.isChecked():
                display_binary_radio.setChecked(True)
            if not divisible_by_8 and display_ascii_radio.isChecked():
                display_binary_radio.setChecked(True)

            # Show info
            info_text = f"Pattern length: {bit_count} bits"
            if divisible_by_8:
                info_text += f" ({byte_count} bytes)"
            elif remainder > 0:
                info_text += f" ({byte_count} bytes + {remainder} bits)"
            info_label.setText(info_text)

        pattern_input.textChanged.connect(update_display_options)
        input_binary_radio.toggled.connect(update_display_options)

        dialog_result = dialog.exec()

        # Clear highlights after dialog closes (whether OK or Cancel)
        self.byte_table.clear_pattern_highlights()

        if dialog_result != QDialog.DialogCode.Accepted:
            return

        # Parse pattern from input (user doesn't have to click Find first)
        pattern_text = pattern_input.text().strip()
        if not pattern_text:
            QMessageBox.warning(self, "Frame Sync", "Please enter a pattern.")
            return

        # Parse pattern to bits
        bits = None
        if input_binary_radio.isChecked():
            bits_str = pattern_text.upper().replace('X', '1').replace('O', '0')
            if all(c in '10' for c in bits_str):
                bits = [int(b) for b in bits_str]
        else:
            hex_str = pattern_text.upper()
            if hex_str.startswith('0X'):
                hex_str = hex_str[2:]
            if all(c in '0123456789ABCDEF' for c in hex_str):
                bits = []
                for hex_char in hex_str:
                    val = int(hex_char, 16)
                    bits.extend([int(b) for b in format(val, '04b')])

        if not bits:
            QMessageBox.warning(self, "Frame Sync", "Invalid pattern.")
            return

        self.save_state()

        # Actually perform framing now
        # ALWAYS use bit-level search to find patterns at any bit position
        # (not just byte-aligned positions)
        frame_count = self.byte_table.set_frame_pattern(None, pattern_bits=bits)

        if frame_count == 0:
            QMessageBox.information(self, "Frame Sync", "Pattern not found.")
            return

        # Update 'Bytes per row' to max frame length and disable it
        frames = self.byte_table.frames
        if frames:
            max_len = max(len(f) for f in frames)
            self.row_size_spin.blockSignals(True)
            self.row_size_spin.setValue(max_len)
            self.row_size_spin.blockSignals(False)
            self.row_size_spin.setEnabled(False)

        # Create column definition for sync pattern
        label_text = label_input.text().strip() or "Sync"
        color_name = color_combo.currentText()
        if color_name == "None":
            color_name = None
        bit_count = len(bits)

        # Determine display format (map to proper format strings)
        if display_ascii_radio.isChecked() and bit_count % 8 == 0:
            display_fmt = "ascii_be"
        elif display_hex_radio.isChecked() and bit_count % 4 == 0:
            display_fmt = "hex_be"
        else:
            display_fmt = "binary"

        # Create a SINGLE bit-based column definition for the ENTIRE pattern
        # Always use bit-based columns to extract exactly the pattern length
        # This handles both byte-aligned and non-byte-aligned patterns correctly
        # ColumnDefinition(start_byte, end_byte, label, display_format, color_name, unit, start_bit, total_bits)

        col_def = ColumnDefinition(
            0,                  # start_byte (calculated from start_bit)
            0,                  # end_byte (calculated from start_bit + total_bits)
            label_text,         # label
            display_fmt,        # display_format
            color_name,         # color_name
            "bit",              # unit (always bit-based for precise extraction)
            0,                  # start_bit (starts at bit 0 of frame)
            bit_count           # total_bits (exact pattern length in bits)
        )
        self.byte_table.add_column_definition(col_def)

        # Add the column definition to the UI list widget
        desc = f"{col_def.label if col_def.label else '(unnamed)'}: Bit {col_def.start_bit}+{col_def.total_bits} as {col_def.display_format} ({col_def.color_name})"
        item = QListWidgetItem(desc)

        # Set background color to match the column definition color
        if col_def.color_name != "None":
            color = self.byte_table._color_from_name(col_def.color_name)
            if color:
                item.setBackground(color)

        self.columns_list.addItem(item)

        op_text = f"Frame Sync: {pattern_text}"
        self.operations.append(op_text)
        self.add_operation_to_list(op_text)

        # Check if frames include zero-padding (when pattern length isn't byte-aligned)
        if bit_count % 8 != 0:
            padding_bits = 8 - (bit_count % 8)
            self.status_label.setText(f"Framed into {frame_count} frames (zero-padded with {padding_bits} bits per frame)")
        else:
            self.status_label.setText(f"Framed into {frame_count} frames")

    def clear_frame_sync(self):
        if self.bytes_data is None:
            return

        self.save_state()
        self.byte_table.clear_frames()

        # Clear all column definitions (colors and custom display formats)
        self.clear_column_definitions()

        # Reset to default row size (16 bytes per row)
        self.row_size_spin.setValue(16)

        # Re-enable Bytes per row spinbox
        self.row_size_spin.setEnabled(True)

        op_text = "Clear Frames"
        self.operations.append(op_text)
        self.add_operation_to_list(op_text)
        self.status_label.setText("Frames cleared, column definitions cleared, row size reset to 16")

    def highlight_constant_columns(self):
        if self.bytes_data is None:
            return
        self.byte_table.highlight_constant_columns()
        self.status_label.setText("Constant columns highlighted")

    def clear_constant_highlights(self):
        self.byte_table.clear_constant_highlights()
        self.status_label.setText("Constant highlights cleared")

    def save_state(self):
        bits_copy = self.bits.copy() if self.bits is not None else None
        bytes_copy = self.bytes_data.copy() if self.bytes_data is not None else None
        ops_copy = self.operations.copy()
        frame_pat = list(self.byte_table.frame_pattern) if self.byte_table.frame_pattern is not None else None
        state = (bits_copy, bytes_copy, ops_copy, frame_pat)
        self.undo_stack.append(state)
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            self.status_label.setText("Nothing to undo")
            return

        current_state = (
            self.bits.copy() if self.bits is not None else None,
            self.bytes_data.copy() if self.bytes_data is not None else None,
            self.operations.copy(),
            list(self.byte_table.frame_pattern) if self.byte_table.frame_pattern is not None else None
        )
        self.redo_stack.append(current_state)

        bits, bytes_data, operations, frame_pattern = self.undo_stack.pop()
        self.bits = bits.copy() if bits is not None else None
        self.bytes_data = bytes_data.copy() if bytes_data is not None else None
        self.operations = operations.copy()

        if frame_pattern is not None:
            self.byte_table.set_frame_pattern(frame_pattern)
            frames = self.byte_table.frames
            if frames:
                max_len = max(len(f) for f in frames)
                self.row_size_spin.blockSignals(True)
                self.row_size_spin.setValue(max_len)
                self.row_size_spin.blockSignals(False)
                self.row_size_spin.setEnabled(False)
        else:
            self.byte_table.clear_frames()
            self.row_size_spin.setEnabled(True)

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
            list(self.byte_table.frame_pattern) if self.byte_table.frame_pattern is not None else None
        )
        self.undo_stack.append(current_state)

        bits, bytes_data, operations, frame_pattern = self.redo_stack.pop()
        self.bits = bits.copy() if bits is not None else None
        self.bytes_data = bytes_data.copy() if bytes_data is not None else None
        self.operations = operations.copy()

        if frame_pattern is not None:
            self.byte_table.set_frame_pattern(frame_pattern)
            frames = self.byte_table.frames
            if frames:
                max_len = max(len(f) for f in frames)
                self.row_size_spin.blockSignals(True)
                self.row_size_spin.setValue(max_len)
                self.row_size_spin.blockSignals(False)
                self.row_size_spin.setEnabled(False)
        else:
            self.byte_table.clear_frames()
            self.row_size_spin.setEnabled(True)

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

        delete_btn = QPushButton("✕")
        delete_btn.setFixedSize(20, 20)
        delete_btn.setStyleSheet("""
            QPushButton { background-color: transparent; color: #888; border: none; font-weight: bold; font-size: 14px; }
            QPushButton:hover { background-color: #ffcccc; color: #cc0000; border-radius: 10px; }
        """)

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
                    match = re.search(r'Delta (\d+)', op_text)
                    if match:
                        window = int(match.group(1))
                        self.apply_delta_internal(window)
                elif op_text.startswith("Take/Skip"):
                    match = re.search(r'Take/Skip: (.+)', op_text)
                    if match:
                        pattern = match.group(1)
                        self.apply_takeskip_internal(pattern)
                elif op_text.startswith("XOR"):
                    match = re.search(r'XOR: (.+)', op_text)
                    if match:
                        pattern = match.group(1)
                        self.apply_xor_internal(pattern)
                elif op_text == "Invert All":
                    self.apply_invert_internal()
            else:
                if op_text.startswith("Frame Sync:"):
                    pattern_str = op_text.split("Frame Sync:", 1)[1].strip()
                    pattern = self.parse_hex_pattern(pattern_str)
                    if pattern is not None:
                        self.byte_table.set_frame_pattern(pattern)
                        frames = self.byte_table.frames
                        if frames:
                            max_len = max(len(f) for f in frames)
                            self.row_size_spin.blockSignals(True)
                            self.row_size_spin.setValue(max_len)
                            self.row_size_spin.blockSignals(False)
                            self.row_size_spin.setEnabled(False)
                elif op_text == "Clear Frames":
                    self.byte_table.clear_frames()
                    self.row_size_spin.setEnabled(True)

            self.add_operation_to_list(op_text)

        self.update_display()

    def apply_takeskip_internal(self, pattern):
        matches = re.findall(r'([tsir])(\d+)', pattern.lower())

        # Calculate pattern cycle length
        pattern_cycle_len = sum(int(num) for _, num in matches)

        # OPTIMIZED: For repeating patterns, use boolean mask (ultra-fast!)
        # If pattern repeats many times (>1000), use fancy indexing
        num_cycles = len(self.bits) // pattern_cycle_len if pattern_cycle_len > 0 else 0

        if num_cycles > 1000 and all(op in ('t', 's') for op, _ in matches):
            # ULTRA OPTIMIZED: Use reshape + slicing (zero-copy when possible!)

            # Find contiguous take ranges
            take_ranges = []
            pos = 0
            for op, num_str in matches:
                num = int(num_str)
                if op == 't':
                    take_ranges.append((pos, pos + num))
                pos += num

            full_bits = len(self.bits)
            num_full_cycles = full_bits // pattern_cycle_len

            # SPECIAL CASE: t2s2, t8s8, etc. - single take at start (ULTRA FAST!)
            if len(take_ranges) == 1 and take_ranges[0][0] == 0:
                take_len = take_ranges[0][1]
                data_len = num_full_cycles * pattern_cycle_len

                # Reshape and slice - extremely fast!
                reshaped = self.bits[:data_len].reshape(num_full_cycles, pattern_cycle_len)
                result = reshaped[:, :take_len].copy().ravel()

                # Remainder
                remainder = full_bits - data_len
                if remainder > 0:
                    take_remainder = min(take_len, remainder)
                    if take_remainder > 0:
                        result = np.append(result, self.bits[data_len:data_len + take_remainder])

                self.bits = result
                return

            # General case: Process in manageable chunks
            bits_per_cycle = sum(end - start for start, end in take_ranges)
            chunk_cycles = 5_000_000  # Process 5M cycles at a time

            result_parts = []

            for cycle_start in range(0, num_full_cycles, chunk_cycles):
                cycle_end = min(cycle_start + chunk_cycles, num_full_cycles)
                n_cycles = cycle_end - cycle_start

                # Reshape this chunk
                start_bit = cycle_start * pattern_cycle_len
                end_bit = cycle_end * pattern_cycle_len
                chunk = self.bits[start_bit:end_bit].reshape(n_cycles, pattern_cycle_len)

                # Extract each take range
                for t_start, t_end in take_ranges:
                    result_parts.append(chunk[:, t_start:t_end].ravel())

            # Concatenate all parts
            result = np.concatenate(result_parts)

            # Handle remainder
            remainder = full_bits % pattern_cycle_len
            if remainder > 0:
                rem_start = num_full_cycles * pattern_cycle_len
                for t_start, t_end in take_ranges:
                    if t_start < remainder:
                        actual_end = min(t_end, remainder)
                        result = np.append(result, self.bits[rem_start + t_start:rem_start + actual_end])

            self.bits = result
            return

        # ORIGINAL: For complex patterns or few cycles, use operation list
        operations = []
        pos = 0
        total_output = 0

        # Parse pattern and build operation list (lightweight, no data copying)
        while pos < len(self.bits):
            for op, num_str in matches:
                num = int(num_str)
                if pos >= len(self.bits):
                    break

                end = min(pos + num, len(self.bits))
                if op in ('t', 'r', 'i'):
                    operations.append((op, pos, end))
                    total_output += (end - pos)
                # 's' (skip) just advances pos without adding operation
                pos = end

            if pos >= len(self.bits):
                break

        # Pre-allocate result array
        result = np.empty(total_output, dtype=np.uint8)
        result_idx = 0

        # Process operations in batches for better cache locality
        for op, start, end in operations:
            chunk_len = end - start

            if op == 't':
                result[result_idx:result_idx + chunk_len] = self.bits[start:end]
            elif op == 'r':
                result[result_idx:result_idx + chunk_len] = self.bits[start:end][::-1]
            elif op == 'i':
                result[result_idx:result_idx + chunk_len] = 1 - self.bits[start:end]

            result_idx += chunk_len

        self.bits = result

    def apply_delta_internal(self, window):
        if window == 1:
            # OPTIMIZED: Simple case - single bit delta (already optimal)
            result = np.bitwise_xor(self.bits[1:], self.bits[:-1])
        else:
            # OPTIMIZED: Fully vectorized multi-window delta
            # Calculate how many complete window pairs we have
            total_bits = len(self.bits)
            num_complete_windows = (total_bits - window) // window

            # Process all complete windows at once using array slicing
            if num_complete_windows > 0:
                # Reshape into windows for vectorized XOR
                end_idx = window + num_complete_windows * window

                # Extract current windows (starting at 'window' offset)
                current = self.bits[window:end_idx].reshape(num_complete_windows, window)
                # Extract previous windows (starting at 0)
                previous = self.bits[0:end_idx - window].reshape(num_complete_windows, window)

                # Vectorized XOR across all windows at once!
                deltas = np.bitwise_xor(current, previous)
                result = deltas.flatten()

                # Handle remaining bits (if any) that don't form complete window
                remainder_start = end_idx
                if remainder_start < total_bits:
                    remainder_len = total_bits - remainder_start
                    current_remainder = self.bits[remainder_start:total_bits]
                    prev_remainder = self.bits[remainder_start - window:remainder_start - window + remainder_len]
                    delta_remainder = np.bitwise_xor(current_remainder, prev_remainder)
                    result = np.concatenate([result, delta_remainder])
            else:
                # Very short array, just do it manually
                result = np.empty(0, dtype=np.uint8)

        self.bits = result

    def apply_xor_internal(self, pattern):
        if pattern.startswith('0x') or pattern.startswith('0X'):
            pattern_bits = self.hex_to_bits(pattern)
        else:
            pattern_bits = self.binary_to_bits(pattern)

        if pattern_bits is None:
            return

        pattern_len = len(pattern_bits)
        data_len = len(self.bits)

        # OPTIMIZED: Fully vectorized XOR using numpy tile/repeat
        if pattern_len == 1:
            # Special case: single bit XOR (super fast)
            self.bits = np.bitwise_xor(self.bits, pattern_bits[0])
        else:
            # Tile the pattern to match data length
            num_repeats = (data_len + pattern_len - 1) // pattern_len
            extended_pattern = np.tile(pattern_bits, num_repeats)[:data_len]

            # Vectorized XOR - operates on entire arrays at once!
            self.bits = np.bitwise_xor(self.bits, extended_pattern)

    def apply_invert_internal(self):
        self.bits = 1 - self.bits

    def _ensure_bits_loaded(self):
        """Helper to ensure bits are loaded (lazy loading)"""
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

        if filename:
            try:
                with open(filename, 'rb') as f:
                    byte_data = np.frombuffer(f.read(), dtype=np.uint8)

                # OPTIMIZED: Only keep one copy of bytes
                self.original_bytes = byte_data.copy()
                self.bytes_data = self.original_bytes  # Share reference, will copy on modify

                # OPTIMIZED: Don't unpack bits immediately for large files
                # This saves massive memory (300MB file → 2.4GB bit array)
                # Bits will be unpacked on-demand when switching to bit mode
                self.original_bits = None
                self.bits = None

                self.filename = filename
                self.operations = []
                self.operations_list.clear()
                self.undo_stack = []
                self.redo_stack = []

                # Clear frames and column definitions from previous file
                self.byte_table.clear_frames()
                self.clear_column_definitions()

                # Reset to default row size (16 bytes per row)
                self.row_size_spin.setValue(16)
                self.row_size_spin.setEnabled(True)

                file_size = len(byte_data)
                bit_count = file_size * 8

                # OPTIMIZED: Only unpack bits if currently in bit mode AND file is small enough
                is_large_file = file_size >= 10_000_000  # 10MB threshold

                if self.data_mode == "bit" and not is_large_file:
                    # Small file in bit mode: unpack immediately
                    self.bits = np.unpackbits(byte_data)
                    self.original_bits = self.bits
                    ones = np.sum(self.bits)
                    ones_pct = ones / bit_count * 100
                elif self.data_mode == "bit" and is_large_file:
                    # Large file in bit mode: lazy load bits in update_display
                    ones = "N/A (calculating...)"
                    ones_pct = "N/A"
                else:
                    # Byte mode: no need to unpack bits at all
                    ones = "N/A"
                    ones_pct = "N/A"

                self.file_info_label.setText(
                    f"File: {filename.split('/')[-1]}\n"
                    f"Size: {file_size} bytes ({bit_count} bits)\n"
                    f"Ones: {ones} ({ones_pct}{'%' if ones_pct != 'N/A' else ''})"
                )

                self.update_display()
                if is_large_file and self.data_mode == "bit":
                    self.status_label.setText(f"Loaded {file_size} bytes (large file - unpacking bits...)")
                else:
                    self.status_label.setText(f"Loaded {file_size} bytes")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open file: {e}")

    def save_file(self):
        if self.data_mode == "bit" and self.bits is None:
            QMessageBox.warning(self, "Warning", "No data to save")
            return
        elif self.data_mode == "byte" and self.bytes_data is None:
            QMessageBox.warning(self, "Warning", "No data to save")
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Save Processed Data", "",
                                                  "Binary files (*.bin);;All files (*.*)")

        if filename:
            try:
                if self.data_mode == "bit":
                    bits_to_save = self.bits.copy()
                    padding = (8 - (len(bits_to_save) % 8)) % 8
                    if padding:
                        bits_to_save = np.append(bits_to_save, np.zeros(padding, dtype=np.uint8))
                    byte_data = np.packbits(bits_to_save)
                else:
                    byte_data = self.bytes_data

                with open(filename, 'wb') as f:
                    f.write(byte_data.tobytes())

                QMessageBox.information(self, "Success", f"Saved {len(byte_data)} bytes")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {e}")

    def update_display(self):
        if self.data_mode == "bit":
            if self.bits is None:
                # OPTIMIZED: Lazy load bits from bytes on demand
                if self.bytes_data is not None:
                    self.status_label.setText("Unpacking bits from bytes (one-time operation)...")
                    QApplication.processEvents()  # Update UI to show status

                    self.bits = np.unpackbits(self.bytes_data)
                    self.original_bits = self.bits

                    # Update file info with bit count
                    bit_count = len(self.bits)
                    # For large files, skip expensive sum
                    if len(self.bytes_data) < 10_000_000:
                        ones = np.sum(self.bits)
                        ones_pct = ones / bit_count * 100
                    else:
                        ones = "N/A"
                        ones_pct = "N/A"

                    # Update the file info label
                    file_size = len(self.bytes_data)
                    self.file_info_label.setText(
                        f"File: {self.filename.split('/')[-1] if self.filename else 'N/A'}\n"
                        f"Size: {file_size} bytes ({bit_count} bits)\n"
                        f"Ones: {ones} ({ones_pct}{'%' if ones_pct != 'N/A' else ''})"
                    )
                    self.status_label.setText("Bits unpacked - ready to display")
                else:
                    # No data at all
                    self.scroll_area.hide()
                    self.text_display.hide()
                    self.byte_table.hide()
                    self.status_label.setText("No data loaded")
                    return

            mode = self.mode_combo.currentText().lower()
            self.byte_table.hide()

            if mode in ["squares", "circles"]:
                self.scroll_area.show()
                self.text_display.hide()
                self.canvas.bits_per_row = self.width_spin.value()
                self.canvas.bit_size = self.size_spin.value()
                self.canvas.display_mode = mode
                self.canvas.set_bits(self.bits)

            elif mode in ["binary", "hex"]:
                self.scroll_area.hide()
                self.text_display.show()
                self.text_display.width = self.width_spin.value()
                self.text_display.display_mode = mode
                self.text_display.set_bits(self.bits)
        else:
            if self.bytes_data is None:
                return

            self.scroll_area.hide()
            self.text_display.hide()
            self.byte_table.show()
            self.byte_table.set_bytes(self.bytes_data)

    def hex_to_bits(self, hex_string):
        if hex_string.startswith('0x') or hex_string.startswith('0X'):
            hex_string = hex_string[2:]
        try:
            value = int(hex_string, 16)
            bit_length = len(hex_string) * 4
            binary_str = format(value, f'0{bit_length}b')
            return np.array([int(b) for b in binary_str], dtype=np.uint8)
        except ValueError:
            return None

    def binary_to_bits(self, binary_string):
        try:
            return np.array([int(b) for b in binary_string if b in '01'], dtype=np.uint8)
        except ValueError:
            return None

    def find_pattern_positions(self, pattern_bits):
        if self.bits is None or pattern_bits is None:
            return []

        positions = []
        pattern_len = len(pattern_bits)
        error_percent = self.error_tolerance_spin.value()
        max_errors = int((error_percent / 100.0) * pattern_len)

        # OPTIMIZED: Use numpy stride tricks for fast sliding window search
        from numpy.lib.stride_tricks import sliding_window_view

        if len(self.bits) < pattern_len:
            return positions

        # Create sliding window view (no copying, just indexing trick)
        windows = sliding_window_view(self.bits, pattern_len)

        if max_errors == 0:
            # Exact match: compare all windows at once
            matches = np.all(windows == pattern_bits, axis=1)
            positions = np.where(matches)[0].tolist()
        else:
            # Fuzzy match: count differences for all windows at once
            differences = np.sum(windows != pattern_bits, axis=1)
            positions = np.where(differences <= max_errors)[0].tolist()

        return positions

    def highlight_pattern(self):
        if not self._ensure_bits_loaded():
            self.pattern_results_label.setText("No bit data")
            return

        pattern_str = self.pattern_input.text().strip()
        if not pattern_str:
            return

        if pattern_str.startswith('0x') or pattern_str.startswith('0X'):
            pattern_bits = self.hex_to_bits(pattern_str)
        else:
            pattern_bits = self.binary_to_bits(pattern_str)

        if pattern_bits is None:
            self.pattern_results_label.setText("Invalid pattern")
            return

        positions = self.find_pattern_positions(pattern_bits)

        if positions:
            error_percent = self.error_tolerance_spin.value()
            if error_percent > 0:
                self.pattern_results_label.setText(f"Found {len(positions)} (±{error_percent}%)")
            else:
                self.pattern_results_label.setText(f"Found {len(positions)}")
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

        # Create custom dialog with preview
        dialog = QDialog(self)
        dialog.setWindowTitle("Take/Skip Pattern")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)

        # Explanation
        explanation = QLabel("Enter pattern (e.g., t4r3i8s1):\nt=take, r=reverse, i=invert, s=skip")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        # Pattern input
        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("Pattern:"))
        pattern_input = QLineEdit()
        pattern_layout.addWidget(pattern_input)
        layout.addLayout(pattern_layout)

        # Preview label
        preview_label = QLabel("")
        preview_label.setStyleSheet("QLabel { color: blue; font-weight: bold; }")
        layout.addWidget(preview_label)

        # Preview button
        def preview_pattern():
            pattern = pattern_input.text().strip()
            if not pattern:
                preview_label.setText("Enter a pattern first")
                return

            try:
                # Calculate result size
                matches = re.findall(r'([tsir])(\d+)', pattern.lower())
                if not matches:
                    preview_label.setText("Invalid pattern format")
                    return

                pattern_cycle_len = sum(int(num) for _, num in matches)
                if pattern_cycle_len == 0:
                    preview_label.setText("Pattern has zero length")
                    return

                bits_per_cycle = sum(int(num) for op, num in matches if op in ('t', 'r', 'i'))
                num_cycles = len(self.bits) // pattern_cycle_len
                output_bits = num_cycles * bits_per_cycle

                # Add remainder
                remainder = len(self.bits) % pattern_cycle_len
                rem_pos = 0
                for op, num_str in matches:
                    num = int(num_str)
                    if rem_pos >= remainder:
                        break
                    if op in ('t', 'r', 'i'):
                        output_bits += min(num, remainder - rem_pos)
                    rem_pos += num

                input_bits = len(self.bits)
                reduction_pct = (1 - output_bits / input_bits) * 100 if input_bits > 0 else 0

                preview_label.setText(
                    f"✓ Preview: {input_bits:,} bits → {output_bits:,} bits\n"
                    f"  ({reduction_pct:.1f}% reduction, {num_cycles:,} cycles)"
                )
            except Exception as e:
                preview_label.setText(f"Error: {str(e)}")

        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(preview_pattern)
        layout.addWidget(preview_btn)

        # Dialog buttons
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
            scrollbar = self.scroll_area.verticalScrollBar()
            scrollbar.setValue(0)
        elif self.text_display.isVisible():
            scrollbar = self.text_display.verticalScrollBar()
            scrollbar.setValue(0)

    def go_to_end(self):
        if self.scroll_area.isVisible():
            scrollbar = self.scroll_area.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        elif self.text_display.isVisible():
            scrollbar = self.text_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())


def main():
    app = QApplication(sys.argv)
    window = BitViewerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()