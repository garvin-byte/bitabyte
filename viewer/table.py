"""ColumnHighlightDelegate and ByteStructuredTableWidget."""

import math
import re
import numpy as np
from numpy.lib.stride_tricks import as_strided
from collections import Counter
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
                              QSizePolicy, QTableWidget, QTableWidgetItem, QHeaderView,
                              QAbstractItemView, QMenu, QStyledItemDelegate,
                              QApplication, QCheckBox, QSpinBox, QComboBox,
                              QDialog, QDialogButtonBox, QGroupBox, QListWidget,
                              QListWidgetItem, QMessageBox, QPushButton, QLineEdit,
                              QInputDialog, QButtonGroup, QRadioButton, QDockWidget)
from PyQt6.QtCore import Qt, QRect, QSize
from PyQt6.QtGui import QColor, QPen, QBrush, QFont, QPainter, QIcon, QPixmap
from .colors import (COLOR_OPTIONS, COLOR_NAME_TO_QCOLOR, BIT_SYNC_WARNING_BYTES,
                     BIT_SYNC_HARD_LIMIT_BYTES, MAX_SYNC_FRAMES,
                     PATTERN_SCAN_MAX_ELEMENTS, _populate_color_combo)
from .column import ColumnDefinition, AddColumnDialog



class ColumnHighlightDelegate(QStyledItemDelegate):
    """Custom delegate to draw column highlighting on top of everything."""

    def __init__(self, table_widget, parent=None):
        super().__init__(parent)
        self.table_widget = table_widget

    def paint(self, painter, option, index):
        # First, let the default painting happen
        super().paint(painter, option, index)

        # Then draw our highlight overlay if this column is selected
        # Skip header rows (rows 0, 1, 2) - only highlight data rows
        row = index.row()
        col = index.column()
        if row >= 3 and col in self.table_widget.selected_columns:
            # Draw a semi-transparent light blue overlay
            painter.save()
            painter.setCompositionMode(painter.CompositionMode.CompositionMode_SourceOver)
            overlay_color = QColor(220, 235, 255, 100)  # Light blue with alpha
            painter.fillRect(option.rect, overlay_color)
            painter.restore()


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

        # Framing - NOW LAZY!
        self.frames = None  # Will store lazy frame info: list of (start_bit, length_bits) tuples
        self.frame_pattern = None  # stored pattern metadata (dict)
        self.frame_bit_offsets = None  # list of bit offsets within first byte of each frame

        # Constant column highlighting
        self.constant_columns = set()
        self._all_columns_info = None

        # Pattern highlighting (for frame sync search preview)
        self.pattern_highlight_positions = []  # List of bit positions
        self.pattern_highlight_length = 0  # Length of pattern in bits

        # Byte-mode overlay preferences
        self.show_binary_overlay = False
        self.show_ascii_overlay = False

        # Parent window reference (set by parent)
        self.parent_window = None

        # OPTIMIZED: Add row and column limits for large datasets
        self.max_display_rows = 1000  # CRITICAL: Limit frames displayed
        self.max_display_cols = 256   # CRITICAL: Limit columns (bytes per frame) displayed

        self.setFont(QFont("Courier", 9))
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectItems)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)

        # Virtual row rendering state
        self._display_meta = None        # metadata for lazy row population
        self._populated_data_rows = set()  # which data rows have been filled

        # Track selected columns for highlighting
        self.selected_columns = set()

        # Flag to prevent focusOutEvent from clearing selection when context menu is open
        self._showing_context_menu = False
        self._suspend_focus_clear = False

        # Install custom delegate for drawing column highlighting
        self.highlight_delegate = ColumnHighlightDelegate(self)
        self.setItemDelegate(self.highlight_delegate)

        # Connect signals
        self.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.cellClicked.connect(self._on_cell_clicked)
        self.itemSelectionChanged.connect(self._on_selection_changed)
        self.verticalScrollBar().valueChanged.connect(self._populate_visible_rows)

        # Disable automatic context menu - we'll handle it manually in mousePressEvent
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)

        # Install event filter to detect clicks outside the table
        self.viewport().installEventFilter(self)

        # Track focus to clear selection when clicking outside
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Store split column information: {byte_col_idx: {'type': 'binary'/'nibble', 'label': 'Binary'/'Nibble', 'color': 'color_name'}}
        self.split_columns = {}  # Maps byte column index to split info dict

        # Cache unpacked bits for repeated bit-level searches
        self._bit_cache = None
        self._bit_cache_source_id = None
        self._bit_cache_length = 0

    def focusOutEvent(self, event):
        """Clear column selection when focus is lost (clicked outside table)."""
        # Don't clear selection if we're showing a context menu - we want highlighting to persist
        if self._showing_context_menu:
            super().focusOutEvent(event)
            return

        if self._suspend_focus_clear:
            super().focusOutEvent(event)
            return

        # Keep selection when moving focus into the field inspector dock
        keep_selection = False
        if self.parent_window is not None:
            from PyQt6.QtWidgets import QApplication

            target_widget = QApplication.focusWidget()
            inspector = getattr(self.parent_window, "field_inspector", None)
            if inspector and target_widget is not None:
                if target_widget is inspector or inspector.isAncestorOf(target_widget):
                    keep_selection = True

        if keep_selection:
            super().focusOutEvent(event)
            return

        if self.selected_columns:
            # Restore colors for all selected columns before clearing
            for col in self.selected_columns:
                self._restore_column_colors(col)
            self.selected_columns.clear()
            self.clearSelection()
            # Trigger repaint to clear highlighting
            self.viewport().update()
            self._update_live_bit_viewer()
        super().focusOutEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press - show context menu ONLY on right-click."""
        from PyQt6.QtCore import Qt

        # Get the cell at this position
        item = self.itemAt(event.pos())
        if item:
            row = self.row(item)
            col = self.column(item)

        if event.button() == Qt.MouseButton.RightButton:
            # Right-click - show context menu WITHOUT clearing selection
            # Don't call parent or trigger any selection changes
            self._show_context_menu(event.pos())
            # Important: Don't call super() - that would clear the selection
            return
        else:
            # Left-click - pass to parent first, then handle column selection
            super().mousePressEvent(event)

            # Manually trigger column selection logic since cellClicked signal isn't firing
            if item:
                self._on_cell_clicked(row, col)

    def _on_cell_double_clicked(self, row, col):
        """
        Handle double-click on header labels:
        - If this column belongs to an existing ColumnDefinition -> open edit dialog
        - If this column is an undefined '?' region -> open a 'define bits' dialog
        - If this column is part of a split -> allow editing the split label
        """
        # Only react on the top label row
        if row != 0:
            return

        if self._all_columns_info is None:
            return
        if col < 0 or col >= len(self._all_columns_info):
            return

        # Check if this column is part of a split
        split_col_info = self._find_split_for_column(col)
        if split_col_info is not None:
            # This is a split column - allow editing the label
            self._edit_split_label(split_col_info)
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

    def eventFilter(self, source, event):
        """Filter events to detect clicks outside the table."""
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QMouseEvent

        if source == self.viewport() and event.type() == QEvent.Type.MouseButtonPress:
            if isinstance(event, QMouseEvent):
                pos = event.pos()
                item = self.itemAt(pos)

                if item is None:
                    # Clicked on empty space in viewport - clear selection
                    self.selected_columns.clear()
                    self.clearSelection()
                    self.update_display()
                    self._update_live_bit_viewer()
                    return False

        return super().eventFilter(source, event)

    def _on_cell_clicked(self, _row, col):
        """Handle cell click - select ONLY that column (clears previous)."""
        # Single click = clear all previous selections and select only this column
        self.blockSignals(True)  # Prevent recursive signals
        self.clearSelection()  # Clear Qt's built-in selection
        self.blockSignals(False)

        self.selected_columns.clear()
        self.selected_columns.add(col)

        # Trigger repaint with our custom delegate
        self.viewport().update()

        # Update live bit viewer
        self._update_live_bit_viewer()

    def _on_selection_changed(self):
        """Handle selection changes from click+drag to select multiple columns."""
        # Get all selected items from Qt's selection
        selected_items = self.selectedItems()

        if not selected_items:
            # No items selected - clear highlighting
            if self.selected_columns:
                self.selected_columns.clear()
                self.viewport().update()
                self._update_live_bit_viewer()
            return

        # Extract unique column indices from selected items (for drag selection)
        new_columns = set()
        for item in selected_items:
            col = self.column(item)
            if col >= 0:
                new_columns.add(col)

        # Only update if this is a drag selection (multiple items)
        # Single clicks are handled by _on_cell_clicked
        if len(selected_items) > 1:
            self.selected_columns = new_columns
            # Trigger repaint with our custom delegate
            self.viewport().update()
            # Update live bit viewer
            self._update_live_bit_viewer()

    def _restore_column_colors(self, col):
        """Restore original colors for a specific column based on column definitions."""
        for row in range(self.rowCount()):
            item = self.item(row, col)
            if item:
                text = item.text()

                # Restore based on column type (same logic as update_display)
                if col in self.constant_columns and text != "":
                    # Constant column - yellow
                    item.setBackground(QColor(255, 255, 180))
                elif col < len(self._all_columns_info):
                    # Check if this column has a definition
                    _, _, _, col_def, _ = self._all_columns_info[col]
                    if col_def is not None:
                        col_color = self._color_from_name(col_def.color_name)
                        if col_color is not None and text != "":
                            item.setBackground(col_color)
                        else:
                            # No color for this column definition - clear
                            item.setData(Qt.ItemDataRole.BackgroundRole, None)
                    else:
                        # No column definition - clear to default
                        item.setData(Qt.ItemDataRole.BackgroundRole, None)
                else:
                    # Column not in _all_columns_info - clear to default
                    item.setData(Qt.ItemDataRole.BackgroundRole, None)

    def _apply_selection_highlighting(self):
        """Apply light blue highlighting to all currently selected columns."""
        if not self.selected_columns:
            return

        for col in self.selected_columns:
            for row in range(self.rowCount()):
                item = self.item(row, col)
                if item:
                    item.setBackground(QColor(220, 235, 255))

    def _update_column_highlighting(self):
        """Update background colors to show selected columns - preserves existing column definition colors."""
        if self.rowCount() == 0 or self.columnCount() == 0:
            return

        # Only update cells in selected columns or cells that need unhighlighting
        for row in range(self.rowCount()):
            for col in range(self.columnCount()):
                item = self.item(row, col)
                if item is None:
                    continue

                # Only process if this column is selected OR needs to be cleared from previous selection
                # Get the current background to check if it's the selection color
                current_bg = item.background().color()
                is_currently_highlighted = (current_bg == QColor(220, 235, 255) or
                                           current_bg.red() >= 200 and current_bg.green() >= 200 and current_bg.blue() >= 200)

                # Skip this cell if column not selected and not currently highlighted
                if col not in self.selected_columns and not is_currently_highlighted:
                    continue

                # Get original color from item's data (we'll store it there)
                original_color = item.data(Qt.ItemDataRole.UserRole)

                if col in self.selected_columns:
                    # Store original color if not already stored
                    if original_color is None:
                        current_color = item.background().color()
                        # Don't store if current color is already the selection color
                        selection_color = QColor(220, 235, 255)
                        if current_color != selection_color:
                            item.setData(Qt.ItemDataRole.UserRole, current_color)
                            original_color = current_color

                    # Apply light blue highlight to entire column (uniform color, no darker cell)
                    # DON'T blend - just use pure light blue for consistent color
                    base_color = QColor(220, 235, 255)  # Light blue
                    item.setBackground(base_color)
                else:
                    # Unselected column that was previously highlighted
                    # Restore original color - if no UserRole, recalculate from column definition
                    if original_color:
                        item.setBackground(original_color)
                    else:
                        # No stored color - need to recalculate based on column defs, patterns, etc.
                        # This happens after table re-render. Just call restore for this column.
                        # But we're already in a loop, so just recalculate for this specific cell
                        if col < len(self._all_columns_info):
                            byte_offset, _, _, col_def, _ = self._all_columns_info[col]

                            if row < 3:
                                # Header rows
                                item.setBackground(QColor(255, 255, 255))
                            else:
                                # Data rows
                                data_row_idx = row - 3
                                base_byte_idx = data_row_idx * self.row_size
                                absolute_byte_idx = base_byte_idx + byte_offset

                                # Check pattern highlight first
                                if self._is_byte_in_pattern_highlight(absolute_byte_idx):
                                    item.setBackground(QColor(255, 200, 0))
                                # Then check constant columns
                                elif col in self.constant_columns and item.text() != "":
                                    item.setBackground(QColor(255, 255, 180))
                                # Then check column definition
                                elif col_def is not None:
                                    col_color = self._color_from_name(col_def.color_name)
                                    item.setBackground(col_color)
                                # Default: white
                                else:
                                    item.setBackground(QColor(255, 255, 255))
                        else:
                            # Fallback to white
                            item.setBackground(QColor(255, 255, 255))

    def _show_context_menu(self, position):
        """Show context menu for adding column definitions from selection."""
        # Get the column at the click position
        item = self.itemAt(position)
        clicked_col = None
        if item:
            clicked_col = self.column(item)

        # Store selected columns BEFORE showing menu (menu might trigger events that clear it)
        selected_cols_snapshot = self.selected_columns.copy()

        # If no columns are selected but user right-clicked on a column, use that column
        if not selected_cols_snapshot and clicked_col is not None:
            selected_cols_snapshot = {clicked_col}

        # Set flag to prevent focusOutEvent from clearing selection
        self._showing_context_menu = True

        # Disconnect the selection changed signal to prevent it from firing when menu clears selection
        self.itemSelectionChanged.disconnect(self._on_selection_changed)

        # Always show menu
        menu = QMenu(self)

        if selected_cols_snapshot:
            add_col_action = menu.addAction("Define Column(s)")
            frame_sync_action = menu.addAction("Frame Sync")

            # Determine if selected columns are nibbles (4 bits each)
            is_nibble_selection = False
            if self._all_columns_info and len(selected_cols_snapshot) > 0:
                first_col = min(selected_cols_snapshot)
                if first_col < len(self._all_columns_info):
                    _, bit_start, bit_end, _, _ = self._all_columns_info[first_col]
                    num_bits = bit_end - bit_start + 1
                    if num_bits == 4:
                        is_nibble_selection = True

            # Add Split submenu
            split_menu = menu.addMenu("Split")
            if is_nibble_selection:
                split_binary_action = split_menu.addAction("Binary (4 columns)")
                split_nibble_action = None
            else:
                split_binary_action = split_menu.addAction("Binary (8 columns)")
                split_nibble_action = split_menu.addAction("Nibble (2 columns)")

            # Add Undo Split option
            undo_split_action = menu.addAction("Undo Split")
        else:
            # No selection - show disabled options
            add_col_action = menu.addAction("Define Column(s)")
            add_col_action.setEnabled(False)
            frame_sync_action = menu.addAction("Frame Sync")
            frame_sync_action.setEnabled(False)

            # Add disabled Split submenu
            split_menu = menu.addMenu("Split")
            split_menu.setEnabled(False)
            split_binary_action = split_menu.addAction("Binary (8 columns)")
            split_nibble_action = split_menu.addAction("Nibble (2 columns)")

            # Add disabled Undo Split option
            undo_split_action = menu.addAction("Undo Split")
            undo_split_action.setEnabled(False)

        # Get the global position centered on cursor
        global_pos = self.mapToGlobal(position)

        # Execute menu centered at cursor (Qt will handle menu sizing and positioning)
        action = menu.exec(global_pos)

        # Clear flag now that menu is closed
        self._showing_context_menu = False

        # Reconnect the signal after menu closes
        self.itemSelectionChanged.connect(self._on_selection_changed)

        # If Define Column(s) was clicked and we had a selection (use snapshot)
        if action == add_col_action and selected_cols_snapshot:
            # Temporarily set selected_columns for the dialog
            self.selected_columns = selected_cols_snapshot
            # Open the dialog
            self._add_column_from_selection()
        elif action == frame_sync_action and selected_cols_snapshot:
            # Temporarily set selected_columns for frame sync
            self.selected_columns = selected_cols_snapshot
            # Open frame sync dialog with pattern
            self._frame_sync_from_selection()
        elif action == split_binary_action and selected_cols_snapshot:
            # Split selected columns into binary (8 columns each, or 4 for nibbles)
            print(f"DEBUG: split_binary_action triggered, selected_cols_snapshot = {selected_cols_snapshot}")
            self._split_columns(selected_cols_snapshot, 'binary')
        elif action == split_nibble_action and selected_cols_snapshot:
            # Split selected columns into nibbles (2 columns each)
            self._split_columns(selected_cols_snapshot, 'nibble')
        elif action == undo_split_action and selected_cols_snapshot:
            # Undo split for selected columns
            self._undo_split(selected_cols_snapshot)
        elif action is None:
            # Menu was canceled (clicked outside or ESC) - clear the highlighting
            self.selected_columns.clear()
            self.clearSelection()
            self._update_live_bit_viewer()

        # Trigger repaint to update highlighting
        self.viewport().update()

    def _frame_sync_from_selection(self):
        """Extract bit pattern from selected cells and open Frame Sync dialog with it."""
        if not self.selected_columns or self.parent_window is None:
            return

        # Get all selected items to determine which rows are selected
        selected_items = self.selectedItems()
        if not selected_items:
            return

        # Get unique row indices from selected items (skip header rows 0, 1, 2)
        selected_rows = set()
        for item in selected_items:
            row = self.row(item)
            if row >= 3:  # Only data rows
                selected_rows.add(row)

        if not selected_rows:
            QMessageBox.information(self.parent_window, "Frame Sync",
                                   "Please select data row(s), not header rows.")
            return

        # Use first selected data row
        data_row = min(selected_rows) - 3  # Convert to 0-based data row index

        # Get sorted column indices
        sorted_cols = sorted(self.selected_columns)

        # Extract bits from the selected columns in this row
        if self._all_columns_info and len(self._all_columns_info) > 0:
            start_col_idx = sorted_cols[0]
            end_col_idx = sorted_cols[-1]

            if start_col_idx < len(self._all_columns_info) and end_col_idx < len(self._all_columns_info):
                # Get info for first and last selected columns
                start_byte_idx, start_bit_start, _, _, _ = self._all_columns_info[start_col_idx]
                end_byte_idx, _, end_bit_end, _, _ = self._all_columns_info[end_col_idx]

                # Calculate absolute bit positions within the row
                start_abs_bit = start_byte_idx * 8 + start_bit_start
                end_abs_bit = end_byte_idx * 8 + end_bit_end
                total_bits = end_abs_bit - start_abs_bit + 1

                # Extract bits from the data
                if self.frames is not None and len(self.frames) > data_row:
                    # Framed mode - get bits from this frame
                    frame_bytes = self._get_frame_bytes(data_row)
                    if start_byte_idx < len(frame_bytes):
                        # Unpack bytes to bits
                        frame_bits = np.unpackbits(frame_bytes)
                        # Extract the selected bit range
                        pattern_bits = frame_bits[start_abs_bit:start_abs_bit + total_bits]
                    else:
                        QMessageBox.warning(self.parent_window, "Frame Sync", "Selection is out of range.")
                        return
                else:
                    # Continuous mode - get bits from the row
                    base_byte_idx = data_row * self.row_size
                    absolute_start_byte = base_byte_idx + start_byte_idx
                    absolute_end_byte = base_byte_idx + end_byte_idx

                    if absolute_start_byte >= len(self.bytes_data):
                        QMessageBox.warning(self.parent_window, "Frame Sync", "Selection is out of range.")
                        return

                    # Extract byte slice
                    byte_slice = self.bytes_data[absolute_start_byte:min(absolute_end_byte + 1, len(self.bytes_data))]
                    # Unpack to bits
                    all_bits = np.unpackbits(byte_slice)
                    # Extract the selected bit range
                    pattern_bits = all_bits[start_bit_start:start_bit_start + total_bits]

                # Convert bits to hex string
                # Pad to nibble boundary if needed for clean hex representation
                pattern_bits_list = pattern_bits.tolist()

                # Convert to hex
                hex_str = ""
                if len(pattern_bits_list) % 4 == 0:
                    # Clean nibble alignment - convert to hex
                    for i in range(0, len(pattern_bits_list), 4):
                        nibble = pattern_bits_list[i:i+4]
                        val = (nibble[0] << 3) | (nibble[1] << 2) | (nibble[2] << 1) | nibble[3]
                        hex_str += f"{val:X}"
                    hex_str = "0x" + hex_str
                else:
                    # Not nibble-aligned - show as binary
                    hex_str = "".join(str(b) for b in pattern_bits_list)

                # Call parent's frame_sync with pre-filled pattern
                self.parent_window.frame_sync(prefill_pattern=hex_str)

                # Clear selection after opening dialog
                self.selected_columns.clear()
                self.clearSelection()
                self._update_live_bit_viewer()
                return

        # Fallback for simple byte-based selection (no column definitions)
        start_byte = sorted_cols[0]
        end_byte = sorted_cols[-1]

        if self.frames is not None and len(self.frames) > data_row:
            # Framed mode
            frame_bytes = self._get_frame_bytes(data_row)
            if start_byte < len(frame_bytes):
                byte_slice = frame_bytes[start_byte:min(end_byte + 1, len(frame_bytes))]
            else:
                QMessageBox.warning(self.parent_window, "Frame Sync", "Selection is out of range.")
                return
        else:
            # Continuous mode
            base_byte_idx = data_row * self.row_size
            absolute_start = base_byte_idx + start_byte
            absolute_end = base_byte_idx + end_byte

            if absolute_start >= len(self.bytes_data):
                QMessageBox.warning(self.parent_window, "Frame Sync", "Selection is out of range.")
                return

            byte_slice = self.bytes_data[absolute_start:min(absolute_end + 1, len(self.bytes_data))]

        # Convert to hex string
        hex_str = "0x" + "".join(f"{b:02X}" for b in byte_slice)

        # Call parent's frame_sync with pre-filled pattern
        self.parent_window.frame_sync(prefill_pattern=hex_str)

        # Clear selection after opening dialog
        self.selected_columns.clear()
        self.clearSelection()
        self._update_live_bit_viewer()

    def _add_column_from_selection(self):
        """Create a column definition from selected columns - handles bit-level selections."""
        if not self.selected_columns or self.parent_window is None:
            return

        self._suspend_focus_clear = True
        # Need to check if selection is byte-aligned or bit-level
        sorted_cols = sorted(self.selected_columns)
        try:
            # Check if we have bit-level columns (from _all_columns_info)
            if self._all_columns_info and len(self._all_columns_info) > 0:
                # Analyze all selected columns to find min/max absolute bit positions (MSB-first)
                # Table stores bit positions as 7=MSB..0=LSB; convert to absolute MSB-first index
                abs_start_bits = []
                abs_end_bits = []

                for col_idx in sorted_cols:
                    if col_idx >= len(self._all_columns_info):
                        continue
                    byte_idx, bit_start, bit_end, _, _ = self._all_columns_info[col_idx]

                    # Convert to MSB-first absolute bit indices
                    start_bit_msb = 7 - bit_start
                    end_bit_msb = 7 - bit_end

                    abs_start = byte_idx * 8 + min(start_bit_msb, end_bit_msb)
                    abs_end = byte_idx * 8 + max(start_bit_msb, end_bit_msb)

                    abs_start_bits.append(abs_start)
                    abs_end_bits.append(abs_end)

                if abs_start_bits and abs_end_bits:
                    start_abs_bit = min(abs_start_bits)
                    end_abs_bit = max(abs_end_bits)

                    total_bits = end_abs_bit - start_abs_bit + 1

                    # Check if this spans non-byte-aligned boundaries
                    is_byte_aligned = (start_abs_bit % 8 == 0 and total_bits % 8 == 0)

                    if is_byte_aligned:
                        # Use byte mode
                        start_byte = start_abs_bit // 8
                        end_byte = end_abs_bit // 8
                        if hasattr(self.parent_window, 'add_column_definition_prefilled'):
                            self.parent_window.add_column_definition_prefilled(start_byte, end_byte)
                        else:
                            self.parent_window.add_column_definition()
                    else:
                        # Use bit mode
                        if hasattr(self.parent_window, 'add_column_definition_prefilled_bits'):
                            self.parent_window.add_column_definition_prefilled_bits(start_abs_bit, total_bits)
                        else:
                            # Fallback to undefined bits method
                            self.parent_window.add_definition_from_undefined(start_abs_bit, total_bits)
                    return

            # Fallback: simple byte-based selection (continuous mode with no column definitions)
            start_byte = sorted_cols[0]
            end_byte = sorted_cols[-1]
            if hasattr(self.parent_window, 'add_column_definition_prefilled'):
                self.parent_window.add_column_definition_prefilled(start_byte, end_byte)
            else:
                self.parent_window.add_column_definition()
        finally:
            self._suspend_focus_clear = False

    def _split_columns(self, selected_cols, split_type):
        """Split selected byte columns into sub-columns (binary or nibble).

        Args:
            selected_cols: Set of column indices to split
            split_type: 'binary' for 8 columns (or 4 for nibbles), 'nibble' for 2 columns
        """
        import random

        print(f"DEBUG _split_columns: called with selected_cols={selected_cols}, split_type={split_type}")
        print(f"DEBUG _split_columns: _all_columns_info exists? {self._all_columns_info is not None}")
        if self._all_columns_info:
            for col in selected_cols:
                if col < len(self._all_columns_info):
                    info = self._all_columns_info[col]
                    print(f"DEBUG _split_columns: col {col} info = {info}")

        # Determine the label based on split type
        label = "Binary" if split_type == 'binary' else "Nibble"

        # Pick a random color
        available_colors = [name for name, hex_code in COLOR_OPTIONS if name != "None"]
        random_color = random.choice(available_colors) if available_colors else "None"

        # Map selected column indices to their byte indices or nibble keys
        # IMPORTANT: Store by byte index, not column index, since column indices change after splitting
        if not self._all_columns_info:
            # Simple case: columns are bytes directly
            for col in selected_cols:
                # Store by byte index (which is the same as col in this case)
                self.split_columns[col] = {'type': split_type, 'label': label, 'color': random_color}
        else:
            # Need to map table columns to byte/nibble indices
            for col in selected_cols:
                if col < len(self._all_columns_info):
                    byte_idx, bit_start, bit_end, col_def, is_undef = self._all_columns_info[col]
                    num_bits = bit_end - bit_start + 1

                    # Full byte column - can split into binary (8) or nibble (2)
                    if bit_start == 0 and bit_end == 7 and col_def is None:
                        self.split_columns[byte_idx] = {'type': split_type, 'label': label, 'color': random_color}
                    # Nibble column (4 bits) - can split into binary (4)
                    elif num_bits == 4 and split_type == 'binary' and col_def is None:
                        # Store with a nibble key: (byte_idx, 'high'/'low')
                        nibble_type = 'high' if bit_start == 4 else 'low'
                        nibble_key = (byte_idx, nibble_type)
                        print(f"DEBUG: Splitting nibble - byte_idx={byte_idx}, bits {bit_start}-{bit_end}, nibble_type={nibble_type}, key={nibble_key}")
                        self.split_columns[nibble_key] = {'type': 'nibble_binary', 'label': label, 'color': random_color, 'bit_start': bit_start}
                        print(f"DEBUG: split_columns now = {self.split_columns}")

        # Clear selection and refresh display
        self.selected_columns.clear()
        self.clearSelection()
        print(f"DEBUG _split_columns: split_columns = {self.split_columns}")
        self.update_display()
        self._update_live_bit_viewer()

    def _undo_split(self, selected_cols):
        """Remove split formatting from selected columns.

        Args:
            selected_cols: Set of column indices to unsplit
        """
        # Remove splits for selected columns
        cols_to_remove = []
        for col in self.split_columns.keys():
            if col in selected_cols:
                cols_to_remove.append(col)

        for col in cols_to_remove:
            del self.split_columns[col]

        # Clear selection and refresh display
        self.selected_columns.clear()
        self.clearSelection()
        self.update_display()
        self._update_live_bit_viewer()

    def _resize_columns_for_splits_and_definitions(self, split_labels, field_spans, num_table_cols):
        """Resize columns to be as small as possible for splits and definitions.

        Args:
            split_labels: Dict mapping start column to (span_width, label_text)
            field_spans: List of (start_col, span_len, col_def) tuples
            num_table_cols: Total number of columns
        """
        # Set a uniform default width — resizeColumnToContents on every column is O(rows×cols)
        DEFAULT_COL_WIDTH = 42
        self.horizontalHeader().setDefaultSectionSize(DEFAULT_COL_WIDTH)

        # Resize only definition-spanned columns (typically few)
        for start_col, span_len, col_def in field_spans:
            if col_def is not None:
                # Give definition columns a bit more room
                for i in range(span_len):
                    self.setColumnWidth(start_col + i, max(30, DEFAULT_COL_WIDTH))

        # Narrow split columns
        for start_col, label_info in split_labels.items():
            span_width = label_info[0] if len(label_info) >= 1 else 1
            for i in range(span_width):
                self.setColumnWidth(start_col + i, 24)

    def _find_split_for_column(self, col):
        """Find if a column is part of a split and return the byte index.

        Args:
            col: The table column index to check

        Returns:
            The byte index from split_columns dict, or None if not a split
        """
        if not self._all_columns_info or col >= len(self._all_columns_info):
            return None

        # Get the byte index for this column
        byte_idx, bit_start, bit_end, col_def, is_undef = self._all_columns_info[col]

        # Check if this byte has a split (splits are stored by byte index)
        if byte_idx in self.split_columns:
            # Check if this column is actually part of the split (not a column definition)
            if col_def is None and not is_undef and (bit_start != 0 or bit_end != 7):
                return byte_idx

        return None

    def _edit_split_label(self, byte_idx):
        """Open a dialog to edit the split label and color, or remove the split.

        Args:
            byte_idx: The byte index key in split_columns dict
        """
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, QDialogButtonBox

        if byte_idx not in self.split_columns:
            return

        split_info = self.split_columns[byte_idx]
        current_label = split_info['label']
        current_color = split_info.get('color', 'None')

        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Split")
        layout = QVBoxLayout()

        # Label field
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("Label:"))
        label_edit = QLineEdit(current_label)
        label_layout.addWidget(label_edit)
        layout.addLayout(label_layout)

        # Color field
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Color:"))
        color_combo = QComboBox()
        _populate_color_combo(color_combo, current_color if current_color else "None")
        color_layout.addWidget(color_combo)
        layout.addLayout(color_layout)

        # Remove split button
        remove_button = QPushButton("Remove Split")
        remove_button.clicked.connect(dialog.reject)  # Will use reject to signal removal
        layout.addWidget(remove_button)

        # OK/Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(lambda: dialog.done(2))  # Use code 2 for cancel
        layout.addWidget(button_box)

        dialog.setLayout(layout)

        # Show dialog
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            # Update label and color
            new_label = label_edit.text()
            new_color = color_combo.currentText()
            if new_label:
                self.split_columns[byte_idx]['label'] = new_label
                self.split_columns[byte_idx]['color'] = new_color
                self.update_display()
        elif result == QDialog.DialogCode.Rejected:
            # Remove split
            del self.split_columns[byte_idx]
            self.update_display()

    def _update_live_bit_viewer(self):
        """Update the live bit viewer with bits from selected columns."""
        if self.parent_window is None or not hasattr(self.parent_window, 'live_bit_viewer_canvas'):
            return

        inspector_widget = getattr(self.parent_window, 'field_inspector', None)
        stats_widget = getattr(self.parent_window, 'field_stats_widget', None)

        # Only works in framed mode
        if self.frames is None or len(self.frames) == 0:
            self.parent_window.live_bit_viewer_canvas.set_frame_bits([])
            if inspector_widget:
                inspector_widget.set_frames_bits([])
            if stats_widget:
                stats_widget.set_frames_bits([])
            return

        # If no columns selected, clear the viewer
        if not self.selected_columns:
            self.parent_window.live_bit_viewer_canvas.set_frame_bits([])
            if inspector_widget:
                inspector_widget.set_frames_bits([])
            if stats_widget:
                stats_widget.set_frames_bits([])
            return

        # Extract bits from selected columns for all frames
        sorted_cols = sorted(self.selected_columns)

        if not self._all_columns_info or len(self._all_columns_info) == 0:
            self.parent_window.live_bit_viewer_canvas.set_frame_bits([])
            if inspector_widget:
                inspector_widget.set_frames_bits([])
            if stats_widget:
                stats_widget.set_frames_bits([])
            return

        start_col_idx = sorted_cols[0]
        end_col_idx = sorted_cols[-1]

        if start_col_idx >= len(self._all_columns_info) or end_col_idx >= len(self._all_columns_info):
            return

        # Check if any selected columns belong to a column definition
        # If so, expand selection to include ALL columns of that definition
        expanded_cols = set(sorted_cols)

        for col_idx in sorted_cols:
            if col_idx < len(self._all_columns_info):
                _, _, _, col_def, _ = self._all_columns_info[col_idx]

                if col_def is not None:
                    # This column belongs to a definition - find ALL columns with this same definition
                    for check_idx in range(len(self._all_columns_info)):
                        _, _, _, check_def, _ = self._all_columns_info[check_idx]
                        if check_def is col_def:
                            expanded_cols.add(check_idx)

        # Use expanded selection
        sorted_cols = sorted(expanded_cols)
        start_col_idx = sorted_cols[0]
        end_col_idx = sorted_cols[-1]

        # Extract bits by reading the actual table cell values
        # This shows exactly what's displayed in the table, no byte interpretation
        frames_bits = []

        # Get the number of data rows (skip 2 header rows)
        num_frames = self.rowCount() - 2

        for row_idx in range(num_frames):
            data_row = row_idx + 2  # Skip header rows

            # Collect bits from selected columns by reading cell text
            collected_bits = []
            for col_idx in sorted_cols:
                if col_idx >= self.columnCount():
                    continue

                item = self.item(data_row, col_idx)
                if item is None:
                    continue

                cell_text = item.text().strip()
                if not cell_text or cell_text == "":
                    continue

                # Convert cell value to bits
                # Handle different formats: binary (0/1), hex (0-F), or other
                _, bit_start, bit_end, _, _ = self._all_columns_info[col_idx]
                num_bits = bit_end - bit_start + 1

                if num_bits == 1:
                    # Single bit - cell shows "0" or "1"
                    if cell_text in ["0", "1"]:
                        collected_bits.append(int(cell_text))
                elif cell_text:
                    # Multi-bit field (nibble, byte, etc.) - convert hex to bits
                    try:
                        # Try to parse as hex
                        hex_val = int(cell_text, 16)
                        # Convert to bits (MSB first)
                        for bit_pos in range(num_bits - 1, -1, -1):
                            collected_bits.append((hex_val >> bit_pos) & 1)
                    except (ValueError, TypeError):
                        # If not hex, skip this column
                        pass

            if collected_bits:
                frames_bits.append(np.array(collected_bits, dtype=np.uint8))

        # Update the live bit viewer
        self.parent_window.live_bit_viewer_canvas.set_frame_bits(frames_bits)
        if inspector_widget:
            inspector_widget.set_frames_bits(frames_bits)
        if stats_widget:
            bit_order = getattr(inspector_widget, "bit_order", "msb") if inspector_widget else "msb"
            scale_factor = getattr(inspector_widget, "scale_factor", 1.0) if inspector_widget else 1.0
            stats_widget.set_frames_bits(frames_bits, bit_order=bit_order, scale_factor=scale_factor)


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
        self._invalidate_bit_cache()
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

    def _invalidate_bit_cache(self):
        """Reset cached unpacked bits."""
        self._bit_cache = None
        self._bit_cache_source_id = None
        self._bit_cache_length = 0

    def _ensure_uint8_array(self, data):
        """Return a contiguous np.uint8 array for the given data."""
        if data is None:
            return None
        if isinstance(data, (bytes, bytearray, memoryview)):
            array = np.frombuffer(data, dtype=np.uint8)
            if not array.flags["C_CONTIGUOUS"]:
                array = np.ascontiguousarray(array)
            return array

        array = np.asarray(data, dtype=np.uint8)
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)
        return array

    def _get_contiguous_bytes(self):
        """Return the loaded byte data as a contiguous uint8 numpy array."""
        if self.bytes_data is None:
            return None
        return self._ensure_uint8_array(self.bytes_data)

    def _get_data_bits(self):
        """Return cached unpacked bits for the current byte data."""
        if self.bytes_data is None:
            return None
        source_id = id(self.bytes_data)
        data_len = len(self.bytes_data)
        if (
            self._bit_cache is None
            or self._bit_cache_source_id != source_id
            or self._bit_cache_length != data_len
        ):
            contiguous = self._get_contiguous_bytes()
            if contiguous is None:
                return None
            self._bit_cache = np.unpackbits(contiguous)
            self._bit_cache_source_id = source_id
            self._bit_cache_length = data_len
        return self._bit_cache

    def set_overlay_options(self, show_binary=None, show_ascii=None):
        """Enable/disable byte-level overlay displays."""
        changed = False
        if show_binary is not None and show_binary != self.show_binary_overlay:
            self.show_binary_overlay = show_binary
            changed = True
        if show_ascii is not None and show_ascii != self.show_ascii_overlay:
            self.show_ascii_overlay = show_ascii
            changed = True
        if changed:
            self.update_display()

    def _apply_byte_overlays(self, text, byte_val):
        """Append binary/ASCII lines under a byte's hex text."""
        if byte_val is None or not (self.show_binary_overlay or self.show_ascii_overlay):
            return text, 1

        lines = [text]
        if self.show_binary_overlay:
            lines.append(format(byte_val, '08b'))
        if self.show_ascii_overlay:
            if 32 <= byte_val <= 126:
                lines.append(chr(byte_val))
            else:
                lines.append('.')
        return "\n".join(lines), len(lines)

    def _set_overlay_size_hint(self, item, line_count):
        """Increase row height when multi-line overlays are shown."""
        if line_count <= 1 or item is None:
            return
        metrics = self.fontMetrics()
        height = metrics.lineSpacing() * line_count + 6
        item.setSizeHint(QSize(0, height))

    def _adjust_overlay_column_widths(self, all_columns_info):
        """Ensure byte columns are wide enough to show overlay text."""
        if not (self.show_binary_overlay or self.show_ascii_overlay):
            return

        metrics = self.fontMetrics()
        min_width = 0
        if self.show_binary_overlay:
            min_width = max(min_width, metrics.horizontalAdvance("00000000") + 20)
        if self.show_ascii_overlay:
            min_width = max(min_width, metrics.horizontalAdvance("WW") + 16)
        min_width = max(min_width, 40)

        for col_idx, (_, bit_start, bit_end, col_def, is_undef) in enumerate(all_columns_info):
            if col_def is None and not is_undef and bit_start == 0 and bit_end == 7:
                if self.columnWidth(col_idx) < min_width:
                    self.setColumnWidth(col_idx, min_width)

    def _scan_positions(self, data_array, pattern_array, spacing_bits):
        """
        Generic sliding-window scanner that returns non-overlapping match positions.

        data_array: contiguous np.uint8 array with underlying data (bytes or bits)
        pattern_array: contiguous np.uint8 array with the pattern
        spacing_bits: multiply start index by this many bits (1 for bit mode, 8 for byte mode)
        """
        if data_array is None or pattern_array is None:
            return []

        pattern_len = len(pattern_array)
        if pattern_len == 0 or pattern_len > len(data_array):
            return []

        # Build strided view once; slices reference the original buffer (no data copies)
        stride = data_array.strides[0]
        num_windows = len(data_array) - pattern_len + 1
        windows = as_strided(
            data_array,
            shape=(num_windows, pattern_len),
            strides=(stride, stride),
            writeable=False,
        )

        # Limit block size so block_size * pattern_len stays under PATTERN_SCAN_MAX_ELEMENTS
        max_block = max(1, PATTERN_SCAN_MAX_ELEMENTS // pattern_len)
        positions = []
        hit_limit = False
        prev_bit = -pattern_len * spacing_bits

        for block_start in range(0, num_windows, max_block):
            block = windows[block_start:block_start + max_block]
            if block.size == 0:
                break
            # Broadcast compare within the block and collapse to a single bool per position
            matches = np.all(block == pattern_array, axis=1)
            match_offsets = np.flatnonzero(matches)

            for offset in match_offsets:
                bit_pos = (block_start + offset) * spacing_bits
                if bit_pos >= prev_bit + pattern_len * spacing_bits:
                    positions.append(bit_pos)
                    prev_bit = bit_pos
                    if len(positions) >= MAX_SYNC_FRAMES:
                        hit_limit = True
                        return positions, hit_limit

        return positions, hit_limit

    def _byte_aligned_frame_positions(self, pattern_bytes_array):
        """Find frame positions assuming the pattern is byte-aligned."""
        data_array = self._get_contiguous_bytes()
        if data_array is None:
            return [], False
        pattern_array = self._ensure_uint8_array(pattern_bytes_array)
        return self._scan_positions(data_array, pattern_array, spacing_bits=8)

    def _bit_aligned_frame_positions(self, pattern_bits_array):
        """Find frame positions with bit accuracy."""
        data_bits = self._get_data_bits()
        if data_bits is None:
            return [], False
        pattern_array = self._ensure_uint8_array(pattern_bits_array)
        return self._scan_positions(data_bits, pattern_array, spacing_bits=1)

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

    def set_frame_pattern(self, pattern_list, pattern_bits=None, force_byte_mode=False):
        """
        Enable framing based on a sync pattern.
        pattern_list: list of bytes (for byte-aligned patterns)
        pattern_bits: list of bits (for bit-level patterns)
        Returns number of frames found.
        """

        # Support restoring from saved frame pattern state dictionaries
        if isinstance(pattern_list, dict):
            state = pattern_list
            force_byte_mode = state.get("search_mode") == "byte"
            if pattern_bits is None:
                pattern_bits = state.get("bit_values")
            pattern_list = state.get("byte_values")

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

        pattern_bytes_array = self._ensure_uint8_array(pattern_list) if pattern_list else None

        if pattern_bits is not None:
            pattern_bits_array = self._ensure_uint8_array(pattern_bits)
        elif pattern_bytes_array is not None:
            pattern_bits_array = np.unpackbits(pattern_bytes_array)
        else:
            self.frames = None
            self.frame_pattern = None
            self.update_display()
            return 0

        pattern_len = len(pattern_bits_array)
        if pattern_len == 0:
            self.frames = None
            self.frame_pattern = None
            self.update_display()
            return 0

        byte_aligned = (pattern_len % 8 == 0)
        if byte_aligned and pattern_bytes_array is None:
            pattern_bytes_array = np.packbits(pattern_bits_array)

        use_byte_mode = bool(force_byte_mode and byte_aligned and pattern_bytes_array is not None)

        if use_byte_mode:
            starts_bit_positions, hit_limit = self._byte_aligned_frame_positions(pattern_bytes_array)
            search_mode = "byte"
        else:
            starts_bit_positions, hit_limit = self._bit_aligned_frame_positions(pattern_bits_array)
            search_mode = "bit"

        if hit_limit:
            QMessageBox.information(
                None,
                "Frame Limit",
                f"Found at least {MAX_SYNC_FRAMES:,} frame sync matches. "
                "Processing only the first matches to avoid memory issues.",
            )

        if not starts_bit_positions:
            self.frames = None
            self.frame_pattern = None
            self.update_display()
            return 0

        # Check if first match isn't at bit 0 - this means we're discarding data
        first_bit_offset = starts_bit_positions[0]
        if first_bit_offset > 0:
            # Simple message: just say "first X bits"
            reply = QMessageBox.warning(
                None,
                "Frame Sync Offset",
                f"First sync pattern found at bit {first_bit_offset} (not at start of file).\n\n"
                f"This will discard the first {first_bit_offset} bits of data before the first frame.\n\n"
                f"Continue with frame sync?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                self.frames = None
                self.frame_pattern = None
                return 0

        # CRITICAL: Limit frame creation to prevent crashes
        # Don't even create frame objects for more than we'll display
        total_frames = len(starts_bit_positions)
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

        # LAZY FRAMES: Only store (start_bit, length_bits) tuples!
        # No more unpacking/packing bits or creating huge byte arrays
        frames = []  # Will be list of (start_bit, length_bits) tuples
        create_count = min(total_frames, safe_limit)

        total_bits = len(data) * 8

        for idx in range(create_count):
            start_bit = starts_bit_positions[idx]
            # Frame goes from this pattern start to the next pattern start (or end of data)
            if idx + 1 < len(starts_bit_positions):
                end_bit = starts_bit_positions[idx + 1]
            else:
                end_bit = total_bits  # End of data in bits

            length_bits = end_bit - start_bit
            frames.append((start_bit, length_bits))

        pattern_state = {
            "byte_values": pattern_bytes_array.tolist() if pattern_bytes_array is not None else None,
            "bit_values": pattern_bits_array.tolist(),
            "search_mode": search_mode,
        }

        self.frames = frames
        self.frame_pattern = pattern_state
        self.frame_bit_offsets = None  # No longer needed with lazy frames

        # Update display with lazy frames
        self.update_display()

        return total_frames

    def _get_frame_bytes(self, frame_index):
        """
        LAZY FRAME EXTRACTION: Extract frame bytes on-demand from (start_bit, length_bits) tuple.
        This avoids storing massive frame arrays in memory.
        """
        if self.frames is None or frame_index >= len(self.frames):
            return np.array([], dtype=np.uint8)

        start_bit, length_bits = self.frames[frame_index]

        # Unpack only the bits we need for this frame
        start_byte = start_bit // 8
        # Calculate end byte (need enough bytes to contain start_bit + length_bits)
        end_byte = (start_bit + length_bits + 7) // 8

        # Extract byte slice
        byte_slice = self.bytes_data[start_byte:end_byte]

        # Unpack to bits
        frame_bits = np.unpackbits(byte_slice)

        # Calculate bit offset within first byte
        bit_offset = start_bit % 8

        # Extract exact bits we need
        frame_bits = frame_bits[bit_offset:bit_offset + length_bits]

        # Pad to byte boundary if needed
        if len(frame_bits) % 8 != 0:
            padding_bits = 8 - (len(frame_bits) % 8)
            frame_bits = np.concatenate([frame_bits, np.zeros(padding_bits, dtype=np.uint8)])

        # Pack back to bytes
        frame_bytes = np.packbits(frame_bits)

        return frame_bytes

    def find_pattern_positions(self, pattern_bits, byte_aligned=False):
        """
        Find all positions of a bit pattern without creating frames.
        Returns (list of bit positions, max_frame_length_bytes).
        Used for highlighting patterns before framing.
        """
        if self.bytes_data is None or len(self.bytes_data) == 0:
            return [], 0

        if not pattern_bits:
            return [], 0

        pattern_bits_array = self._ensure_uint8_array(pattern_bits)
        if pattern_bits_array is None or len(pattern_bits_array) == 0:
            return [], 0

        is_byte_aligned = (len(pattern_bits_array) % 8 == 0)

        if byte_aligned and is_byte_aligned:
            pattern_bytes = np.packbits(pattern_bits_array)
            positions, _ = self._byte_aligned_frame_positions(pattern_bytes)
        else:
            positions, _ = self._bit_aligned_frame_positions(pattern_bits_array)

        # Calculate max frame length
        max_frame_len = 0
        if positions:
            data_len_bits = len(self.bytes_data) * 8 if self.bytes_data is not None else 0
            for i in range(len(positions)):
                if i + 1 < len(positions):
                    # Frame length from this position to next
                    frame_bits = positions[i + 1] - positions[i]
                    frame_bytes = (frame_bits + 7) // 8  # Round up
                    max_frame_len = max(max_frame_len, frame_bytes)
                else:
                    # Last frame: from this position to end of data
                    frame_bits = max(0, data_len_bits - positions[i])
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
            # LAZY FRAMES: calculate max length from frame tuples
            max_len = 0
            for _, length_bits in self.frames:
                frame_bytes = (length_bits + 7) // 8  # Round up to bytes
                max_len = max(max_len, frame_bytes)

            _, _, _, _, all_columns_info, _, _ = self._build_headers(max_len)

            # Check each table column for constant values
            for table_col_idx, (byte_idx, bit_start, bit_end, col_def, is_undef) in enumerate(all_columns_info):
                val_text = None
                constant = True

                for frame_idx in range(len(self.frames)):
                    # LAZY: Extract frame bytes on-demand
                    frame = self._get_frame_bytes(frame_idx)

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
            _, _, _, _, all_columns_info, _, _ = self._build_headers(self.row_size)

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
        if not name or name == "None":
            return None
        if name in COLOR_NAME_TO_QCOLOR:
            return COLOR_NAME_TO_QCOLOR[name]

        # Backwards compatibility with legacy names
        legacy_map = {
            "Yellow": QColor(255, 255, 180),
            "Cyan": QColor(180, 255, 255),
            "Magenta": QColor(255, 180, 255),
            "Green": QColor(200, 255, 200),
            "lightblue": QColor(180, 220, 255),
            "lightgray": QColor(220, 220, 220),
        }
        return legacy_map.get(name, None)

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
        byte_to_original_table_col = {}  # Map byte_idx to the original table column before splitting

        for byte_idx in range(num_cols):
            # Record the starting table column for this byte (before any splitting)
            byte_to_original_table_col[byte_idx] = table_col

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

        # Now handle split columns - we need to expand columns that should be split
        # This requires rebuilding all_columns_info with splits applied
        split_all_columns_info = []
        split_col_map = {}
        split_table_col = 0
        split_labels = {}  # Map from starting table_col to (span_width, label_text)

        for byte_idx in range(num_cols):
            # Check if this byte should be split (look up by BYTE INDEX)
            split_info = self.split_columns.get(byte_idx, None)

            if split_info and byte_idx in byte_ranges:
                # Check if this byte is a simple full-byte column (not already bit-split)
                byte_range_list = byte_ranges[byte_idx]
                is_simple_byte = (len(byte_range_list) == 1 and
                                 byte_range_list[0][0] == 0 and
                                 byte_range_list[0][1] == 7 and
                                 byte_range_list[0][2] is None)  # No column definition

                if is_simple_byte:
                    # Apply split
                    split_col_map[byte_idx] = []
                    start_col = split_table_col
                    split_type = split_info['type']
                    split_label = split_info['label']
                    split_color = split_info.get('color', 'None')

                    if split_type == 'binary':
                        # Split into 8 bit columns (MSB first: bit 7 down to bit 0)
                        print(f"DEBUG: Creating binary split for byte {byte_idx}, bits 7->0")
                        for bit in range(7, -1, -1):
                            split_col_map[byte_idx].append((split_table_col, (bit, bit), None, False))
                            split_all_columns_info.append((byte_idx, bit, bit, None, False))
                            print(f"  Column {split_table_col}: bit {bit}")
                            split_table_col += 1
                        split_labels[start_col] = (8, split_label, split_color)
                    elif split_type == 'nibble':
                        # Split into 2 nibble columns (high nibble first, then low nibble - MSB first).
                        # If a nibble itself was further split into bits, expand it to 4 bit columns here.
                        nibble_binary_present = False
                        unsplit_nibbles = []  # Track nibble start columns when one nibble is binary-split

                        for nibble_type, (nibble_start, nibble_end) in (('high', (4, 7)), ('low', (0, 3))):
                            nibble_key = (byte_idx, nibble_type)
                            nibble_split_info = self.split_columns.get(nibble_key, None)
                            nibble_start_col = split_table_col

                            if nibble_split_info and nibble_split_info.get('type') == 'nibble_binary':
                                nibble_binary_present = True
                                bit_start_val = nibble_split_info['bit_start']
                                split_label_bits = nibble_split_info['label']
                                split_color_bits = nibble_split_info.get('color', 'None')

                                for bit_offset in range(3, -1, -1):
                                    bit_pos = bit_start_val + bit_offset
                                    split_col_map[byte_idx].append((split_table_col, (bit_pos, bit_pos), None, False))
                                    split_all_columns_info.append((byte_idx, bit_pos, bit_pos, None, False))
                                    split_table_col += 1
                                split_labels[nibble_start_col] = (4, split_label_bits, split_color_bits)
                            else:
                                split_col_map[byte_idx].append((split_table_col, (nibble_start, nibble_end), None, False))
                                split_all_columns_info.append((byte_idx, nibble_start, nibble_end, None, False))
                                unsplit_nibbles.append(nibble_start_col)
                                split_table_col += 1

                        # If any nibble was split into bits, label remaining single-nibble columns individually
                        if nibble_binary_present:
                            for start in unsplit_nibbles:
                                split_labels[start] = (1, split_label, split_color)
                        else:
                            # No nibble-level split - behave like the original 2-column nibble split
                            total_span = split_table_col - start_col
                            split_labels[start_col] = (total_span, split_label, split_color)
                    continue

            # No byte-level split - check for nibble-level splits or copy original columns
            if byte_idx in col_map:
                split_col_map[byte_idx] = []
                for _, (start_in_byte, end_in_byte), col_def, is_undef in col_map[byte_idx]:
                    # Check if this nibble should be split into bits
                    nibble_type = 'high' if start_in_byte == 4 else ('low' if start_in_byte == 0 and end_in_byte == 3 else None)
                    nibble_key = (byte_idx, nibble_type) if nibble_type else None
                    nibble_split_info = self.split_columns.get(nibble_key, None) if nibble_key else None

                    if nibble_split_info and nibble_split_info['type'] == 'nibble_binary':
                        # Split this nibble into 4 bits (MSB first)
                        start_col = split_table_col
                        split_label = nibble_split_info['label']
                        split_color = nibble_split_info.get('color', 'None')
                        bit_start_val = nibble_split_info['bit_start']

                        for bit_offset in range(3, -1, -1):
                            bit_pos = bit_start_val + bit_offset
                            split_col_map[byte_idx].append((split_table_col, (bit_pos, bit_pos), None, False))
                            split_all_columns_info.append((byte_idx, bit_pos, bit_pos, None, False))
                            split_table_col += 1
                        split_labels[start_col] = (4, split_label, split_color)
                    else:
                        # No nibble split - copy as is
                        split_col_map[byte_idx].append((split_table_col, (start_in_byte, end_in_byte), col_def, is_undef))
                        split_all_columns_info.append((byte_idx, start_in_byte, end_in_byte, col_def, is_undef))
                        split_table_col += 1

        # Use the split versions
        all_columns_info = split_all_columns_info
        col_map = split_col_map

        # Build label spans, byte labels, and sub-bit labels
        label_spans = []
        byte_labels = []
        subbit_labels = []
        byte_label_spans = []  # New: spans for byte labels

        current_label = None
        current_color = "None"
        current_start = None
        current_len = 0

        # Track byte label spans
        current_byte_idx = None
        current_byte_start = None
        current_byte_len = 0

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

            # Byte label: track spans for same byte_idx
            if byte_idx == current_byte_idx:
                # Same byte - extend span
                current_byte_len += 1
            else:
                # Different byte - flush previous span
                if current_byte_idx is not None:
                    byte_label_spans.append((current_byte_start, current_byte_len, str(current_byte_idx)))
                # Start new byte span
                current_byte_idx = byte_idx
                current_byte_start = idx
                current_byte_len = 1

            byte_labels.append(f"{byte_idx}")

            # Sub-bit label: show bit range within that byte
            subbit_labels.append(f"{bit_start}-{bit_end}")

        # Flush trailing label span
        if current_label is not None:
            label_spans.append((current_start, current_len, current_label, current_color))

        # Flush trailing byte span
        if current_byte_idx is not None:
            byte_label_spans.append((current_byte_start, current_byte_len, str(current_byte_idx)))

        # Add split labels to label_spans
        for start_col, label_info in split_labels.items():
            if len(label_info) == 3:
                span_width, label_text, color = label_info
            else:
                span_width, label_text = label_info
                color = "None"
            label_spans.append((start_col, span_width, label_text, color))

        return col_map, label_spans, byte_labels, subbit_labels, all_columns_info, byte_label_spans, split_labels

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

    def _populate_visible_rows(self):
        """Fill only the data rows currently visible in the viewport (plus a small buffer)."""
        if self._display_meta is None:
            return

        meta = self._display_meta
        all_columns_info  = meta['all_columns_info']
        field_spans       = meta['field_spans']
        first_col_for_def = meta['first_col_for_def']
        num_data_rows     = meta['num_data_rows']
        is_continuous     = meta.get('mode') == 'continuous'
        max_len           = meta.get('max_len')  # framed mode only

        vp_height = self.viewport().height()
        first_vis = self.rowAt(0)
        last_vis  = self.rowAt(vp_height - 1)
        if first_vis < 0:
            first_vis = 3
        if last_vis < 0:
            last_vis = self.rowCount() - 1

        BUFFER = 10
        row_start = max(3, first_vis - BUFFER)
        row_end   = min(self.rowCount() - 1, last_vis + BUFFER)

        self.setUpdatesEnabled(False)
        for table_row in range(row_start, row_end + 1):
            data_row_idx = table_row - 3
            if data_row_idx < 0 or data_row_idx >= num_data_rows:
                continue
            if data_row_idx in self._populated_data_rows:
                continue
            if is_continuous:
                self._populate_continuous_data_row(data_row_idx, table_row,
                                                   all_columns_info, field_spans,
                                                   first_col_for_def)
            else:
                self._populate_data_row(data_row_idx, table_row, all_columns_info,
                                        field_spans, first_col_for_def, max_len)
            self._populated_data_rows.add(data_row_idx)
        self.setUpdatesEnabled(True)

    def _populate_data_row(self, data_row_idx, table_row, all_columns_info,
                           field_spans, first_col_for_def, max_len):
        """Fill one data row of the table."""
        frame = self._get_frame_bytes(data_row_idx)
        flen = len(frame)

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
            line_count = 1

            if col_def is not None:
                if col_def.unit == "bit":
                    if table_col_idx == first_col_for_def.get(col_def, table_col_idx):
                        start_byte = col_def.start_byte
                        end_byte = col_def.end_byte
                        if start_byte < flen:
                            byte_slice = frame[start_byte:min(end_byte + 1, flen)]
                            text = self._format_bit_range(byte_slice, col_def)
                            tooltip = "{} (bit {}-{}, {} bits, {})".format(
                                col_def.label, col_def.start_bit,
                                col_def.start_bit + col_def.total_bits - 1,
                                col_def.total_bits, col_def.display_format)
                elif col_def.unit == "byte":
                    if table_col_idx == first_col_for_def.get(col_def, table_col_idx):
                        start_byte = col_def.start_byte
                        end_byte = col_def.end_byte
                        if start_byte < flen:
                            byte_slice = frame[start_byte:min(end_byte + 1, flen)]
                            text = self._format_multi_byte_value(byte_slice, col_def)
                            tooltip = "{} ({})".format(col_def.label, col_def.display_format) if col_def.label else ""
            elif is_undef:
                abs_start_bit = byte_idx * 8 + bit_start
                num_bits = bit_end - bit_start + 1
                text = self._format_undefined_bits(frame[byte_idx:byte_idx + 1], abs_start_bit, num_bits)
                tooltip = f"{num_bits} undefined bits (bit {abs_start_bit}-{abs_start_bit + num_bits - 1})"
            else:
                if bit_start != 0 or bit_end != 7:
                    abs_start_bit = byte_idx * 8 + bit_start
                    num_bits = bit_end - bit_start + 1
                    if num_bits == 1:
                        text = str((byte_val >> bit_end) & 1)
                    elif num_bits == 4:
                        nibble_val = (byte_val >> 4) & 0x0F if bit_start == 4 else byte_val & 0x0F
                        text = "{:X}".format(nibble_val)
                    else:
                        extracted_val = self._extract_bits_from_bytes(frame[byte_idx:byte_idx + 1], bit_start, num_bits)
                        text = "{:X}".format(extracted_val)
                else:
                    text = "{:02X}".format(byte_val)
                    if not is_undef and bit_start == 0 and bit_end == 7:
                        text, line_count = self._apply_byte_overlays(text, byte_val)

            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if tooltip and text:
                item.setToolTip(tooltip)
            self._set_overlay_size_hint(item, line_count)

            if table_col_idx in self.constant_columns and text != "":
                item.setBackground(QColor(255, 255, 180))
            elif col_def is not None and col_def.color_name and col_def.color_name != "None":
                col_color = self._color_from_name(col_def.color_name)
                if col_color is not None and text != "":
                    item.setBackground(col_color)
            elif byte_idx in self.split_columns:
                split_color_name = self.split_columns[byte_idx].get('color', 'None')
                split_color = self._color_from_name(split_color_name)
                if split_color is not None and text != "":
                    item.setBackground(split_color)

            self.setItem(table_row, table_col_idx, item)

    def _populate_continuous_data_row(self, data_row_idx, table_row, all_columns_info,
                                      field_spans, first_col_for_def):
        """Fill one data row for continuous (non-framed) byte mode."""
        data = self.bytes_data
        num_bytes = len(data)
        base_byte_idx = data_row_idx * self.row_size

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
            line_count = 1

            if col_def is not None:
                if col_def.unit == "bit":
                    if table_col_idx == first_col_for_def.get(col_def, table_col_idx):
                        start_byte = col_def.start_byte + base_byte_idx
                        end_byte = col_def.end_byte + base_byte_idx
                        if start_byte < num_bytes:
                            byte_slice = data[start_byte:min(end_byte + 1, num_bytes)]
                            text = self._format_bit_range(byte_slice, col_def)
                            tooltip = "{} (bit {}-{}, {} bits, {})".format(
                                col_def.label, col_def.start_bit,
                                col_def.start_bit + col_def.total_bits - 1,
                                col_def.total_bits, col_def.display_format)
                elif col_def.unit == "byte":
                    if table_col_idx == first_col_for_def.get(col_def, table_col_idx):
                        start_byte = col_def.start_byte + base_byte_idx
                        end_byte = col_def.end_byte + base_byte_idx
                        if start_byte < num_bytes:
                            byte_slice = data[start_byte:min(end_byte + 1, num_bytes)]
                            text = self._format_multi_byte_value(byte_slice, col_def)
                            tooltip = "{} ({})".format(col_def.label, col_def.display_format) if col_def.label else ""
            elif is_undef:
                abs_start_bit = absolute_byte_idx * 8 + bit_start
                num_bits = bit_end - bit_start + 1
                text = self._format_undefined_bits(data[absolute_byte_idx:absolute_byte_idx + 1], abs_start_bit, num_bits)
                tooltip = f"{num_bits} undefined bits (bit {abs_start_bit}-{abs_start_bit + num_bits - 1})"
            else:
                if bit_start != 0 or bit_end != 7:
                    num_bits = bit_end - bit_start + 1
                    if num_bits == 1:
                        text = str((byte_val >> (7 - bit_end)) & 1)
                    elif num_bits == 4:
                        nibble_val = (byte_val >> 4) & 0x0F if bit_start == 4 else byte_val & 0x0F
                        text = "{:X}".format(nibble_val)
                    else:
                        extracted_val = self._extract_bits_from_bytes(
                            data[absolute_byte_idx:absolute_byte_idx + 1], bit_start, num_bits)
                        text = "{:X}".format(extracted_val)
                else:
                    text = "{:02X}".format(byte_val)
                    text, line_count = self._apply_byte_overlays(text, byte_val)

            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if tooltip and text:
                item.setToolTip(tooltip)
            self._set_overlay_size_hint(item, line_count)

            is_pattern_highlighted = self._is_byte_in_pattern_highlight(absolute_byte_idx)

            if is_pattern_highlighted and text != "":
                item.setBackground(QColor(255, 200, 0))
            elif table_col_idx in self.constant_columns and text != "":
                item.setBackground(QColor(255, 255, 180))
            elif col_def is not None and col_def.color_name and col_def.color_name != "None":
                col_color = self._color_from_name(col_def.color_name)
                if col_color is not None and text != "":
                    item.setBackground(col_color)
            elif byte_offset in self.split_columns:
                split_color_name = self.split_columns[byte_offset].get('color', 'None')
                split_color = self._color_from_name(split_color_name)
                if split_color is not None and text != "":
                    item.setBackground(split_color)

            self.setItem(table_row, table_col_idx, item)

    def update_display(self):
        """Render table either in continuous mode or framed mode with three header rows:
           row 0: labels, row 1: byte index, row 2: sub-byte bit ranges.
           For any ColumnDefinition that spans multiple bytes, both the header
           and the data cell are merged so you only see one value.
        """

        # If nothing loaded, clear table cleanly
        if self.bytes_data is None:
            self._display_meta = None
            self._populated_data_rows = set()
            self.setRowCount(0)
            self.setColumnCount(0)
            return

        # Clear virtual row state before rebuilding
        self._display_meta = None
        self._populated_data_rows = set()

        # CRITICAL: Disable updates during table construction to prevent crashes
        self.setUpdatesEnabled(False)

        # Clear existing spans for a fresh layout
        self.clearSpans()

        # =====================================================
        # ================  FRAMED MODE  ======================
        # =====================================================
        if self.frames is not None and len(self.frames) > 0:
            # LAZY FRAMES: frames is now list of (start_bit, length_bits) tuples
            total_data_rows = len(self.frames)
            num_data_rows = min(total_data_rows, self.max_display_rows)

            # Calculate max frame length from tuples
            frame_lengths = [(length_bits + 7) // 8
                             for _, length_bits in self.frames[:num_data_rows]]
            max_len = max(frame_lengths) if frame_lengths else 0

            # When the last frame is anomalously long (e.g. zero-padded tail with no
            # further sync matches), don't let it inflate the column count.
            # Use the max of all non-last frames if the last frame is 10× longer.
            if len(frame_lengths) > 1:
                non_last_max = max(frame_lengths[:-1])
                if max_len > non_last_max * 10 and max_len > 64:
                    max_len = non_last_max

            # Always cap columns to prevent freeze regardless of file size
            if max_len > self.max_display_cols:
                max_len = self.max_display_cols

            # Build the column structure
            col_map, label_spans, byte_labels, subbit_labels, all_columns_info, byte_label_spans, split_labels = self._build_headers(max_len)
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
            # First, set all cells with empty items
            for col_idx in range(num_table_cols):
                item = QTableWidgetItem("")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setBackground(QColor(240, 240, 240))
                self.setItem(1, col_idx, item)

            # Then set the byte labels with spans
            for table_col, span_width, byte_label in byte_label_spans:
                item = QTableWidgetItem(byte_label)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setBackground(QColor(240, 240, 240))
                item.setFont(QFont("Courier", 8))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.setItem(1, table_col, item)

                if span_width > 1:
                    # Span across multiple columns for split bytes
                    self.setSpan(1, table_col, 1, span_width)

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


            # Pre-compute row labels from frame tuple lengths (no byte extraction needed)
            for data_row_idx in range(num_data_rows):
                _, length_bits = self.frames[data_row_idx]
                flen = (length_bits + 7) // 8
                if flen > max_len:
                    row_labels.append(f"{data_row_idx} ({flen}→{max_len})")
                else:
                    row_labels.append(f"{data_row_idx} ({flen})")
            self.setVerticalHeaderLabels(row_labels)

            # Store metadata for virtual/lazy row population
            self._display_meta = {
                'all_columns_info': all_columns_info,
                'field_spans': field_spans,
                'first_col_for_def': first_col_for_def,
                'max_len': max_len,
                'num_data_rows': num_data_rows,
            }
            self._populated_data_rows = set()
            # OPTIMIZED: Don't resize on every update - too slow with many columns
            # User can manually resize if needed

            # Resize columns for split columns and defined columns
            self._resize_columns_for_splits_and_definitions(split_labels, field_spans, num_table_cols)
            self._adjust_overlay_column_widths(all_columns_info)

            self.setUpdatesEnabled(True)  # Re-enable updates before returning

            # Populate only the visible rows — remaining rows filled on scroll
            self._populate_visible_rows()

            # Apply column highlighting if any columns are selected
            if self.selected_columns:
                self._apply_selection_highlighting()

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
        col_map, label_spans, byte_labels, subbit_labels, all_columns_info, byte_label_spans, split_labels = self._build_headers(self.row_size)
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
        # First, set all cells with empty items
        for col_idx in range(num_table_cols):
            item = QTableWidgetItem("")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setBackground(QColor(240, 240, 240))
            self.setItem(1, col_idx, item)

        # Then set the byte labels with spans
        for table_col, span_width, byte_label in byte_label_spans:
            item = QTableWidgetItem(byte_label)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setBackground(QColor(240, 240, 240))
            item.setFont(QFont("Courier", 8))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(1, table_col, item)

            if span_width > 1:
                # Span across multiple columns for split bytes
                self.setSpan(1, table_col, 1, span_width)

        # === Header row 2: bit ranges ===
        for col_idx, sub_label in enumerate(subbit_labels):
            item = QTableWidgetItem(sub_label)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setBackground(QColor(250, 250, 250))
            item.setFont(QFont("Courier", 8))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(2, col_idx, item)

        # === Data rows — lazy/virtual (only visible rows populated on load) ===
        row_labels = ["Labels", "Bytes", "Bits"]

        if num_data_rows < total_data_rows:
            row_labels[0] += f" (showing {num_data_rows}/{total_data_rows})"

        for data_row_idx in range(num_data_rows):
            row_labels.append(str(data_row_idx * self.row_size))

        self.setVerticalHeaderLabels(row_labels)

        self._display_meta = {
            'mode': 'continuous',
            'all_columns_info': all_columns_info,
            'field_spans': field_spans,
            'first_col_for_def': first_col_for_def,
            'num_data_rows': num_data_rows,
        }
        self._populated_data_rows = set()

        self._resize_columns_for_splits_and_definitions(split_labels, field_spans, num_table_cols)
        self._adjust_overlay_column_widths(all_columns_info)
        self.setUpdatesEnabled(True)
        self._populate_visible_rows()

        if self.selected_columns:
            self._apply_selection_highlighting()
        return

