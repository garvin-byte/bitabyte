"""BitViewerWindow — main application window."""

import sys
import copy
import numpy as np
import re
import math
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
                             QFileDialog, QScrollArea, QMessageBox, QComboBox,
                             QLineEdit, QListWidget, QGroupBox, QCheckBox,
                             QSplitter, QTextEdit, QDialog, QDialogButtonBox,
                             QInputDialog, QListWidgetItem, QRadioButton, QButtonGroup,
                             QTableWidget, QTableWidgetItem, QMenu, QStyledItemDelegate,
                             QDockWidget, QSizePolicy)
from PyQt6.QtCore import Qt, QRect, QSize, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPalette, QFont, QIcon, QPixmap
from .colors import (COLOR_OPTIONS, _populate_color_combo,
                     BIT_SYNC_WARNING_BYTES, BIT_SYNC_HARD_LIMIT_BYTES,
                     MAX_SYNC_FRAMES, PATTERN_SCAN_MAX_ELEMENTS)
from .canvas import BitCanvas, LiveBitViewerCanvas
from .widgets import LogScaleSpinBox, FieldInspectorWidget, FieldStatisticsWidget, TextDisplayWidget
from .column import ColumnDefinition, AddColumnDialog
from .table import ByteStructuredTableWidget




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
        self.field_inspector = None
        self.field_stats_widget = None

        # Mode
        self.data_mode = "bit"  # "bit" or "byte"

        # Used to restore framed byte-mode layout after visiting bit mode
        self._return_to_byte_framed = False
        self._saved_byte_frame_pattern = None

        # Undo/Redo stacks
        self.undo_stack = []
        self.redo_stack = []

        self.init_ui()
        self.apply_theme()

    def eventFilter(self, obj, event):
        """Filter events to detect clicks outside byte_table for clearing column selection."""
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QMouseEvent
        from PyQt6.QtWidgets import QMenu

        if event.type() == QEvent.Type.MouseButtonPress:
            if isinstance(event, QMouseEvent):
                # Check if byte_table exists and has column selection
                if hasattr(self, 'byte_table') and self.byte_table.selected_columns:
                    # Don't clear selection if clicking on a menu
                    widget_under_mouse = self.childAt(obj.mapTo(self, event.pos()))
                    if isinstance(widget_under_mouse, QMenu):
                        return super().eventFilter(obj, event)

                    # Get the global position of the click
                    global_pos = event.globalPosition().toPoint()

                    # Check if click is outside the byte_table widget
                    table_rect = self.byte_table.rect()
                    table_global_pos = self.byte_table.mapToGlobal(table_rect.topLeft())
                    table_global_rect = table_rect.translated(table_global_pos)

                    if not table_global_rect.contains(global_pos):
                        # Click is outside table - clear selection
                        for col in self.byte_table.selected_columns:
                            self.byte_table._restore_column_colors(col)
                        self.byte_table.selected_columns.clear()
                        self.byte_table.clearSelection()
                        self.byte_table._update_live_bit_viewer()

        return super().eventFilter(obj, event)

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

        # Create Live Bit Viewer dock widget
        self.live_bit_viewer_dock = QDockWidget("Live Bit Viewer", self)
        self.live_bit_viewer_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.BottomDockWidgetArea
        )

        # Create live viewer and field inspector stacked vertically via splitter
        live_viewer_container = QWidget()
        live_viewer_layout = QVBoxLayout(live_viewer_container)
        live_viewer_layout.setContentsMargins(0, 0, 0, 0)
        live_viewer_layout.setSpacing(4)

        live_splitter = QSplitter(Qt.Orientation.Vertical)
        live_viewer_layout.addWidget(live_splitter)

        live_viewer_scroll = QScrollArea()
        live_viewer_scroll.setWidgetResizable(True)
        live_viewer_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        live_viewer_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.live_bit_viewer_canvas = LiveBitViewerCanvas()
        live_viewer_scroll.setWidget(self.live_bit_viewer_canvas)
        live_splitter.addWidget(live_viewer_scroll)

        inspector_container = QGroupBox("Field Inspector")
        inspector_layout = QVBoxLayout(inspector_container)
        inspector_layout.setContentsMargins(6, 6, 6, 6)

        self.field_inspector = FieldInspectorWidget()
        self.field_inspector.parent_window = self
        inspector_layout.addWidget(self.field_inspector)
        live_splitter.addWidget(inspector_container)

        stats_container = QGroupBox("Field Statistics")
        stats_layout = QVBoxLayout(stats_container)
        stats_layout.setContentsMargins(6, 6, 6, 6)

        self.field_stats_widget = FieldStatisticsWidget()
        stats_layout.addWidget(self.field_stats_widget)
        live_splitter.addWidget(stats_container)

        live_splitter.setStretchFactor(0, 1)
        live_splitter.setStretchFactor(1, 1)
        live_splitter.setStretchFactor(2, 1)
        live_splitter.setSizes([250, 250, 250])

        self.live_bit_viewer_dock.setWidget(live_viewer_container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.live_bit_viewer_dock)

        # Set preferred size for the dock widget (wider) and let it grow vertically
        self.live_bit_viewer_dock.setMinimumWidth(350)

        # Hide by default - will be shown when user switches to byte mode with frames
        self.live_bit_viewer_dock.hide()

    def _restore_byte_frames_from_bit_layout(self, frame_width_bits, total_frame_bits):
        """Restore framed byte-mode rows from a bit-mode layout width."""
        if frame_width_bits <= 0:
            self.byte_table.frames = None
            self.byte_table.frame_pattern = None
            self.byte_table.frame_bit_offsets = None
            self.row_size_spin.setEnabled(True)
            return

        restored_frames = []
        for start_bit in range(0, total_frame_bits, frame_width_bits):
            length_bits = min(frame_width_bits, total_frame_bits - start_bit)
            if length_bits > 0:
                restored_frames.append((start_bit, length_bits))

        self.byte_table.frames = restored_frames if restored_frames else None
        self.byte_table.frame_pattern = copy.deepcopy(self._saved_byte_frame_pattern)
        self.byte_table.frame_bit_offsets = None
        self.row_size_spin.setEnabled(bool(restored_frames))
        if restored_frames:
            self.row_size_spin.setEnabled(False)
        else:
            self.row_size_spin.setEnabled(True)

    def _refresh_byte_mode_view(self):
        """Refresh byte mode without destroying existing frame state."""
        if self.bytes_data is None:
            return

        preserve_frames = self.byte_table.frames is not None and len(self.byte_table.frames) > 0
        self.scroll_area.hide()
        self.text_display.hide()
        self.byte_table.show()

        self.byte_table.bytes_data = self.bytes_data
        if hasattr(self.byte_table, '_invalidate_bit_cache'):
            self.byte_table._invalidate_bit_cache()
        self.byte_table.update_display()

        if preserve_frames:
            self.byte_table._update_live_bit_viewer()
            if hasattr(self, 'live_bit_viewer_dock'):
                self.live_bit_viewer_dock.show()
        else:
            if hasattr(self, 'live_bit_viewer_dock'):
                self.live_bit_viewer_dock.hide()
                self.live_bit_viewer_canvas.set_frame_bits([])
            if getattr(self, 'field_inspector', None):
                self.field_inspector.set_frames_bits([])
            if getattr(self, 'field_stats_widget', None):
                self.field_stats_widget.set_frames_bits([])

    def on_mode_changed(self):
        """Handle mode change between Bit and Byte."""
        try:
            previous_mode = self.data_mode
            new_mode = "bit" if self.bit_mode_radio.isChecked() else "byte"

            if new_mode == previous_mode:
                return
        except Exception as e:
            print(f"ERROR in on_mode_changed (initial check): {e}")
            import traceback
            traceback.print_exc()
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
            # Since we're using original bytes in BIT mode, just pack back to bytes
            # Frames should still be valid!
            self.bytes_data = np.packbits(bits_to_save)
            self.byte_table.bytes_data = self.bytes_data
            # Frames are still there from BYTE mode - nothing to restore!

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

            if self._return_to_byte_framed and frame_width_bits > 0:
                self._restore_byte_frames_from_bit_layout(frame_width_bits, len(bits_to_save))
            else:
                self.byte_table.frames = None
                self.byte_table.frame_pattern = None
                self.byte_table.frame_bit_offsets = None
                self.row_size_spin.setEnabled(True)

            # Clear bit operations when switching to byte mode
            if self.operations:
                self.operations = []
                self.operations_list.clear()
                self.undo_stack.clear()
                self.redo_stack.clear()
                self.clear_highlights()

        # BYTE -> BIT
        elif previous_mode == "byte" and new_mode == "bit":
            try:
                self._return_to_byte_framed = False
                self._saved_byte_frame_pattern = None
                if self.bytes_data is not None:
                    # If framed, ask user if they want to zero-pad
                    if self.byte_table.frames is not None and len(self.byte_table.frames) > 0:
                        self._saved_byte_frame_pattern = copy.deepcopy(self.byte_table.frame_pattern)
                        # Extract all frames using the lazy _get_frame_bytes method
                        frame_count = len(self.byte_table.frames)

                        # Extract actual byte arrays from lazy frame tuples
                        extracted_frames = []
                        for i in range(frame_count):
                            frame_bytes = self.byte_table._get_frame_bytes(i)
                            if len(frame_bytes) > 0:
                                extracted_frames.append(frame_bytes)

                        if not extracted_frames:
                            work_bytes = self.bytes_data
                        else:
                            # Get frame lengths
                            frame_lengths = [len(f) for f in extracted_frames]
                            max_len = max(frame_lengths)
                            min_len = min(frame_lengths)

                            # Only ask if frames have different lengths
                            if min_len < max_len:
                                reply = QMessageBox.question(
                                    self,
                                    "Zero-Pad Frames?",
                                    f"You have {len(extracted_frames)} frames with varying lengths.\n\n"
                                    f"Min length: {min_len:,} bytes\n"
                                    f"Max length: {max_len:,} bytes\n\n"
                                    f"Zero-pad all frames to max length ({max_len:,} bytes)?\n\n"
                                    f"Yes = Pad frames and set width to {max_len * 8:,} bits\n"
                                    f"No = Use original data without padding",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                                )
                                use_padding = (reply == QMessageBox.StandardButton.Yes)
                            else:
                                # All frames same length, automatically pad (even if all same length)
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
                                for i, f in enumerate(extracted_frames):
                                    if len(f) < max_len:
                                        pad = np.zeros(max_len - len(f), dtype=np.uint8)
                                        padded_f = np.concatenate([f, pad])
                                    else:
                                        padded_f = f
                                    padded_frames.append(padded_f)

                                # Concatenate all padded frames into one byte array for bit mode
                                work_bytes = np.concatenate(padded_frames)
                                self._return_to_byte_framed = True
                            else:
                                # No padding - just concatenate frames as-is
                                work_bytes = np.concatenate(extracted_frames)
                    else:
                        # No framing: just use the raw bytes
                        work_bytes = self.bytes_data

                # Unpack to bits for Bit mode
                # Ensure work_bytes is uint8 (required by unpackbits)
                if work_bytes.dtype != np.uint8:
                    work_bytes = work_bytes.astype(np.uint8)

                self.bits = np.unpackbits(work_bytes)
                self.original_bits = self.bits.copy()

                # Update file info label with new (possibly zero-padded) counts
                bit_count = len(self.bits)
                byte_count = len(work_bytes)
                self.file_info_label.setText(
                    f"File: {self.filename.split('/')[-1] if self.filename else 'N/A'}\n"
                    f"Size: {byte_count} bytes ({bit_count} bits)\n"
                    f"Ones: {int(np.sum(self.bits))} ({int(np.sum(self.bits)) / bit_count * 100:.1f}%)"
                )

                # Hide live bit viewer — only relevant in framed byte mode
                if hasattr(self, 'live_bit_viewer_dock'):
                    self.live_bit_viewer_dock.hide()

                # Byte mode keeps its own independent frame state — don't clear it.
                # When the user switches back, byte mode resumes exactly where it left off.

                # Reset bit-mode operations & highlights
                self.operations = []
                self.operations_list.clear()
                self.undo_stack.clear()
                self.redo_stack.clear()
                self.clear_highlights()
            except Exception as e:
                print(f"ERROR in on_mode_changed (byte->bit): {e}")
                import traceback
                traceback.print_exc()
                # Show error to user
                QMessageBox.critical(self, "Mode Change Error",
                    f"Failed to switch to bit mode:\n{str(e)}\n\nCheck console for details.")
                return

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
        self.width_spin.setKeyboardTracking(False)

        self._width_timer = QTimer()
        self._width_timer.setSingleShot(True)
        self._width_timer.setInterval(75)
        self._pending_width_value = self.width_spin.value()
        self.width_spin.valueChanged.connect(self._queue_width_change)
        self._width_timer.timeout.connect(self._apply_width_change)

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

        view_group = QGroupBox("View Options")
        view_layout = QVBoxLayout()
        self.byte_view_binary_checkbox = QCheckBox("Show binary under hex")
        self.byte_view_binary_checkbox.toggled.connect(self.update_byte_view_overlays)
        view_layout.addWidget(self.byte_view_binary_checkbox)

        self.byte_view_ascii_checkbox = QCheckBox("Show ASCII under hex")
        self.byte_view_ascii_checkbox.toggled.connect(self.update_byte_view_overlays)
        view_layout.addWidget(self.byte_view_ascii_checkbox)

        view_group.setLayout(view_layout)
        byte_mode_layout.addWidget(view_group)

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

        framing_group.setLayout(framing_layout)
        byte_mode_layout.addWidget(framing_group)

        const_cols_group = QGroupBox("Constant Columns")
        const_cols_layout = QVBoxLayout()

        const_label = QLabel("Highlight:")
        const_cols_layout.addWidget(const_label)

        self.const_col_button_group = QButtonGroup(self)

        self.const_off_radio = QRadioButton("Off")
        self.const_off_radio.setChecked(True)
        self.const_off_radio.toggled.connect(self.on_constant_highlight_changed)
        self.const_col_button_group.addButton(self.const_off_radio)
        const_cols_layout.addWidget(self.const_off_radio)

        self.const_on_radio = QRadioButton("On")
        self.const_on_radio.toggled.connect(self.on_constant_highlight_changed)
        self.const_col_button_group.addButton(self.const_on_radio)
        const_cols_layout.addWidget(self.const_on_radio)

        const_cols_layout.addStretch()

        const_cols_group.setLayout(const_cols_layout)
        byte_mode_layout.addWidget(const_cols_group)

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

        # Wrap in a scroll area so controls don't squish when panel is narrow
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setMinimumWidth(240)
        return scroll

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

        # Create horizontal layout for display + entropy heatmap
        display_heatmap_layout = QHBoxLayout()

        self.display_container = QWidget()
        self.display_layout = QVBoxLayout(self.display_container)
        self.display_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea()
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
        self.update_byte_view_overlays()

        # Install event filter on main window to detect clicks outside table
        self.centralWidget().installEventFilter(self)

        # Add display container to horizontal layout (no more sidebar heatmap)
        display_heatmap_layout.addWidget(self.display_container, stretch=1)

        layout.addLayout(display_heatmap_layout, stretch=1)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        return widget

    def update_byte_view_overlays(self):
        """Sync byte-table overlay options with the UI checkboxes."""
        if not hasattr(self, 'byte_table'):
            return
        show_binary = getattr(self, 'byte_view_binary_checkbox', None)
        show_ascii = getattr(self, 'byte_view_ascii_checkbox', None)
        binary_enabled = show_binary.isChecked() if show_binary is not None else False
        ascii_enabled = show_ascii.isChecked() if show_ascii is not None else False
        self.byte_table.set_overlay_options(
            show_binary=binary_enabled,
            show_ascii=ascii_enabled,
        )

    def update_row_size(self):
        self.byte_table.set_row_size(self.row_size_spin.value())
        # Auto-refresh constant column highlighting if enabled
        if hasattr(self, 'const_on_radio') and self.const_on_radio.isChecked():
            self.highlight_constant_columns()

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

    def add_column_definition_prefilled(self, start_byte, end_byte):
        """Add a column definition with pre-filled start and end bytes."""
        dialog = AddColumnDialog(self)
        dialog.setWindowTitle("Add Column Definition from Selection")

        # Pre-fill with byte mode and selected range
        dialog.byte_radio.setChecked(True)
        dialog.start_byte_spin.setValue(start_byte)
        dialog.end_byte_spin.setValue(end_byte)

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

            # Clear column selection after adding definition
            self.byte_table.selected_columns.clear()
            self.byte_table.update_display()
            self.byte_table._update_live_bit_viewer()

    def add_column_definition_prefilled_bits(self, start_bit, total_bits):
        """Add a column definition with pre-filled bit positions."""
        dialog = AddColumnDialog(self)
        dialog.setWindowTitle("Add Column Definition from Selection")

        # Pre-fill with bit mode and selected range
        dialog.bit_radio.setChecked(True)
        dialog.bit_pos_radio.setChecked(True)  # Absolute bit position mode
        dialog.abs_bit_position_spin.setValue(start_bit)
        dialog.total_bits_spin.setValue(total_bits)

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

            # Clear column selection after adding definition
            self.byte_table.selected_columns.clear()
            self.byte_table.update_display()
            self.byte_table._update_live_bit_viewer()

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

    def frame_sync(self, prefill_pattern=None):
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
        input_hex_radio = QRadioButton("Hex (0x414B or 414B)")
        input_binary_radio = QRadioButton("Binary (1/0 or X/O)")
        input_binary_radio.setChecked(False)
        input_hex_radio.setChecked(True)
        input_format_layout.addWidget(input_hex_radio)
        input_format_layout.addWidget(input_binary_radio)
        input_format_group.setLayout(input_format_layout)
        layout.addWidget(input_format_group)

        # Pattern input
        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("Sync Pattern:"))
        pattern_input = QLineEdit()
        pattern_input.setPlaceholderText("e.g. 0x414B or 414B")

        # Pre-fill pattern if provided
        if prefill_pattern:
            pattern_input.setText(prefill_pattern)
            # Detect format based on pattern (default to Hex)
            if prefill_pattern.startswith('0x') or prefill_pattern.startswith('0X'):
                input_hex_radio.setChecked(True)

        pattern_layout.addWidget(pattern_input)
        def update_placeholder():
            if input_hex_radio.isChecked():
                pattern_input.setPlaceholderText("e.g. 0x414B or 414B")
            else:
                pattern_input.setPlaceholderText("e.g. 11001010 or xoxxoo")

        input_hex_radio.toggled.connect(update_placeholder)
        input_binary_radio.toggled.connect(update_placeholder)
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
        _populate_color_combo(color_combo, "Sky")
        color_layout.addWidget(color_combo)
        layout.addWidget(QWidget())
        layout.itemAt(layout.count() - 1).widget().setLayout(color_layout)

        # Display format selection (will be updated based on pattern length)
        display_format_group = QGroupBox("Display Format")
        display_format_layout = QVBoxLayout()
        display_hex_radio = QRadioButton("Hex")
        display_binary_radio = QRadioButton("Binary")
        display_ascii_radio = QRadioButton("ASCII")
        display_hex_radio.setChecked(True)
        display_format_layout.addWidget(display_hex_radio)
        display_format_layout.addWidget(display_binary_radio)
        display_format_layout.addWidget(display_ascii_radio)
        display_format_group.setLayout(display_format_layout)
        layout.addWidget(display_format_group)

        mode_group = QGroupBox("Search Mode")
        mode_layout = QVBoxLayout()
        self.mode_bit_radio = QRadioButton("Bit-accurate (finds offset sync; slower)")
        self.mode_byte_radio = QRadioButton("Byte-aligned only (fast; requires byte pattern)")
        self.mode_bit_radio.setChecked(True)
        mode_layout.addWidget(self.mode_bit_radio)
        mode_layout.addWidget(self.mode_byte_radio)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

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
        found_info = {'count': 0, 'max_len': 0, 'bits': None, 'positions': None, 'mode': 'bit'}

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

            use_byte_mode, allowed = determine_search_mode(bits)
            if not allowed:
                result_label.setText("Search cancelled")
                return
            found_info['mode'] = 'byte' if use_byte_mode else 'bit'

            # Find pattern positions WITHOUT framing (just for preview)
            positions, max_len = self.byte_table.find_pattern_positions(bits, byte_aligned=use_byte_mode)
            found_info['positions'] = positions
            found_info['count'] = len(positions)
            found_info['max_len'] = max_len

            if len(positions) > 0:
                mode_text = "byte-aligned" if use_byte_mode else "bit-accurate"
                result_label.setText(f"✓ Found {len(positions):,} frames ({mode_text}, max length: {max_len:,} bytes)")
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
            bits_str = pattern_text.replace(' ', '').upper().replace('X', '1').replace('O', '0')
            if bits_str and all(c in '10' for c in bits_str):
                bits = [int(b) for b in bits_str]
        else:
            hex_str = pattern_text.replace(' ', '').upper()
            if hex_str.startswith('0X'):
                hex_str = hex_str[2:]
            if hex_str and len(hex_str) % 2 == 0 and all(c in '0123456789ABCDEF' for c in hex_str):
                try:
                    byte_values = bytes.fromhex(hex_str)
                    bits = np.unpackbits(np.frombuffer(byte_values, dtype=np.uint8)).tolist()
                except ValueError:
                    bits = None

        if not bits:
            QMessageBox.warning(self, "Frame Sync", "Invalid pattern.")
            return

        self.save_state()

        use_byte_mode, allowed = determine_search_mode(bits)
        if not allowed:
            return

        # Actually perform framing now
        frame_count = self.byte_table.set_frame_pattern(None, pattern_bits=bits, force_byte_mode=use_byte_mode)

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

        mode_label = "Byte-aligned" if use_byte_mode else "Bit-accurate"
        op_text = f"Frame Sync ({mode_label}): {pattern_text}"
        self.operations.append(op_text)
        self.add_operation_to_list(op_text)

        mode_text = "byte-aligned" if use_byte_mode else "bit-accurate"

        # Show live bit viewer dock widget now that we have frames
        if hasattr(self, 'live_bit_viewer_dock'):
            self.live_bit_viewer_dock.show()

        # Check if frames include zero-padding (when pattern length isn't byte-aligned)
        if bit_count % 8 != 0:
            padding_bits = 8 - (bit_count % 8)
            self.status_label.setText(
                f"Framed into {frame_count} frames ({mode_text}, zero-padded with {padding_bits} bits per frame)"
            )
        else:
            self.status_label.setText(f"Framed into {frame_count} frames ({mode_text})")

        # Auto-refresh constant column highlighting if enabled
        if hasattr(self, 'const_on_radio') and self.const_on_radio.isChecked():
            self.highlight_constant_columns()

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

        # Hide live bit viewer when frames are cleared
        if hasattr(self, 'live_bit_viewer_dock'):
            self.live_bit_viewer_dock.hide()
            self.live_bit_viewer_canvas.set_frame_bits([])
        if getattr(self, 'field_inspector', None):
            self.field_inspector.set_frames_bits([])
        if getattr(self, 'field_stats_widget', None):
            self.field_stats_widget.set_frames_bits([])

        # Clear constant column highlighting when frames are cleared
        if hasattr(self, 'const_on_radio') and self.const_on_radio.isChecked():
            self.clear_constant_highlights()
            self.const_off_radio.setChecked(True)

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

    def on_constant_highlight_changed(self):
        """Handle constant column highlight radio button changes"""
        if self.const_on_radio.isChecked():
            self.highlight_constant_columns()
        else:
            self.clear_constant_highlights()

    def save_state(self):
        bits_copy = self.bits.copy() if self.bits is not None else None
        bytes_copy = self.bytes_data.copy() if self.bytes_data is not None else None
        ops_copy = self.operations.copy()
        frame_pat = copy.deepcopy(self.byte_table.frame_pattern) if self.byte_table.frame_pattern is not None else None
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
            copy.deepcopy(self.byte_table.frame_pattern) if self.byte_table.frame_pattern is not None else None
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
            copy.deepcopy(self.byte_table.frame_pattern) if self.byte_table.frame_pattern is not None else None
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
                if op_text.startswith("Frame Sync"):
                    match = re.match(r"Frame Sync(?: \(([^)]+)\))?:\s*(.+)", op_text)
                    pattern_str = match.group(2).strip() if match else op_text.split("Frame Sync:", 1)[1].strip()
                    prefer_byte = False
                    if match and match.group(1):
                        mode_text = match.group(1).lower()
                        prefer_byte = "byte" in mode_text
                    pattern = self.parse_hex_pattern(pattern_str)
                    if pattern is not None:
                        self.byte_table.set_frame_pattern(pattern, force_byte_mode=prefer_byte)
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

                # Entropy heatmap will update when columns are selected in framed mode

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

    def _queue_width_change(self, value):
        """Coalesce repeated spinner steps so holding the arrow does one redraw after input settles."""
        self._pending_width_value = value
        self._width_timer.start()

    def _apply_width_change(self):
        """Lightweight handler for width spinner ??? avoids re-assigning bit data."""
        width = getattr(self, '_pending_width_value', self.width_spin.value())
        if self.data_mode != "bit" or self.bits is None:
            self.update_display()
            return
        mode = self.mode_combo.currentText().lower()
        if mode in ["squares", "circles"]:
            self.canvas.set_bits_per_row(width)
        elif mode in ["binary", "hex"]:
            self.text_display.width = width
            self.text_display.update_display()

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

            # Entropy heatmap will update when columns are selected in framed mode

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
