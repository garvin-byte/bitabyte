"""Layout helpers for the main bit viewer window."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .canvas import BitCanvas
from .table import ByteStructuredTableWidget
from .window_columns import ColumnDefinitionsTreeWidget
from .widgets import TextDisplayWidget


class BitViewerWindowLayoutMixin:
    LEFT_PANEL_WIDTH = 380

    def create_left_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        file_group = QGroupBox("File")
        file_layout = QVBoxLayout()

        btn_open = QPushButton("Open File")
        btn_open.clicked.connect(self.open_file)
        file_layout.addWidget(btn_open)

        btn_save = QPushButton("Save Processed")
        btn_save.clicked.connect(self.save_file)
        file_layout.addWidget(btn_save)

        self.file_info_label = QLabel("No file loaded")
        self.file_info_label.setWordWrap(True)
        file_layout.addWidget(self.file_info_label)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        self.bit_mode_widget = QWidget()
        bit_mode_layout = QVBoxLayout(self.bit_mode_widget)
        bit_mode_layout.setContentsMargins(0, 0, 0, 0)

        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout()

        mode_label = QLabel("Mode:")
        display_layout.addWidget(mode_label)

        self.display_mode_group = QButtonGroup(self)
        mode_buttons_layout = QHBoxLayout()
        mode_rows = [
            ("Squares", True),
            ("Circles", False),
            ("Binary", False),
            ("Hex", False),
        ]
        for mode_text, checked in mode_rows:
            mode_button = QRadioButton(mode_text)
            mode_button.setChecked(checked)
            mode_button.toggled.connect(self.update_display)
            self.display_mode_group.addButton(mode_button)
            mode_buttons_layout.addWidget(mode_button)
        mode_buttons_layout.addStretch()
        display_layout.addLayout(mode_buttons_layout)

        width_layout = QHBoxLayout()
        width_label = QLabel("Width:")
        width_label.setMinimumWidth(48)
        width_layout.addWidget(width_label)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(8, 1_000_000)
        self.width_spin.setValue(64)
        self.width_spin.setKeyboardTracking(False)
        self.width_spin.setFixedWidth(150)

        self._width_timer = QTimer()
        self._width_timer.setSingleShot(True)
        self._width_timer.setInterval(75)
        self._pending_width_value = self.width_spin.value()
        self.width_spin.valueChanged.connect(self._queue_width_change)
        self._width_timer.timeout.connect(self._apply_width_change)

        width_layout.addWidget(self.width_spin)
        display_layout.addLayout(width_layout)

        size_layout = QHBoxLayout()
        size_label = QLabel("Size:")
        size_label.setMinimumWidth(48)
        size_layout.addWidget(size_label)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(4, 30)
        self.size_spin.setValue(10)
        self.size_spin.valueChanged.connect(self.update_display)
        self.size_spin.setFixedWidth(150)
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
        self.error_tolerance_spin.setPrefix("Error +/-")
        self.error_tolerance_spin.setSuffix("%")
        pattern_layout.addWidget(self.error_tolerance_spin)

        btn_highlight = QPushButton("Highlight Pattern")
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

        self.byte_mode_widget = QWidget()
        byte_mode_layout = QVBoxLayout(self.byte_mode_widget)
        byte_mode_layout.setContentsMargins(0, 0, 0, 0)

        byte_display_group = QGroupBox("Live Viewer")
        byte_display_layout = QVBoxLayout()

        byte_mode_label = QLabel("Mode:")
        byte_display_layout.addWidget(byte_mode_label)

        self.byte_display_mode_group = QButtonGroup(self)
        byte_mode_buttons_layout = QHBoxLayout()
        byte_modes = [
            ("Squares", "squares", True),
            ("Circles", "circles", False),
            ("Binary", "binary", False),
            ("Hex", "hex", False),
        ]
        for mode_text, mode_value, checked in byte_modes:
            mode_button = QRadioButton(mode_text)
            mode_button.mode_value = mode_value  # type: ignore[attr-defined]
            mode_button.setChecked(checked)
            mode_button.toggled.connect(self.update_byte_view_overlays)
            self.byte_display_mode_group.addButton(mode_button)
            byte_mode_buttons_layout.addWidget(mode_button)
        byte_mode_buttons_layout.addStretch()
        byte_display_layout.addLayout(byte_mode_buttons_layout)

        row_layout = QHBoxLayout()
        row_size_label = QLabel("Bytes per row:")
        row_size_label.setMinimumWidth(96)
        row_layout.addWidget(row_size_label)
        self.row_size_spin = QSpinBox()
        self.row_size_spin.setRange(1, 1_000_000)
        self.row_size_spin.setValue(16)
        self.row_size_spin.valueChanged.connect(self.update_row_size)
        self.row_size_spin.setFixedWidth(150)
        row_layout.addWidget(self.row_size_spin)
        byte_display_layout.addLayout(row_layout)

        byte_size_layout = QHBoxLayout()
        byte_size_label = QLabel("Size:")
        byte_size_label.setMinimumWidth(96)
        byte_size_layout.addWidget(byte_size_label)
        self.byte_size_spin = QSpinBox()
        self.byte_size_spin.setRange(8, 20)
        self.byte_size_spin.setValue(10)
        self.byte_size_spin.valueChanged.connect(self.update_byte_display_size)
        self.byte_size_spin.setFixedWidth(150)
        byte_size_layout.addWidget(self.byte_size_spin)
        byte_display_layout.addLayout(byte_size_layout)

        byte_display_group.setLayout(byte_display_layout)
        byte_mode_layout.addWidget(byte_display_group)

        columns_group = QGroupBox("Column Definitions")
        columns_layout = QVBoxLayout()

        self.columns_list = ColumnDefinitionsTreeWidget()
        self.columns_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.columns_list.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.columns_list.itemDoubleClicked.connect(lambda item, _column: self.edit_column_definition(item))
        columns_layout.addWidget(self.columns_list)

        btn_add_column = QPushButton("Add Column")
        btn_add_column.clicked.connect(self.add_column_definition)
        columns_layout.addWidget(btn_add_column)

        btn_remove_column = QPushButton("Remove Selected")
        btn_remove_column.clicked.connect(self.remove_column_definition)
        columns_layout.addWidget(btn_remove_column)

        btn_combine_columns = QPushButton("Combine Selected")
        btn_combine_columns.clicked.connect(self.combine_selected_column_definitions)
        columns_layout.addWidget(btn_combine_columns)

        btn_clear_columns = QPushButton("Clear All")
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
        const_cols_layout.addWidget(QLabel("Highlight:"))

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

        history_group = QGroupBox("Applied Operations")
        history_layout = QVBoxLayout()

        self.operations_list = QListWidget()
        history_layout.addWidget(self.operations_list)

        btn_undo = QPushButton("Undo")
        btn_undo.clicked.connect(self.undo)
        history_layout.addWidget(btn_undo)

        btn_redo = QPushButton("Redo")
        btn_redo.clicked.connect(self.redo)
        history_layout.addWidget(btn_redo)

        btn_reset = QPushButton("Reset")
        btn_reset.clicked.connect(self.reset_to_original)
        history_layout.addWidget(btn_reset)

        history_group.setLayout(history_layout)
        layout.addWidget(history_group)

        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setMinimumWidth(self.LEFT_PANEL_WIDTH)
        scroll.setMaximumWidth(self.LEFT_PANEL_WIDTH)
        self.left_panel_scroll = scroll
        return scroll

    def update_left_panel_visibility(self):
        if self.data_mode == "bit":
            self.bit_mode_widget.show()
            self.byte_mode_widget.hide()
            if hasattr(self, "combine_selection_button"):
                self.combine_selection_button.hide()
        else:
            self.bit_mode_widget.hide()
            self.byte_mode_widget.show()
            if hasattr(self, "combine_selection_button"):
                self.combine_selection_button.show()

    def current_bit_display_mode(self):
        if not hasattr(self, "display_mode_group"):
            return "squares"
        checked_button = self.display_mode_group.checkedButton()
        if checked_button is None:
            return "squares"
        return checked_button.text().lower()

    def create_right_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        nav_layout = QHBoxLayout()
        btn_up = QPushButton("Up")
        btn_up.clicked.connect(self.scroll_up)
        nav_layout.addWidget(btn_up)

        btn_down = QPushButton("Down")
        btn_down.clicked.connect(self.scroll_down)
        nav_layout.addWidget(btn_down)

        btn_start = QPushButton("Start")
        btn_start.clicked.connect(self.go_to_start)
        nav_layout.addWidget(btn_start)

        btn_end = QPushButton("End")
        btn_end.clicked.connect(self.go_to_end)
        nav_layout.addWidget(btn_end)

        nav_layout.addStretch()
        layout.addLayout(nav_layout)

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

        self.combine_selection_button = QPushButton("Combine Selection")
        self.combine_selection_button.clicked.connect(self.combine_selected_table_columns_from_current_selection)
        self.combine_selection_button.hide()
        self.display_layout.addWidget(self.combine_selection_button)

        self.byte_table = ByteStructuredTableWidget()
        self.byte_table.parent_window = self
        self.display_layout.addWidget(self.byte_table)
        self.byte_table.hide()
        self.update_byte_view_overlays()

        self.centralWidget().installEventFilter(self)
        display_heatmap_layout.addWidget(self.display_container, stretch=1)
        layout.addLayout(display_heatmap_layout, stretch=1)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        return widget

    def update_byte_view_overlays(self):
        live_viewer = getattr(self, "live_bit_viewer_canvas", None)
        if live_viewer is None:
            return
        mode_value = self.current_byte_overlay_mode()
        live_viewer.set_display_mode(mode_value)

    def current_byte_overlay_mode(self):
        if not hasattr(self, "byte_display_mode_group"):
            return "none"
        checked_button = self.byte_display_mode_group.checkedButton()
        if checked_button is None:
            return "none"
        return getattr(checked_button, "mode_value", "none")

    def update_row_size(self):
        self.byte_table.set_row_size(self.row_size_spin.value())
        if hasattr(self, "const_on_radio") and self.const_on_radio.isChecked():
            self.highlight_constant_columns()

    def update_byte_display_size(self):
        live_viewer = getattr(self, "live_bit_viewer_canvas", None)
        if live_viewer is None or not hasattr(self, "byte_size_spin"):
            return
        live_viewer.set_size_value(self.byte_size_spin.value())
