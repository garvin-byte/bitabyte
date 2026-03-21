"""Standalone GUI for the next-generation viewer."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QSplitter,
    QLabel,
    QGroupBox,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from .data import ByteDataSource
from .delegates import ByteCellDelegate
from .headers import MultiRowHeaderView
from .models import ByteTableModel, HeaderBand, HeaderModel
from .frame_sync import FrameSyncController
from .columns import ColumnDefinitionsPanel


class NextGenBitViewerWindow(QMainWindow):
    """Separated prototype window using QTableView + custom models."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Next-Gen Bit Viewer Prototype")
        self.resize(1200, 800)

        self.data_source = ByteDataSource(bytes_per_row=16)
        self.model = ByteTableModel(self.data_source)
        self.frame_sync_controller = FrameSyncController(self, self.data_source, self.model)
        self.frame_label_spans: list[tuple[int, int, str]] = []
        self._byte_row_free = self.data_source.bytes_per_row
        self._byte_row_locked = None
        self._bit_row_setting = 64

        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        controls = self._build_toolbar()
        root_layout.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter, stretch=1)

        self.left_panel = self._build_left_panel()
        splitter.addWidget(self.left_panel)
        splitter.setStretchFactor(0, 0)

        display_container = QWidget()
        display_layout = QVBoxLayout(display_container)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.setSpacing(4)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setItemDelegate(ByteCellDelegate(self.table))
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(False)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectItems)
        self.table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.table.verticalHeader().setVisible(True)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.verticalHeader().setStyleSheet("QHeaderView::section { background: #E2F6FF; }")
        self.table.horizontalHeader().setMinimumSectionSize(38)

        self.header_view = MultiRowHeaderView(Qt.Orientation.Horizontal, self.table)
        self.header_model = self._build_header_model(self.data_source.bytes_per_row)
        self.header_view.setHeaderModel(self.header_model)
        self.table.setHorizontalHeader(self.header_view)
        self._resize_columns()
        display_layout.addWidget(self.table, stretch=1)

        self.status_label = QLabel("No data loaded")
        display_layout.addWidget(self.status_label)

        splitter.addWidget(display_container)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(central)

    def _set_row_spin_state(self, minimum: int, maximum: int, step: int, value: int, *, enabled: bool) -> None:
        self.row_spin.blockSignals(True)
        self.row_spin.setRange(minimum, maximum)
        self.row_spin.setSingleStep(step)
        self.row_spin.setValue(value)
        self.row_spin.blockSignals(False)
        self.row_spin.setEnabled(enabled)

    def _refresh_header(self, width: int | None = None) -> None:
        header_width = self.row_spin.value() if width is None else width
        self.header_model = self._build_header_model(header_width)
        self.header_view.setHeaderModel(self.header_model)
        self._resize_columns()

    def _build_toolbar(self):
        layout = QHBoxLayout()
        layout.setSpacing(10)

        layout.addWidget(QLabel("View Mode:"))
        mode_group = QHBoxLayout()
        self.mode_button_group = QButtonGroup(self)
        self.byte_radio = QRadioButton("Byte View")
        self.byte_radio.setChecked(True)
        self.bit_radio = QRadioButton("Bit View")
        self.mode_button_group.addButton(self.byte_radio, 0)
        self.mode_button_group.addButton(self.bit_radio, 1)
        self.mode_button_group.idToggled.connect(self._on_mode_toggled)
        mode_group.addWidget(self.byte_radio)
        mode_group.addWidget(self.bit_radio)
        layout.addLayout(mode_group)

        layout.addStretch()
        return layout

    def _build_file_group(self) -> QGroupBox:
        group = QGroupBox("File")
        layout = QVBoxLayout(group)
        self.open_btn = QPushButton("Open File…")
        self.open_btn.clicked.connect(self._open_file_dialog)
        layout.addWidget(self.open_btn)

        self.save_btn = QPushButton("Save Processed…")
        self.save_btn.clicked.connect(self._save_file)
        self.save_btn.setEnabled(False)
        layout.addWidget(self.save_btn)

        self.file_info_label = QLabel("No file loaded")
        self.file_info_label.setWordWrap(True)
        layout.addWidget(self.file_info_label)
        return group

    def _build_row_group(self) -> QGroupBox:
        group = QGroupBox("Row Settings")
        layout = QVBoxLayout(group)
        self.row_label = QLabel("Bytes / Row:")
        layout.addWidget(self.row_label)
        self.row_spin = QSpinBox()
        self.row_spin.setRange(1, 512)
        self.row_spin.setSingleStep(1)
        self.row_spin.setValue(self.data_source.bytes_per_row)
        self.row_spin.valueChanged.connect(self._update_row_width)
        layout.addWidget(self.row_spin)
        return group

    def _build_byte_mode_panel(self) -> QGroupBox:
        panel = QGroupBox("Byte Mode")
        layout = QVBoxLayout(panel)
        self.columns_panel = ColumnDefinitionsPanel(self)
        layout.addWidget(self.columns_panel)
        self.columns_panel.refresh()

        layout.addWidget(QLabel("Framing"))
        self.frame_sync_btn = QPushButton("Frame Sync…")
        self.frame_sync_btn.clicked.connect(self._open_frame_sync)
        layout.addWidget(self.frame_sync_btn)

        self.clear_frames_btn = QPushButton("Clear Frames")
        self.clear_frames_btn.clicked.connect(self._clear_frames)
        self.clear_frames_btn.setEnabled(False)
        layout.addWidget(self.clear_frames_btn)

        self.highlight_const_btn = QPushButton("Highlight Constant Columns")
        self.highlight_const_btn.clicked.connect(self._highlight_constant_columns)
        layout.addWidget(self.highlight_const_btn)
        self.clear_const_btn = QPushButton("Clear Constant Highlights")
        self.clear_const_btn.clicked.connect(self._clear_constant_highlights)
        layout.addWidget(self.clear_const_btn)
        return panel

    def _build_bit_mode_panel(self) -> QGroupBox:
        panel = QGroupBox("Bit Mode")
        layout = QVBoxLayout(panel)
        layout.addWidget(QPushButton("Highlight Pattern"))
        layout.addWidget(QPushButton("Run Bit Operation"))
        return panel

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(10)

        layout.addWidget(self._build_file_group())
        layout.addWidget(self._build_row_group())

        self.byte_mode_panel = self._build_byte_mode_panel()
        layout.addWidget(self.byte_mode_panel)

        self.bit_mode_panel = self._build_bit_mode_panel()
        layout.addWidget(self.bit_mode_panel)
        self.bit_mode_panel.hide()

        layout.addStretch()
        return container

    def _build_header_model(self, bytes_per_row: int) -> HeaderModel:
        if self.model.display_mode == "byte":
            byte_labels = [f"Byte {idx:02X}" for idx in range(bytes_per_row)]
            bit_labels = [f"{idx * 8}-{idx * 8 + 7}" for idx in range(bytes_per_row)]
            bands: list[HeaderBand] = []
            label_labels, label_spans = self._build_label_band(bytes_per_row)
            bands.append(HeaderBand(labels=label_labels, spans=label_spans if label_spans else None, height=28))
            bands.append(HeaderBand(labels=byte_labels, height=26))
            bands.append(HeaderBand(labels=bit_labels, height=20))
            return HeaderModel(bands=bands)

        bit_labels = [f"Bit {col}" for col in range(bytes_per_row)]
        return HeaderModel(
            bands=[
                HeaderBand(labels=bit_labels, height=24),
            ]
        )

    def _build_label_band(self, width: int) -> tuple[list[str], list[tuple[int, int, str]]]:
        labels = ["" for _ in range(width)]
        spans: list[tuple[int, int, str]] = list(self.frame_label_spans)
        for col_def in self.model.column_definitions:
            if not col_def.label:
                continue
            span = col_def.normalized_byte_span(width)
            if span is None:
                continue
            start, end = span
            length = end - start + 1
            span_tuple = (start, length, col_def.label)
            if span_tuple not in spans:
                spans.append(span_tuple)
            for idx in range(start, end + 1):
                labels[idx] = col_def.label
        return labels, spans

    # Slots
    def _open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Data", "", "Binary files (*.bin);;All Files (*.*)")
        if not file_name:
            return
        path = Path(file_name)
        self.data_source.load_from_file(path)
        self.model.beginResetModel()
        self.model.endResetModel()
        self.status_label.setText(f"Loaded {path.name} ({self.data_source.byte_length:,} bytes)")
        self.file_info_label.setText(f"{path.name}\n{self.data_source.byte_length:,} bytes")
        self.save_btn.setEnabled(True)
        self._refresh_header()
        self._clear_frames(update_status=False)

    def _update_row_width(self, value: int):
        if self.model.display_mode == "byte":
            if self.model.has_frames():
                return
            self._byte_row_free = value
            self.model.set_bytes_per_row(value)
        else:
            self._bit_row_setting = value
            self.model.set_bits_per_row(value)
        self._refresh_header(value)

    def _save_file(self):
        if self.data_source.byte_length == 0:
            QMessageBox.information(self, "Save Processed", "No data to save.")
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Processed Data", "", "Binary files (*.bin);;All Files (*.*)")
        if not file_name:
            return
        with open(file_name, "wb") as f:
            f.write(self.data_source.raw_bytes())
        QMessageBox.information(self, "Save Processed", f"Saved {self.data_source.byte_length:,} bytes.")

    def _resize_columns(self):
        for col in range(self.model.columnCount()):
            width = 50 if self.model.display_mode == "byte" else 26
            self.table.setColumnWidth(col, width)

    def _highlight_constant_columns(self):
        QMessageBox.information(
            self,
            "Not yet implemented",
            "Constant column analysis will be available in a future iteration.",
        )

    def _clear_constant_highlights(self):
        QMessageBox.information(
            self,
            "Not yet implemented",
            "Constant column analysis will be available in a future iteration.",
        )

    def _on_mode_toggled(self, button_id: int, checked: bool):
        if not checked:
            return
        mode = "byte" if button_id == 0 else "bit"
        self.model.set_display_mode(mode)
        if mode == "byte":
            self.byte_mode_panel.show()
            self.bit_mode_panel.hide()
            self.row_label.setText("Bytes / Row:")
            self._set_row_spin_state(1, 512, 1, self._byte_row_free, enabled=True)
            self.model.set_bytes_per_row(self._byte_row_free)
            if self.model.has_frames():
                self._lock_row_size_to_frame()
        else:
            self.byte_mode_panel.hide()
            self.bit_mode_panel.show()
            self.row_label.setText("Bits / Row:")
            self._set_row_spin_state(8, 4096, 8, self._bit_row_setting, enabled=True)
            self.model.set_bits_per_row(self._bit_row_setting)
        self._refresh_header()
        self.clear_frames_btn.setEnabled(self.model.has_frames())
        if self.model.has_frames() and self.model.display_mode == "byte":
            self._lock_row_size_to_frame()
        elif not self.model.has_frames():
            self._byte_row_locked = None
            self._unlock_row_size()

    def _open_frame_sync(self):
        self.frame_sync_controller.run_dialog()

    def _clear_frames(self, update_status: bool = True):
        self.frame_sync_controller.clear_frames(update_status=update_status)
        self.frame_label_spans = []

    def on_frames_changed(self):
        self._refresh_header()
        self.clear_frames_btn.setEnabled(self.model.has_frames())
        if self.model.has_frames() and self.model.display_mode == "byte":
            self._lock_row_size_to_frame()
        elif not self.model.has_frames():
            self._unlock_row_size()

    def _lock_row_size_to_frame(self):
        max_len = max(1, self.model.frame_max_length)
        self._byte_row_locked = max_len
        self._set_row_spin_state(max_len, max_len, 1, max_len, enabled=False)
        self.model.set_bytes_per_row(max_len)
        self._refresh_header(max_len)

    def _unlock_row_size(self):
        self._set_row_spin_state(1, 512, 1, self._byte_row_free, enabled=True)
        self.model.set_bytes_per_row(self._byte_row_free)
        self._refresh_header(self._byte_row_free)

    def on_column_definitions_changed(self):
        self.model.notify_column_definitions_changed()
        self.columns_panel.refresh()
        self._refresh_header()
        self.table.viewport().update()
