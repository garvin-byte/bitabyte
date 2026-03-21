"""BitViewerWindow main application window."""

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDockWidget,
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .canvas import LiveBitViewerCanvas
from .widgets import FieldInspectorWidget, FieldStatisticsWidget
from .window_bitops import BitViewerWindowBitOpsMixin
from .window_columns import BitViewerWindowColumnsMixin
from .window_file_ops import BitViewerWindowFileOpsMixin
from .window_frames import BitViewerWindowFramesMixin
from .window_layout import BitViewerWindowLayoutMixin
from .window_state import BitViewerWindowStateMixin


class BitViewerWindow(
    BitViewerWindowLayoutMixin,
    BitViewerWindowColumnsMixin,
    BitViewerWindowFramesMixin,
    BitViewerWindowStateMixin,
    BitViewerWindowBitOpsMixin,
    BitViewerWindowFileOpsMixin,
    QMainWindow,
):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Bit Viewer - Enhanced")
        self.setGeometry(100, 100, 1400, 900)
        self.setFont(QFont("Segoe UI", 10))

        self.bits = None
        self.bytes_data = None
        self.original_bits = None
        self.original_bytes = None
        self.filename = ""
        self.operations = []
        self.field_inspector = None
        self.field_stats_widget = None

        self.data_mode = "bit"
        self._return_to_byte_framed = False
        self._saved_byte_frame_pattern = None
        self._saved_byte_frames = None
        self._restore_padded_byte_frames = False
        self._saved_padded_frame_width_bits = None
        self.undo_stack = []
        self.redo_stack = []

        self.init_ui()
        self.update_left_panel_visibility()
        self._sync_live_viewer_dock_visibility()
        self.apply_theme()

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QMouseEvent
        from PyQt6.QtWidgets import QMenu

        if event.type() == QEvent.Type.MouseButtonPress and isinstance(event, QMouseEvent):
            if hasattr(self, "byte_table") and self.byte_table.selected_columns:
                widget_under_mouse = self.childAt(obj.mapTo(self, event.pos()))
                if isinstance(widget_under_mouse, QMenu):
                    return super().eventFilter(obj, event)

                global_pos = event.globalPosition().toPoint()
                table_rect = self.byte_table.rect()
                table_global_pos = self.byte_table.mapToGlobal(table_rect.topLeft())
                table_global_rect = table_rect.translated(table_global_pos)

                if not table_global_rect.contains(global_pos):
                    for col in self.byte_table.selected_columns:
                        self.byte_table._restore_column_colors(col)
                    self.byte_table.selected_columns.clear()
                    self.byte_table.clearSelection()
                    self.byte_table._update_live_bit_viewer()

        return super().eventFilter(obj, event)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

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

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 1)

        splitter.addWidget(self.create_left_panel())
        splitter.addWidget(self.create_right_panel())
        splitter.setSizes([400, 1000])

        self.live_bit_viewer_dock = QDockWidget("Live Bit Viewer", self)
        self.live_bit_viewer_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )

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
        if hasattr(self, "update_byte_view_overlays"):
            self.update_byte_view_overlays()
        if hasattr(self, "update_byte_display_size"):
            self.update_byte_display_size()
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
        self.live_bit_viewer_dock.setMinimumWidth(350)

    def apply_theme(self):
        if self.data_mode == "bit":
            background = "#f4f8ff"
            border = "#6d9dff"
            button = "#3b78ff"
            button_hover = "#6898ff"
            indicator = "#3b78ff"
            indicator_border = "#2456c8"
            panel = "#fbfdff"
            input_background = "#ffffff"
            table_background = "#f8fbff"
        else:
            background = "#f5fff5"
            border = "#66cc66"
            button = "#33cc33"
            button_hover = "#66ff66"
            indicator = "#33cc33"
            indicator_border = "#009900"
            panel = "#fbfffb"
            input_background = "#ffffff"
            table_background = "#f8fff8"

        self.setStyleSheet(
            f"""
            QWidget {{
                background-color: {background};
                color: #111;
                font-size: 10.5pt;
            }}
            QMainWindow {{
                background-color: {background};
            }}
            QGroupBox {{
                background-color: {panel};
                font-weight: 600;
                font-size: 10.5pt;
                border: 2px solid {border};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }}
            QPushButton {{
                background-color: {button};
                color: white;
                border: none;
                padding: 7px 10px;
                border-radius: 4px;
                font-size: 10.5pt;
                min-height: 18px;
            }}
            QPushButton:hover {{ background-color: {button_hover}; }}
            QPushButton:disabled {{
                background-color: #9fb3c8;
                color: #eef3f7;
            }}
            QLabel {{
                font-size: 10.5pt;
            }}
            QRadioButton, QCheckBox {{
                font-size: 10.5pt;
                spacing: 6px;
            }}
            QRadioButton::indicator:checked {{ background-color: {indicator}; border: 2px solid {indicator_border}; }}
            QCheckBox::indicator:checked {{
                background-color: {indicator};
                border: 2px solid {indicator_border};
            }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QTextEdit, QTableWidget {{
                background-color: {input_background};
                border: 1px solid #b7c6d8;
                border-radius: 4px;
                padding: 4px 6px;
                font-size: 10.5pt;
                selection-background-color: {button};
                selection-color: white;
            }}
            QListWidget, QTextEdit, QTableWidget {{
                background-color: {table_background};
            }}
            QSpinBox, QDoubleSpinBox {{
                padding-right: 26px;
                min-height: 24px;
            }}
            QSpinBox::up-button, QDoubleSpinBox::up-button {{
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 20px;
                height: 12px;
                border-left: 1px solid #b7c6d8;
                border-bottom: 1px solid #d5deea;
                background-color: #eef3f9;
                border-top-right-radius: 4px;
            }}
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 20px;
                height: 12px;
                border-left: 1px solid #b7c6d8;
                background-color: #eef3f9;
                border-bottom-right-radius: 4px;
            }}
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
                background-color: #dce8f6;
            }}
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow,
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
                width: 12px;
                height: 8px;
            }}
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
                image: url(viewer/icons/spin_up.svg);
            }}
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
                image: url(viewer/icons/spin_down.svg);
            }}
            QHeaderView::section {{
                background-color: #e9eef5;
                color: #111;
                border: 1px solid #c3d0df;
                padding: 6px 4px;
                font-size: 10pt;
                font-weight: 600;
            }}
            QDockWidget {{
                font-size: 10.5pt;
                font-weight: 600;
            }}
            QDockWidget::title {{
                background-color: {panel};
                padding: 6px 8px;
                border-bottom: 1px solid #c8d4e0;
            }}
            QScrollBar:vertical {{
                width: 14px;
                background: {panel};
            }}
            QScrollBar:horizontal {{
                height: 14px;
                background: {panel};
            }}
            """
        )


def main():
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = BitViewerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
