"""Column definition models and dialogs for next-gen viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QListWidget,
    QListWidgetItem,
)

from .models import ColumnDefinition
from .colors import COLOR_NAME_TO_QCOLOR, populate_color_combo

if TYPE_CHECKING:  # pragma: no cover
    from .main_window import NextGenBitViewerWindow


class ColumnDefinitionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Column Definition")
        self.setMinimumWidth(360)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.label_input = QLineEdit()
        form.addRow("Label:", self.label_input)

        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["Byte Range", "Bit Range"])
        form.addRow("Unit:", self.unit_combo)

        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, 10_000_000)
        form.addRow("Start (byte):", self.start_spin)

        self.end_spin = QSpinBox()
        self.end_spin.setRange(0, 10_000_000)
        form.addRow("End (byte):", self.end_spin)

        self.bit_total_spin = QSpinBox()
        self.bit_total_spin.setRange(1, 512)
        self.bit_total_spin.setValue(8)
        form.addRow("Total bits:", self.bit_total_spin)

        self.format_combo = QComboBox()
        self.format_combo.addItems(["Hex", "Binary", "Decimal"])
        form.addRow("Display format:", self.format_combo)

        self.color_combo = QComboBox()
        populate_color_combo(self.color_combo, "Sunshine")
        form.addRow("Color:", self.color_combo)

        layout.addLayout(form)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_definition(self) -> ColumnDefinition:
        unit = "byte" if self.unit_combo.currentIndex() == 0 else "bit"
        start_byte = self.start_spin.value()
        end_byte = self.end_spin.value()
        start_bit = start_byte * 8
        total_bits = self.bit_total_spin.value()
        if unit == "bit":
            start_bit = self.start_spin.value()
            start_byte = 0
            end_byte = 0
        label = self.label_input.text().strip()
        return ColumnDefinition(
            start_byte=start_byte,
            end_byte=end_byte,
            label=label,
            display_format=self.format_combo.currentText().lower(),
            color_name=self.color_combo.currentText(),
            unit=unit,
            start_bit=start_bit,
            total_bits=total_bits,
        )


class ColumnDefinitionsPanel(QWidget):
    def __init__(self, parent_window: NextGenBitViewerWindow):
        super().__init__()
        self.window = parent_window
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(QLabel("<b>Column Definitions</b>"))
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        button_layout = QHBoxLayout()
        self.add_button = QPushButton("Add Column")
        self.add_button.clicked.connect(self._add_column)
        button_layout.addWidget(self.add_button)
        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.clicked.connect(self._remove_selected)
        button_layout.addWidget(self.remove_button)
        self.clear_button = QPushButton("Clear All")
        self.clear_button.clicked.connect(self.clear_all)
        button_layout.addWidget(self.clear_button)
        layout.addLayout(button_layout)

    def refresh(self):
        self.list_widget.clear()
        for col_def in self.window.model.column_definitions:
            item = QListWidgetItem(self._description_for(col_def))
            color = COLOR_NAME_TO_QCOLOR.get(col_def.color_name)
            if color:
                item.setBackground(color)
            self.list_widget.addItem(item)

    def _add_column(self):
        dialog = ColumnDefinitionDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            definition = dialog.get_definition()
            self.window.model.column_definitions.append(definition)
            self.window.on_column_definitions_changed()

    def _remove_selected(self):
        row = self.list_widget.currentRow()
        if row < 0:
            return
        del self.window.model.column_definitions[row]
        self.window.on_column_definitions_changed()

    def clear_all(self):
        self.window.model.column_definitions.clear()
        self.window.on_column_definitions_changed()

    def _description_for(self, col_def: ColumnDefinition) -> str:
        if col_def.unit == "byte":
            return f"{col_def.label or '(unnamed)'}: Bytes {col_def.start_byte}-{col_def.end_byte}"
        return f"{col_def.label or '(unnamed)'}: Bit {col_def.start_bit}+{col_def.total_bits}"
