"""Shared color palette helpers."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QPixmap

COLOR_OPTIONS = [
    ("None", None),
    ("Sunshine", "#FFF4B5"),
    ("Mint", "#C8FACC"),
    ("Sky", "#B9DEFF"),
    ("Coral", "#FFB3AB"),
    ("Lilac", "#E0C6FF"),
    ("Seafoam", "#BFFFE3"),
    ("Rose", "#FFC7DA"),
    ("Lavender", "#D7C0FF"),
    ("Orange", "#FFD299"),
    ("Teal", "#9FE3D7"),
    ("Gold", "#FFE08A"),
    ("Aqua", "#9ADBF2"),
    ("Moss", "#C4E5A2"),
    ("Plum", "#E3B0FF"),
    ("Salmon", "#FFCDC1"),
    ("Slate", "#C7D2E5"),
    ("Peach", "#FFD8B2"),
    ("Lime", "#DBFF95"),
    ("Berry", "#FFB2CF"),
    ("Ocean", "#A4C8FF"),
]

COLOR_NAME_TO_QCOLOR = {
    name: QColor(hex_code) for name, hex_code in COLOR_OPTIONS if hex_code
}


def populate_color_combo(combo, current_name: str = "None") -> None:
    combo.clear()
    for name, hex_code in COLOR_OPTIONS:
        combo.addItem(name)
        pixmap = QPixmap(16, 16)
        pixmap.fill(QColor(hex_code) if hex_code else QColor(240, 240, 240))
        combo.setItemData(combo.count() - 1, QIcon(pixmap), Qt.ItemDataRole.DecorationRole)

    index = combo.findText(current_name)
    if index >= 0:
        combo.setCurrentIndex(index)
