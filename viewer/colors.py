"""Color constants and palette helpers for the bit viewer."""
#!/usr/bin/env python3
"""
PyQt6 Bit Viewer - Fast rendering with all your operations
Enhanced byte mode with flexible bit/byte column definitions and combined headers
"""

import sys
import copy
import numpy as np
import re
import math
from collections import Counter
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
                             QFileDialog, QScrollArea, QMessageBox, QComboBox,
                             QLineEdit, QListWidget, QGroupBox, QCheckBox,
                             QSplitter, QTextEdit, QDialog, QDialogButtonBox,
                             QInputDialog, QListWidgetItem, QRadioButton, QButtonGroup,
                             QTableWidget, QTableWidgetItem, QMenu, QStyledItemDelegate,
                             QDockWidget, QSizePolicy)
from PyQt6.QtCore import Qt, QRect, QSize
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPalette, QFont, QIcon, QPixmap
from numpy.lib.stride_tricks import as_strided


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

BIT_SYNC_WARNING_BYTES = 256 * 1024  # 256 KB
BIT_SYNC_HARD_LIMIT_BYTES = 1 * 1024 * 1024  # 1 MB
MAX_SYNC_FRAMES = 10_000
PATTERN_SCAN_MAX_ELEMENTS = 4_000_000  # Limit 2D comparison blocks to ~4M elements

def _populate_color_combo(combo, current_name="None"):
    combo.clear()
    for name, hex_code in COLOR_OPTIONS:
        combo.addItem(name)
        idx = combo.count() - 1
        if hex_code:
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor(hex_code))
            combo.setItemData(idx, QIcon(pixmap), Qt.ItemDataRole.DecorationRole)
        else:
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor(240, 240, 240))
            combo.setItemData(idx, QIcon(pixmap), Qt.ItemDataRole.DecorationRole)
    index = combo.findText(current_name)
    if index >= 0:
        combo.setCurrentIndex(index)

