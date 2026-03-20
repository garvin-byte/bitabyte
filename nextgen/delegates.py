"""Custom painting delegates."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QStyledItemDelegate, QStyle


class ByteCellDelegate(QStyledItemDelegate):
    """Delegate that mimics the classic colored byte grid."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.font = QFont("Consolas", 10)
        self.primary_bg = QColor(247, 255, 247)
        self.alt_bg = QColor(236, 248, 236)
        self.selection_bg = QColor(186, 222, 255)
        self.selection_border = QColor(60, 120, 180)
        self.text_pen = QColor(30, 30, 30)

    def paint(self, painter: QPainter, option, index):
        painter.save()
        painter.setFont(self.font)
        row = index.row()
        is_selected = option.state & QStyle.StateFlag.State_Selected

        model = index.model()
        span_info = getattr(model, "span_for_column", None)
        span = span_info(index.column()) if span_info else None
        if span and model.display_mode == "byte":
            start_col, length, col_def, is_start = span
            if not is_start:
                bg = self.primary_bg if row % 2 == 0 else self.alt_bg
                painter.fillRect(option.rect, bg)
                painter.restore()
                return
            view = option.widget
            left = view.columnViewportPosition(start_col)
            width = sum(view.columnWidth(start_col + i) for i in range(length))
            rect = option.rect
            rect.setLeft(left)
            rect.setWidth(width)
            if is_selected:
                painter.fillRect(rect, self.selection_bg)
                painter.setPen(QPen(self.selection_border, 1))
                painter.drawRect(rect.adjusted(0, 0, -1, -1))
            else:
                painter.fillRect(rect, self.primary_bg if row % 2 == 0 else self.alt_bg)
            text = self._format_span_value(model, row, start_col, length, col_def)
            painter.setPen(self.text_pen)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text or "--")
            painter.restore()
            return

        custom_bg = index.data(Qt.ItemDataRole.BackgroundRole)
        bg = custom_bg if isinstance(custom_bg, QColor) else (self.primary_bg if row % 2 == 0 else self.alt_bg)
        if is_selected:
            painter.fillRect(option.rect, self.selection_bg)
            painter.setPen(QPen(self.selection_border, 1))
            painter.drawRect(option.rect.adjusted(0, 0, -1, -1))
        else:
            painter.fillRect(option.rect, bg)

        painter.setPen(self.text_pen)
        text = index.data(Qt.ItemDataRole.DisplayRole) or "--"
        painter.drawText(option.rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()

    def _format_span_value(self, model, row: int, start: int, length: int, col_def):
        row_bytes = model.get_row_bytes(row)
        if row_bytes.size == 0:
            return ""
        segment = row_bytes[start:start + length]
        if segment.size == 0:
            return ""
        if col_def.display_format in ("dec", "decimal"):
            return str(int.from_bytes(segment.tobytes(), byteorder="big"))
        if col_def.display_format in ("binary",):
            return "b" + "".join(f"{b:08b}" for b in segment)
        if col_def.display_format in ("ascii",):
            return "".join(chr(int(b)) if 32 <= int(b) < 127 else "." for b in segment)
        # default hex
        return "0x" + "".join(f"{int(b):02X}" for b in segment)
