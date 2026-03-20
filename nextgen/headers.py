"""Custom header view that can paint multiple rows."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QHeaderView

from .models import HeaderModel


class MultiRowHeaderView(QHeaderView):
    """Horizontal header that consults HeaderModel to render stacked labels."""

    def __init__(self, orientation: Qt.Orientation, parent=None):
        super().__init__(orientation, parent)
        self._model: Optional[HeaderModel] = None
        self.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSectionsClickable(True)
        self.setStretchLastSection(False)

    def setHeaderModel(self, model: HeaderModel) -> None:
        self._model = model
        total_height = sum(model.height_for(i) for i in range(model.row_count))
        self.setFixedHeight(total_height)
        self.viewport().update()

    def paintSection(self, painter: QPainter, rect: QRect, logicalIndex: int) -> None:  # noqa: N802
        if self.orientation() != Qt.Orientation.Horizontal or rect.isEmpty():
            super().paintSection(painter, rect, logicalIndex)
            return

        painter.save()
        painter.fillRect(rect, QColor(226, 246, 255))
        painter.setPen(QPen(QColor(150, 200, 220)))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))

        if self._model is None or self._model.row_count == 0:
            painter.restore()
            return

        band_colors = [
            QColor(222, 242, 255),
            QColor(210, 236, 255),
            QColor(198, 230, 255),
        ]
        text_pen = QColor(45, 70, 90)
        divider_pen = QPen(QColor(170, 200, 220))

        y = rect.top()
        for band_index in range(self._model.row_count):
            height = self._model.height_for(band_index)
            band = self._model.bands[band_index]
            color = band_colors[band_index % len(band_colors)]
            span = None
            if band.spans:
                for start, length, text in band.spans:
                    if start <= logicalIndex < start + length:
                        span = (start, length, text)
                        break

            if span:
                start, length, text = span
                left = self.sectionViewportPosition(start)
                end = min(start + length, self.model().columnCount())
                span_width = sum(self.sectionSize(col) for col in range(start, end))
                span_rect = QRect(left, y, span_width, height)
                painter.fillRect(span_rect, color)
                painter.setPen(text_pen)
                painter.drawText(span_rect, Qt.AlignmentFlag.AlignCenter, text)
                painter.setPen(divider_pen)
                painter.drawLine(left, y, left, y + height)
                painter.drawLine(left + span_width, y, left + span_width, y + height)
            else:
                band_rect = QRect(rect.left(), y, rect.width(), height)
                painter.fillRect(band_rect, color)
                painter.setPen(text_pen)
                label = self._model.label_for(band_index, logicalIndex)
                painter.drawText(band_rect, Qt.AlignmentFlag.AlignCenter, label)

            y += height
            painter.setPen(divider_pen)
            painter.drawLine(rect.left(), y - 1, rect.right(), y - 1)

        # vertical separator
        painter.setPen(QPen(QColor(160, 190, 210)))
        painter.drawLine(rect.right(), rect.top(), rect.right(), rect.bottom())
        painter.restore()
