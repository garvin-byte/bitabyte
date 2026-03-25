"""BitCanvas and LiveBitViewerCanvas rendering widgets."""

import numpy as np
from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtCore import Qt, QRect, QSize, QLine
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPalette, QFont, QPixmap, QImage
from common.bit_view_utils import build_highlight_intervals, is_bit_highlighted

class BitCanvas(QWidget):
    """Fast bit rendering widget using QPainter with tile-based QPixmap caching"""
    MAX_WIDGET_HEIGHT = 4_000_000

    def __init__(self):
        super().__init__()
        self.bits = None

        # How many bits to draw per row (what used to be self.width)
        self.bits_per_row = 64
        self.bit_size = 10

        self.start_position = 0
        self.highlighted_positions = set()
        self.highlight_intervals = []
        self.pattern_length = 0
        self.display_mode = "squares"  # "squares" or "circles"

        # Frame boundaries (list of bit positions where frames start)
        self.frame_boundaries = []
        self._frame_boundaries_arr = np.empty(0, dtype=np.int64)

        # Tile-based caching for GPU-accelerated rendering
        self.tile_cache = {}  # {tile_key: QPixmap}
        self.tile_rows = 100  # rows per tile (vertical only; horizontal is clipped per paint)
        self.cache_version = 0  # Increment to invalidate all cache
        self.virtual_content_height = 0
        self.actual_content_height = 0

        # Set background
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(255, 255, 255))
        self.setPalette(palette)

    def set_bits(self, bits):
        self.bits = bits
        self._invalidate_cache()
        self.setUpdatesEnabled(False)
        self.update_size()
        self.setUpdatesEnabled(True)
        self.update()

    def set_bits_per_row(self, n):
        """Change row width without re-assigning bit data — fast path for the spinner."""
        if self.bits_per_row == n:
            return
        self.bits_per_row = n
        self._invalidate_cache()
        # Suppress layout-triggered repaints while setMinimumSize runs so we
        # get exactly one paint (the update() below) instead of two.
        self.setUpdatesEnabled(False)
        self.update_size()
        self.setUpdatesEnabled(True)
        self.update()

    def _invalidate_cache(self):
        """Clear all cached tiles when data changes"""
        self.tile_cache.clear()
        self.cache_version += 1

    def update_size(self):
        """Resize the canvas to fit its content so the scroll area shows correct scrollbars."""
        if self.bits is None:
            return

        cols = max(1, self.bits_per_row)
        rows = (len(self.bits) + cols - 1) // cols
        height = rows * self.bit_size
        self.virtual_content_height = height
        self.actual_content_height = min(height, self.MAX_WIDGET_HEIGHT)
        width = cols * self.bit_size + 60  # Extra space for row labels
        self.resize(width, self.actual_content_height)
        self.setMinimumSize(width, self.actual_content_height)

    def _scroll_area(self):
        viewport = self.parent()
        if viewport is None:
            return None
        scroll_area = viewport.parent()
        if scroll_area is not None and hasattr(scroll_area, "verticalScrollBar"):
            return scroll_area
        return None

    def _virtual_scroll_y(self, actual_scroll_y: int, viewport_height: int) -> int:
        """Map the real scrollbar offset onto the full virtual content height."""
        if self.virtual_content_height <= self.actual_content_height:
            return actual_scroll_y

        max_actual_offset = max(0, self.actual_content_height - viewport_height)
        max_virtual_offset = max(0, self.virtual_content_height - viewport_height)
        if max_actual_offset == 0 or max_virtual_offset == 0:
            return 0
        return int(round((actual_scroll_y / max_actual_offset) * max_virtual_offset))

    def set_highlights(self, positions, pattern_len):
        self.highlighted_positions = set(positions)
        self.pattern_length = pattern_len

        # OPTIMIZED: Build sorted intervals for fast lookup
        self._build_highlight_intervals()
        self._invalidate_cache()
        self.update()

    def _build_highlight_intervals(self):
        """Build sorted list of highlight intervals for O(log n) lookup"""
        self.highlight_intervals = build_highlight_intervals(
            self.highlighted_positions,
            self.pattern_length,
        )

    def _is_bit_highlighted(self, bit_index):
        """Fast O(log n) binary search to check if bit is highlighted"""
        return is_bit_highlighted(self.highlight_intervals, bit_index)

    def clear_highlights(self):
        self.highlighted_positions.clear()
        self.pattern_length = 0
        self.highlight_intervals = []
        self._invalidate_cache()
        self.update()

    def set_frame_boundaries(self, boundaries):
        """Set frame boundary positions (list of bit positions where frames start)"""
        self.frame_boundaries = boundaries if boundaries else []
        # Pre-convert to numpy for fast tile-range filtering in _render_tile
        self._frame_boundaries_arr = (
            np.array(self.frame_boundaries, dtype=np.int64)
            if self.frame_boundaries else np.empty(0, dtype=np.int64)
        )
        self._invalidate_cache()
        self.update()

    def _visible_rect(self):
        """Return the rectangle (in widget coords) that is actually on screen.
        Uses the scroll area viewport so full-widget repaints don't re-render
        hidden columns."""
        viewport = self.parent()
        if viewport is not None:
            sa = viewport.parent()
            if sa is not None and hasattr(sa, 'horizontalScrollBar'):
                sx = sa.horizontalScrollBar().value()
                sy = sa.verticalScrollBar().value()
                return QRect(sx, sy, viewport.width(), viewport.height())
        # Fallback: cap to a reasonable viewport width so we don't render the
        # entire stretched canvas (setWidgetResizable may make self.width() huge).
        cap_w = min(self.width(), 2000)
        cap_h = min(self.height(), 2000)
        return QRect(0, 0, cap_w, cap_h)

    def paintEvent(self, event):
        if self.bits is None:
            return

        painter = QPainter(self)
        bs = self.bit_size
        label_w = 55
        cols = max(1, self.bits_per_row)
        rows = (len(self.bits) + cols - 1) // cols

        # Intersect the dirty rect with the actual on-screen viewport so that
        # a full-widget repaint doesn't render all columns (only visible ones).
        visible_rect = event.rect().intersected(self._visible_rect())
        if visible_rect.isEmpty():
            visible_rect = event.rect()

        scroll_area = self._scroll_area()
        viewport_height = scroll_area.viewport().height() if scroll_area is not None else visible_rect.height()
        virtual_visible_top = self._virtual_scroll_y(visible_rect.top(), viewport_height)
        virtual_visible_bottom = virtual_visible_top + visible_rect.height()

        # Visible column range — only render what's horizontally on screen
        col_start = max(0, (visible_rect.left() - label_w) // bs)
        col_end   = min(cols, (visible_rect.right() - label_w) // bs + 1)
        # If the label area is visible, always start from col 0
        if visible_rect.left() < label_w:
            col_start = 0
        col_end = max(col_start + 1, col_end)
        # Visible tile row range
        start_tile = max(0, virtual_visible_top // (self.tile_rows * bs))
        end_tile   = min((rows + self.tile_rows - 1) // self.tile_rows,
                         virtual_visible_bottom // (self.tile_rows * bs) + 1)

        for tile_idx in range(start_tile, end_tile):
            tile_virtual_y = tile_idx * self.tile_rows * bs
            tile_y   = int(round(visible_rect.top() + (tile_virtual_y - virtual_visible_top)))
            tile_key = (tile_idx, col_start, col_end, self.cache_version)
            if tile_key not in self.tile_cache:
                self.tile_cache[tile_key] = self._render_tile(
                    tile_idx, cols, col_start, col_end)

            # Tile is drawn starting at the label edge (x=0) when col_start==0,
            # otherwise at the first visible data column's pixel position.
            tile_x = 0 if col_start == 0 else (label_w + col_start * bs)
            painter.drawPixmap(tile_x, tile_y, self.tile_cache[tile_key])

    def _render_tile(self, tile_idx, cols, col_start=0, col_end=None):
        """Render a tile covering row-tile tile_idx and columns [col_start, col_end).
        Only the visible column slice is rendered, keeping numpy arrays small."""
        if col_end is None:
            col_end = cols

        start_row = tile_idx * self.tile_rows
        end_row   = min(start_row + self.tile_rows, (len(self.bits) + cols - 1) // cols)
        n_rows    = end_row - start_row
        bs        = self.bit_size
        label_w   = 55

        n_vis_cols   = col_end - col_start
        include_label = (col_start == 0)
        data_x        = label_w if include_label else 0
        tile_w        = data_x + n_vis_cols * bs
        tile_h        = n_rows * bs

        if self.display_mode != "circles":
            # --- Fast numpy path for squares (column-clipped) ---
            start_bit = start_row * cols
            end_bit   = min(end_row * cols, len(self.bits))
            n_bits    = end_bit - start_bit
            tile_bits = self.bits[start_bit:end_bit]

            full_size = n_rows * cols
            if n_bits < full_size:
                grid_bits = np.zeros(full_size, dtype=np.uint8)
                grid_bits[:n_bits] = tile_bits
            else:
                grid_bits = tile_bits

            # Slice only the visible columns — this is what keeps arrays small
            grid_2d   = grid_bits.reshape(n_rows, cols)
            vis_slice = grid_2d[:, col_start:col_end]          # (n_rows, n_vis_cols)
            gray      = ((1 - vis_slice) * 255).astype(np.uint8)
            scaled    = np.repeat(np.repeat(gray, bs, axis=0), bs, axis=1)

            img = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)
            img[:, data_x:, 0] = scaled
            img[:, data_x:, 1] = scaled
            img[:, data_x:, 2] = scaled

            if bs >= 4:
                img[::bs, data_x:] = 180
                img[:, data_x::bs] = 180

            # Highlights — only for bits in the visible column range
            if self.highlight_intervals and n_bits > 0:
                highlighted = np.zeros(full_size, dtype=bool)
                for hs, he in self.highlight_intervals:
                    if he <= start_bit or hs >= start_bit + full_size:
                        continue
                    highlighted[max(0, hs - start_bit):min(full_size, he - start_bit)] = True
                if highlighted.any():
                    h_idx  = np.where(highlighted)[0]
                    h_vals = grid_bits[h_idx % n_bits] if n_bits < full_size else grid_bits[h_idx]
                    lr = h_idx // cols
                    lc = h_idx % cols
                    for i in range(len(h_idx)):
                        c = lc[i]
                        if c < col_start or c >= col_end:
                            continue
                        y0 = lr[i] * bs
                        x0 = (c - col_start) * bs + data_x
                        color = (255, 0, 0) if h_vals[i] == 1 else (255, 255, 0)
                        img[y0:y0 + bs - 1, x0:x0 + bs - 1] = color

            img_c   = np.ascontiguousarray(img)
            q_image = QImage(img_c.data, tile_w, tile_h, tile_w * 3,
                             QImage.Format.Format_RGB888)
            pixmap  = QPixmap.fromImage(q_image)

            painter = QPainter(pixmap)
            if include_label:
                painter.setPen(QColor(100, 100, 100))
                for row in range(n_rows):
                    painter.drawText(
                        QRect(0, row * bs, label_w - 3, bs),
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                        str((start_row + row) * cols),
                    )
            if len(self._frame_boundaries_arr) > 0:
                arr = self._frame_boundaries_arr
                b_rows = arr // cols
                in_tile = (arr > 0) & (b_rows >= start_row) & (b_rows < end_row)
                if in_tile.any():
                    t_arr  = arr[in_tile]
                    t_rows = b_rows[in_tile]
                    t_cols = t_arr % cols
                    h_mask = t_cols == 0
                    v_mask = (~h_mask) & (t_cols >= col_start) & (t_cols < col_end)
                    lines = []
                    for y in ((t_rows[h_mask] - start_row) * bs).tolist():
                        lines.append(QLine(data_x, y, tile_w, y))
                    v_y = ((t_rows[v_mask] - start_row) * bs).tolist()
                    v_x = ((t_cols[v_mask] - col_start) * bs + data_x).tolist()
                    for x, y in zip(v_x, v_y):
                        lines.append(QLine(x - 1, y, x - 1, y + bs))
                    if lines:
                        painter.setPen(QPen(QColor(255, 0, 0), 2))
                        painter.drawLines(lines)
            painter.end()
            return pixmap

        else:
            # --- QPainter path for circles (column-clipped) ---
            pixmap = QPixmap(tile_w, tile_h)
            pixmap.fill(QColor(255, 255, 255))
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            for row in range(start_row, end_row):
                local_y = (row - start_row) * bs
                if include_label:
                    painter.setPen(QColor(100, 100, 100))
                    painter.drawText(
                        QRect(0, local_y, label_w - 3, bs),
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                        str(row * cols),
                    )
                for col in range(col_start, col_end):
                    bit_index = row * cols + col
                    if bit_index >= len(self.bits):
                        break
                    bit = self.bits[bit_index]
                    is_highlighted = self._is_bit_highlighted(bit_index)
                    if is_highlighted:
                        fill_color   = QColor(255, 0, 0) if bit == 1 else QColor(255, 255, 0)
                        outline_color = QColor(139, 0, 0) if bit == 1 else QColor(255, 165, 0)
                    else:
                        fill_color   = QColor(0, 0, 0) if bit == 1 else QColor(255, 255, 255)
                        outline_color = QColor(100, 100, 100)
                    painter.setPen(QPen(outline_color, 1))
                    painter.setBrush(QBrush(fill_color))
                    x = (col - col_start) * bs + data_x
                    painter.drawEllipse(x, local_y, bs - 1, bs - 1)

            if len(self._frame_boundaries_arr) > 0:
                arr = self._frame_boundaries_arr
                b_rows = arr // cols
                in_tile = (arr > 0) & (b_rows >= start_row) & (b_rows < end_row)
                if in_tile.any():
                    t_arr  = arr[in_tile]
                    t_rows = b_rows[in_tile]
                    t_cols = t_arr % cols
                    h_mask = t_cols == 0
                    v_mask = (~h_mask) & (t_cols >= col_start) & (t_cols < col_end)
                    lines = []
                    for y in ((t_rows[h_mask] - start_row) * bs).tolist():
                        lines.append(QLine(data_x, y, tile_w, y))
                    v_y = ((t_rows[v_mask] - start_row) * bs).tolist()
                    v_x = ((t_cols[v_mask] - col_start) * bs + data_x).tolist()
                    for x, y in zip(v_x, v_y):
                        lines.append(QLine(x - 1, y, x - 1, y + bs))
                    if lines:
                        painter.setPen(QPen(QColor(255, 0, 0), 2))
                        painter.drawLines(lines)
            painter.end()
            return pixmap


class LiveBitViewerCanvas(QWidget):
    """Live bit viewer for selected columns in framed byte mode."""

    def __init__(self):
        super().__init__()
        self.frames_bits = []
        self.bit_size = 8
        self.display_mode = "squares"
        self.max_frame_bits = 0
        self.size_value = 8

        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(255, 255, 255))
        self.setPalette(palette)
        self.setMinimumSize(300, 200)

    def set_display_mode(self, mode):
        """Set how the live viewer renders selected bits."""
        if mode == self.display_mode:
            return
        self.display_mode = mode
        self.update_size()
        self.update()

    def set_size_value(self, size_value):
        """Set the requested size scaling for the live viewer."""
        size_value = max(4, int(size_value))
        if size_value == self.size_value:
            return
        self.size_value = size_value
        if self.frames_bits:
            self.set_frame_bits(self.frames_bits)
        else:
            self.update()

    def set_frame_bits(self, frames_bits):
        """Set bits to display. frames_bits: list of numpy arrays, one per frame."""
        self.frames_bits = frames_bits

        if not frames_bits:
            self.max_frame_bits = 0
            self.update()
            return

        self.max_frame_bits = max(len(fb) for fb in frames_bits)

        if self.display_mode in {"binary", "hex"}:
            self.bit_size = max(6, self.size_value)
        else:
            available_width = self.width() - 60
            if available_width < 100:
                available_width = 300
            if self.max_frame_bits > 0:
                fitted_size = max(3, min(12, available_width // self.max_frame_bits))
                self.bit_size = min(max(3, self.size_value), fitted_size)
            else:
                self.bit_size = self.size_value

        self.update_size()
        self.update()

    def update_size(self):
        if not self.frames_bits or self.max_frame_bits == 0:
            self.setMinimumSize(300, 100)
            return
        if self.display_mode == "binary":
            width = max(300, 55 + (self.max_frame_bits * max(7, self.bit_size // 2 + 3)))
            height = len(self.frames_bits) * (self.bit_size + 8)
        elif self.display_mode == "hex":
            hex_chars = max(1, (self.max_frame_bits + 3) // 4)
            width = max(300, 55 + (hex_chars * max(9, self.bit_size + 1)))
            height = len(self.frames_bits) * (self.bit_size + 8)
        else:
            width = self.max_frame_bits * self.bit_size + 60
            height = len(self.frames_bits) * self.bit_size
        self.setMinimumSize(width, height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.frames_bits and self.max_frame_bits > 0 and self.display_mode in {"squares", "circles"}:
            available_width = event.size().width() - 60
            if available_width > 0:
                new_bit_size = min(max(3, self.size_value), max(3, min(12, available_width // self.max_frame_bits)))
                if new_bit_size != self.bit_size:
                    self.bit_size = new_bit_size
                    self.update_size()
                    self.update()

    def paintEvent(self, event):
        if not self.frames_bits:
            painter = QPainter(self)
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Select columns in byte table\nto view bits")
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for frame_idx, frame_bits in enumerate(self.frames_bits):
            row_height = self.bit_size if self.display_mode in {"squares", "circles"} else (self.bit_size + 8)
            y = frame_idx * row_height

            painter.setPen(QColor(100, 100, 100))
            painter.setFont(QFont("Courier", 7))
            painter.drawText(
                QRect(0, y, 50, row_height),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"F{frame_idx}"
            )

            padded_bits = np.copy(frame_bits)
            if len(padded_bits) < self.max_frame_bits:
                padding = np.zeros(self.max_frame_bits - len(padded_bits), dtype=np.uint8)
                padded_bits = np.concatenate([padded_bits, padding])

            if self.display_mode in {"binary", "hex"}:
                painter.setPen(QColor(20, 20, 20))
                painter.setFont(QFont("Consolas", max(8, self.bit_size)))
                if self.display_mode == "binary":
                    text = "".join(str(int(bit)) for bit in frame_bits.tolist())
                else:
                    padded_len = ((len(frame_bits) + 3) // 4) * 4
                    text_bits = np.pad(frame_bits, (0, padded_len - len(frame_bits)), constant_values=0)
                    chars = []
                    for idx in range(0, len(text_bits), 4):
                        nibble = text_bits[idx:idx + 4]
                        value = (int(nibble[0]) << 3) | (int(nibble[1]) << 2) | (int(nibble[2]) << 1) | int(nibble[3])
                        chars.append(f"{value:X}")
                    text = "".join(chars)
                painter.drawText(
                    QRect(55, y, max(200, self.width() - 60), row_height),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    text,
                )
                continue

            for bit_idx in range(len(padded_bits)):
                bit = padded_bits[bit_idx]
                is_padding = bit_idx >= len(frame_bits)

                if is_padding:
                    fill_color = QColor(220, 220, 220)
                    outline_color = QColor(180, 180, 180)
                elif bit == 1:
                    fill_color = QColor(0, 0, 0)
                    outline_color = QColor(0, 0, 0)
                else:
                    fill_color = QColor(255, 255, 255)
                    outline_color = QColor(100, 100, 100)

                x = bit_idx * self.bit_size + 55
                painter.setPen(QPen(outline_color, 1))
                painter.setBrush(QBrush(fill_color))

                if self.display_mode == "circles":
                    painter.drawEllipse(x, y, self.bit_size - 1, self.bit_size - 1)
                else:
                    painter.drawRect(x, y, self.bit_size - 1, self.bit_size - 1)

