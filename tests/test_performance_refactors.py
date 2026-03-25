import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from processing.cli import BitProcessorCLI
from viewer.canvas import BitCanvas
from viewer.table import ByteStructuredTableWidget


class CLISyncPatternTests(unittest.TestCase):
    def test_find_sync_pattern_preserves_overlapping_matches(self):
        cli = BitProcessorCLI()
        data = np.array([1, 1, 1], dtype=np.uint8)
        pattern = np.array([1, 1], dtype=np.uint8)

        self.assertEqual(cli.find_sync_pattern(data, pattern), [0, 1])

    def test_find_sync_pattern_respects_error_tolerance(self):
        cli = BitProcessorCLI()
        data = np.array([1, 0, 1, 1], dtype=np.uint8)
        pattern = np.array([1, 1], dtype=np.uint8)

        self.assertEqual(cli.find_sync_pattern(data, pattern, error_percent=50), [0, 1, 2])


class CanvasVirtualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_bit_canvas_caps_widget_height_but_tracks_virtual_height(self):
        canvas = BitCanvas()
        try:
            canvas.bits_per_row = 1
            bit_count = (BitCanvas.MAX_WIDGET_HEIGHT // canvas.bit_size) + 128
            canvas.set_bits(np.zeros(bit_count, dtype=np.uint8))

            self.assertGreater(canvas.virtual_content_height, BitCanvas.MAX_WIDGET_HEIGHT)
            self.assertEqual(canvas.actual_content_height, BitCanvas.MAX_WIDGET_HEIGHT)
            self.assertEqual(canvas.height(), BitCanvas.MAX_WIDGET_HEIGHT)
        finally:
            canvas.deleteLater()


class ConstantHighlightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_highlight_constant_columns_in_framed_mode(self):
        table = ByteStructuredTableWidget()
        try:
            table.set_bytes(np.array([0xAA, 0xAA, 0xAA], dtype=np.uint8))
            table.frames = [(0, 8), (8, 8), (16, 8)]
            table.update_display()

            table.highlight_constant_columns()

            self.assertIn(0, table.constant_columns)
        finally:
            table.deleteLater()


if __name__ == "__main__":
    unittest.main()
