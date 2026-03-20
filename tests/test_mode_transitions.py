import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtWidgets import QApplication, QMessageBox

from viewer.window import BitViewerWindow


class ModeTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = BitViewerWindow()

    def tearDown(self):
        self.window.close()

    def _load_framed_byte_state(self):
        self.window.bytes_data = np.array([0xAA, 0xBB, 0xCC], dtype=np.uint8)
        self.window.byte_table.bytes_data = self.window.bytes_data
        self.window.byte_table.frames = [(0, 8), (8, 16)]
        self.window.byte_table.frame_pattern = {
            "byte_values": [0xAA],
            "bit_values": [1, 0, 1, 0, 1, 0, 1, 0],
            "search_mode": "byte",
        }
        self.window.data_mode = "byte"
        self.window.byte_mode_radio.blockSignals(True)
        self.window.bit_mode_radio.blockSignals(True)
        self.window.byte_mode_radio.setChecked(True)
        self.window.bit_mode_radio.setChecked(False)
        self.window.byte_mode_radio.blockSignals(False)
        self.window.bit_mode_radio.blockSignals(False)

    def test_refresh_byte_mode_view_preserves_frames(self):
        self._load_framed_byte_state()

        original_frames = list(self.window.byte_table.frames)
        original_pattern = dict(self.window.byte_table.frame_pattern)

        self.window.update_display()

        self.assertEqual(self.window.byte_table.frames, original_frames)
        self.assertEqual(self.window.byte_table.frame_pattern, original_pattern)

    def test_zero_padded_round_trip_restores_framed_byte_mode(self):
        self._load_framed_byte_state()

        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            self.window.bit_mode_radio.setChecked(True)
            QApplication.processEvents()

        self.assertEqual(self.window.data_mode, "bit")
        self.assertTrue(self.window._return_to_byte_framed)
        self.assertEqual(self.window.width_spin.value(), 16)
        self.assertEqual(len(self.window.bits), 32)

        self.window.byte_mode_radio.setChecked(True)
        QApplication.processEvents()

        self.assertEqual(self.window.data_mode, "byte")
        self.assertIsNotNone(self.window.byte_table.frames)
        self.assertEqual(self.window.byte_table.frames, [(0, 16), (16, 16)])
        self.assertFalse(self.window.row_size_spin.isEnabled())
        self.assertIsNotNone(self.window.byte_table.frame_pattern)


if __name__ == "__main__":
    unittest.main()
