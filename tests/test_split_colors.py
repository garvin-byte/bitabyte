import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from viewer.column import ColumnDefinition
from viewer.table import ByteStructuredTableWidget


class SplitColorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_split_auto_color_avoids_existing_column_colors(self):
        table = ByteStructuredTableWidget()
        try:
            table.column_definitions.append(
                ColumnDefinition(
                    start_byte=0,
                    end_byte=0,
                    label="Sync",
                    display_format="hex",
                    color_name="Sky",
                )
            )

            self.assertEqual(table._next_split_color_name(), "Coral")
        finally:
            table.deleteLater()


if __name__ == "__main__":
    unittest.main()
