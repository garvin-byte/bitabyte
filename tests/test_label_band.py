import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from nextgen.data import ByteDataSource
from nextgen.main_window import NextGenBitViewerWindow
from nextgen.models import ByteTableModel, ColumnDefinition
from PyQt6.QtWidgets import QApplication


class LabelBandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_normalized_byte_span_clamps_to_visible_width(self):
        definition = ColumnDefinition(
            start_byte=2,
            end_byte=6,
            label="Field",
            display_format="hex",
        )

        self.assertEqual(definition.normalized_byte_span(4), (2, 3))

    def test_model_uses_normalized_spans_for_lookup_and_backgrounds(self):
        model = ByteTableModel(ByteDataSource(bytes_per_row=4))
        definition = ColumnDefinition(
            start_byte=1,
            end_byte=8,
            label="Sync",
            display_format="hex",
            color_name="Sky",
        )

        model.column_definitions.append(definition)
        model.notify_column_definitions_changed()

        self.assertEqual(model.span_for_column(1)[:2], (1, 3))
        self.assertIsNotNone(model._background_for_column(3))
        self.assertIsNone(model._background_for_column(0))

    def test_window_label_band_reuses_normalized_spans(self):
        window = NextGenBitViewerWindow()
        try:
            window.model.column_definitions.append(
                ColumnDefinition(
                    start_byte=0,
                    end_byte=3,
                    label="Sync",
                    display_format="hex",
                )
            )

            labels, spans = window._build_label_band(2)

            self.assertEqual(labels, ["Sync", "Sync"])
            self.assertEqual(spans, [(0, 2, "Sync")])
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
