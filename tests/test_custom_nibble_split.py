import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from viewer.column import ColumnDefinition
from viewer.table import ByteStructuredTableWidget
from viewer.window import BitViewerWindow


class CustomNibbleSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_custom_nibble_relabels_and_recolors_segments(self):
        table = ByteStructuredTableWidget()
        try:
            table.set_bytes(np.array([0xCA], dtype=np.uint8))
            table._split_columns({0}, "binary")
            self.assertEqual(table.split_columns[0]["label"], "BIN 0")
            self.assertEqual(table.split_columns[0]["color"], "Sky")

            table._split_columns({2, 3, 4, 5}, "nibble")
            table.update_display()

            data_row = table.HEADER_ROW_COUNT
            split_info = table.split_columns[0]
            self.assertEqual([segment["label"] for segment in split_info["segments"]], ["BIN 0", "NIBBLE 0", "BIN 1"])
            self.assertEqual([segment["color"] for segment in split_info["segments"]], ["Sky", "Coral", "Mint"])
            self.assertEqual(table.columnCount(), 5)
            self.assertEqual(table.item(0, 0).text(), "BIN 0")
            self.assertEqual(table.item(0, 2).text(), "NIBBLE 0")
            self.assertEqual(table.item(0, 3).text(), "BIN 1")
            self.assertEqual(table.item(2, 0).text(), "7-7")
            self.assertEqual(table.item(2, 1).text(), "6-6")
            self.assertEqual(table.item(2, 2).text(), "2-5")
            self.assertEqual(table.item(2, 3).text(), "1-1")
            self.assertEqual(table.item(2, 4).text(), "0-0")
            self.assertEqual(table.item(data_row, 0).text(), "1")
            self.assertEqual(table.item(data_row, 1).text(), "1")
            self.assertEqual(table.item(data_row, 2).text(), "2")
            self.assertEqual(table.item(data_row, 3).text(), "1")
            self.assertEqual(table.item(data_row, 4).text(), "0")
            self.assertEqual(table.item(data_row, 0).background().color(), table._color_from_name("Sky"))
            self.assertEqual(table.item(data_row, 1).background().color(), table._color_from_name("Sky"))
            self.assertEqual(table.item(data_row, 2).background().color(), table._color_from_name("Coral"))
            self.assertEqual(table.item(data_row, 3).background().color(), table._color_from_name("Mint"))
            self.assertEqual(table.item(data_row, 4).background().color(), table._color_from_name("Mint"))
        finally:
            table.deleteLater()

    def test_bit_definition_preserves_remaining_custom_split_groups(self):
        table = ByteStructuredTableWidget()
        try:
            table.set_bytes(np.array([0xCA], dtype=np.uint8))
            table.split_columns[0] = {
                "type": "custom_bits",
                "label": "Nibble",
                "color": "Coral",
                "segments": [
                    {"start": 4, "end": 7, "format": "binary", "label": "BIN 0", "color": "Sky"},
                    {"start": 0, "end": 3, "format": "hex", "label": "NIBBLE 0", "color": "Coral"},
                ],
            }
            table.column_definitions.append(
                ColumnDefinition(
                    start_byte=0,
                    end_byte=0,
                    label="test",
                    display_format="binary",
                    color_name="Gold",
                    unit="bit",
                    start_bit=0,
                    total_bits=2,
                )
            )

            table.update_display()

            data_row = table.HEADER_ROW_COUNT
            self.assertEqual(table.columnCount(), 5)
            self.assertEqual(table.item(0, 0).text(), "test")
            self.assertEqual(table.columnSpan(0, 0), 2)
            self.assertEqual(table.item(0, 2).text(), "BIN 0")
            self.assertEqual(table.columnSpan(0, 2), 2)
            self.assertEqual(table.item(0, 4).text(), "NIBBLE 0")
            self.assertEqual(table.item(2, 0).text(), "7-7")
            self.assertEqual(table.item(2, 1).text(), "6-6")
            self.assertEqual(table.item(2, 2).text(), "5-5")
            self.assertEqual(table.item(2, 3).text(), "4-4")
            self.assertEqual(table.item(2, 4).text(), "0-3")
            self.assertEqual(table.columnSpan(data_row, 0), 2)
            self.assertEqual(table.item(data_row, 0).text(), "11")
            self.assertEqual(table.item(data_row, 2).text(), "0")
            self.assertEqual(table.item(data_row, 3).text(), "0")
            self.assertEqual(table.item(data_row, 4).text(), "A")
            self.assertEqual(table.item(data_row, 0).background().color(), table._color_from_name("Gold"))
            self.assertEqual(table.item(data_row, 2).background().color(), table._color_from_name("Sky"))
            self.assertEqual(table.item(data_row, 3).background().color(), table._color_from_name("Sky"))
            self.assertEqual(table.item(data_row, 4).background().color(), table._color_from_name("Coral"))
        finally:
            table.deleteLater()

    def test_removing_custom_nibble_restores_binary_group(self):
        table = ByteStructuredTableWidget()
        try:
            table.set_bytes(np.array([0xCA], dtype=np.uint8))
            table.split_columns[0] = {
                "type": "custom_bits",
                "label": "Nibble",
                "color": "Coral",
                "segments": [
                    {"start": 4, "end": 7, "format": "binary", "label": "BIN 0", "color": "Sky"},
                    {"start": 0, "end": 3, "format": "hex", "label": "NIBBLE 0", "color": "Coral"},
                ],
            }

            table._remove_split_entry(("segment", 0, 0, 3))
            table.update_display()

            split_info = table.split_columns[0]
            self.assertEqual(
                [(segment["format"], segment["label"], segment["start"], segment["end"]) for segment in split_info["segments"]],
                [("binary", "BIN 0", 4, 7), ("binary", "BIN 1", 0, 3)],
            )
            self.assertEqual(table.columnCount(), 8)
            self.assertEqual(table.item(0, 0).text(), "BIN 0")
            self.assertEqual(table.columnSpan(0, 0), 4)
            self.assertEqual(table.item(0, 4).text(), "BIN 1")
            self.assertEqual(table.columnSpan(0, 4), 4)
            self.assertTrue(all(table.item(2, idx).text() != "?" for idx in range(8)))
        finally:
            table.deleteLater()

    def test_removing_custom_nibble_keeps_other_definitions_and_binary_groups(self):
        table = ByteStructuredTableWidget()
        try:
            table.set_bytes(np.array([0xCA], dtype=np.uint8))
            table.split_columns[0] = {
                "type": "custom_bits",
                "label": "Nibble",
                "color": "Coral",
                "segments": [
                    {"start": 4, "end": 7, "format": "binary", "label": "BIN 0", "color": "Sky"},
                    {"start": 0, "end": 3, "format": "hex", "label": "NIBBLE 0", "color": "Coral"},
                ],
            }
            table.column_definitions.append(
                ColumnDefinition(
                    start_byte=0,
                    end_byte=0,
                    label="test",
                    display_format="binary",
                    color_name="Gold",
                    unit="bit",
                    start_bit=0,
                    total_bits=2,
                )
            )

            table._remove_split_entry(("segment", 0, 0, 3))
            table.update_display()

            data_row = table.HEADER_ROW_COUNT
            self.assertEqual(table.columnCount(), 8)
            self.assertEqual(table.item(0, 0).text(), "test")
            self.assertEqual(table.columnSpan(0, 0), 2)
            self.assertEqual(table.item(0, 2).text(), "BIN 0")
            self.assertEqual(table.columnSpan(0, 2), 2)
            self.assertEqual(table.item(0, 4).text(), "BIN 1")
            self.assertEqual(table.columnSpan(0, 4), 4)
            self.assertEqual(table.item(data_row, 0).text(), "11")
            self.assertEqual(table.item(data_row, 2).text(), "0")
            self.assertEqual(table.item(data_row, 3).text(), "0")
            self.assertEqual(table.item(data_row, 4).text(), "1")
            self.assertEqual(table.item(data_row, 7).text(), "0")
        finally:
            table.deleteLater()

    def test_removing_middle_custom_nibble_uses_next_chronological_binary_label(self):
        table = ByteStructuredTableWidget()
        try:
            table.set_bytes(np.array([0xCA], dtype=np.uint8))
            table.split_columns[0] = {
                "type": "custom_bits",
                "label": "Nibble",
                "color": "Coral",
                "segments": [
                    {"start": 6, "end": 7, "format": "binary", "label": "BIN 0", "color": "Sky"},
                    {"start": 2, "end": 5, "format": "hex", "label": "NIBBLE 0", "color": "Coral"},
                    {"start": 0, "end": 1, "format": "binary", "label": "BIN 1", "color": "Mint"},
                ],
            }

            table._remove_split_entry(("segment", 0, 2, 5))

            split_info = table.split_columns[0]
            self.assertEqual(
                [(segment["format"], segment["label"], segment["start"], segment["end"]) for segment in split_info["segments"]],
                [("binary", "BIN 0", 6, 7), ("binary", "BIN 2", 2, 5), ("binary", "BIN 1", 0, 1)],
            )
        finally:
            table.deleteLater()

    def test_binary_split_bit_definition_keeps_remaining_bit_columns(self):
        table = ByteStructuredTableWidget()
        try:
            table.set_bytes(np.array([0xCA], dtype=np.uint8))
            table._split_columns({0}, "binary")
            table.column_definitions.append(
                ColumnDefinition(
                    start_byte=0,
                    end_byte=0,
                    label="test",
                    display_format="dec_be",
                    color_name="Gold",
                    unit="bit",
                    start_bit=0,
                    total_bits=2,
                )
            )

            table.update_display()

            data_row = table.HEADER_ROW_COUNT
            self.assertEqual(table.columnCount(), 8)
            self.assertEqual(table.item(0, 0).text(), "test")
            self.assertEqual(table.columnSpan(0, 0), 2)
            self.assertEqual(table.item(0, 2).text(), "BIN 0")
            self.assertEqual(table.columnSpan(0, 2), 6)
            self.assertEqual(table.item(2, 0).text(), "7-7")
            self.assertEqual(table.item(2, 1).text(), "6-6")
            self.assertEqual(table.item(2, 2).text(), "5-5")
            self.assertEqual(table.item(2, 7).text(), "0-0")
            self.assertEqual(table.item(data_row, 0).text(), "3")
            self.assertEqual(table.item(data_row, 2).text(), "0")
            self.assertEqual(table.item(data_row, 3).text(), "0")
            self.assertEqual(table.item(data_row, 4).text(), "1")
            self.assertEqual(table.item(data_row, 5).text(), "0")
            self.assertEqual(table.item(data_row, 6).text(), "1")
            self.assertEqual(table.item(data_row, 7).text(), "0")
        finally:
            table.deleteLater()


class CombineSelectedColumnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = BitViewerWindow()

    def tearDown(self):
        self.window.close()

    def test_combine_selected_binary_definitions_stays_binary(self):
        self.window.byte_table.set_bytes(np.array([0xCA], dtype=np.uint8))
        self.window.byte_table.column_definitions = [
            ColumnDefinition(
                start_byte=0,
                end_byte=0,
                label="BIN A",
                display_format="binary",
                color_name="Gold",
                unit="bit",
                start_bit=0,
                total_bits=2,
            ),
            ColumnDefinition(
                start_byte=0,
                end_byte=0,
                label="BIN B",
                display_format="binary",
                color_name="Mint",
                unit="bit",
                start_bit=2,
                total_bits=3,
            ),
        ]
        self.window.refresh_column_definitions_list()

        self.window.columns_list.item(0).setSelected(True)
        self.window.columns_list.item(1).setSelected(True)
        self.window.combine_selected_column_definitions()

        self.assertEqual(len(self.window.byte_table.column_definitions), 1)
        combined = self.window.byte_table.column_definitions[0]
        self.assertEqual(combined.start_bit, 0)
        self.assertEqual(combined.total_bits, 5)
        self.assertEqual(combined.display_format, "binary")
        self.assertEqual(combined.color_name, "Gold")
        self.assertEqual(combined.label, "Combo 0")

    def test_combine_selected_binary_and_nibble_defaults_to_hex(self):
        self.window.byte_table.set_bytes(np.array([0xCA], dtype=np.uint8))
        self.window.byte_table._split_columns({0}, "binary")
        self.window.byte_table._split_columns({4, 5, 6, 7}, "nibble")
        self.window.refresh_column_definitions_list()

        self.window.columns_list.item(0).setSelected(True)
        self.window.columns_list.item(1).setSelected(True)
        self.window.combine_selected_column_definitions()

        self.assertEqual(len(self.window.byte_table.column_definitions), 1)
        combined = self.window.byte_table.column_definitions[0]
        self.assertEqual(combined.start_bit, 0)
        self.assertEqual(combined.total_bits, 8)
        self.assertEqual(combined.display_format, "hex_be")
        self.assertEqual(combined.color_name, "Sky")
        self.assertEqual(combined.label, "Combo 0")
        self.assertEqual(self.window.columns_list.count(), 1)
        self.assertIn("Combo 0", self.window.columns_list.item(0).text())

    def test_combine_selected_visible_table_columns_creates_combo_label(self):
        self.window.byte_table.set_bytes(np.array([0xCA], dtype=np.uint8))
        self.window.byte_table._split_columns({0}, "binary")
        self.window.byte_table.update_display()

        self.assertTrue(self.window.can_combine_selected_table_columns({0, 1}))

        self.window.combine_selected_table_columns({0, 1})

        self.assertEqual(len(self.window.byte_table.column_definitions), 1)
        combined = self.window.byte_table.column_definitions[0]
        self.assertEqual(combined.start_bit, 0)
        self.assertEqual(combined.total_bits, 2)
        self.assertEqual(combined.display_format, "binary")
        self.assertEqual(combined.label, "Combo 0")


if __name__ == "__main__":
    unittest.main()
