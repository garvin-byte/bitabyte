import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QApplication, QDialog

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
            self.assertEqual(table.item(2, 0).text(), "0")
            self.assertEqual(table.item(2, 1).text(), "1")
            self.assertEqual(table.item(2, 2).text(), "2-5")
            self.assertEqual(table.item(2, 3).text(), "6")
            self.assertEqual(table.item(2, 4).text(), "7")
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
            self.assertEqual(table.item(2, 0).text(), "0")
            self.assertEqual(table.item(2, 1).text(), "1")
            self.assertEqual(table.item(2, 2).text(), "2")
            self.assertEqual(table.item(2, 3).text(), "3")
            self.assertEqual(table.item(2, 4).text(), "4-7")
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
            self.assertEqual(table.item(2, 0).text(), "0")
            self.assertEqual(table.item(2, 1).text(), "1")
            self.assertEqual(table.item(2, 2).text(), "2")
            self.assertEqual(table.item(2, 7).text(), "7")
            self.assertEqual(table.item(data_row, 0).text(), "3")
            self.assertEqual(table.item(data_row, 2).text(), "0")
            self.assertEqual(table.item(data_row, 3).text(), "0")
            self.assertEqual(table.item(data_row, 4).text(), "1")
            self.assertEqual(table.item(data_row, 5).text(), "0")
            self.assertEqual(table.item(data_row, 6).text(), "1")
            self.assertEqual(table.item(data_row, 7).text(), "0")
        finally:
            table.deleteLater()

    def test_simple_binary_split_does_not_keep_duplicate_byte_column(self):
        table = ByteStructuredTableWidget()
        try:
            table.set_row_size(2)
            table.set_bytes(np.array([0xCA, 0x55], dtype=np.uint8))
            table._split_columns({0}, "binary")

            data_row = table.HEADER_ROW_COUNT
            split_color = table._color_from_name(table.split_columns[0]["color"])

            self.assertEqual(table.columnCount(), 9)
            self.assertEqual(table.item(1, 8).text(), "1")
            self.assertEqual(table.item(data_row, 8).text(), "55")
            self.assertNotEqual(table.item(data_row, 8).background().color(), split_color)
        finally:
            table.deleteLater()

    def test_simple_nibble_split_does_not_keep_duplicate_byte_column(self):
        table = ByteStructuredTableWidget()
        try:
            table.set_row_size(2)
            table.set_bytes(np.array([0xCA, 0x55], dtype=np.uint8))
            table._split_columns({0}, "nibble")

            data_row = table.HEADER_ROW_COUNT
            split_color = table._color_from_name(table.split_columns[0]["color"])

            self.assertEqual(table.columnCount(), 3)
            self.assertEqual(table.item(2, 0).text(), "0-3")
            self.assertEqual(table.item(2, 1).text(), "4-7")
            self.assertEqual(table.item(1, 2).text(), "1")
            self.assertEqual(table.item(data_row, 2).text(), "55")
            self.assertNotEqual(table.item(data_row, 2).background().color(), split_color)
        finally:
            table.deleteLater()

    def test_find_split_for_custom_segment_returns_segment_payload(self):
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
            table.update_display()

            self.assertEqual(table._find_split_for_column(0), ("segment", 0, 4, 7))
            self.assertEqual(table._find_split_for_column(4), ("segment", 0, 0, 3))
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

    def test_combine_selected_four_binary_definition_bits_defaults_to_hex(self):
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
                total_bits=2,
            ),
        ]
        self.window.refresh_column_definitions_list()

        self.window.columns_list.item(0).setSelected(True)
        self.window.columns_list.item(1).setSelected(True)
        self.window.combine_selected_column_definitions()

        self.assertEqual(len(self.window.byte_table.column_definitions), 1)
        combined = self.window.byte_table.column_definitions[0]
        self.assertEqual(combined.start_bit, 0)
        self.assertEqual(combined.total_bits, 4)
        self.assertEqual(combined.display_format, "hex_be")
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

    def test_binary_split_multiple_bytes_get_unique_labels(self):
        self.window.byte_table.set_row_size(2)
        self.window.byte_table.set_bytes(np.array([0xCA, 0x55], dtype=np.uint8))
        self.window.byte_table.update_display()

        self.window.byte_table._split_columns({0, 1}, "binary")

        self.assertEqual(self.window.byte_table.item(0, 0).text(), "BIN 0")
        self.assertEqual(self.window.byte_table.item(0, 8).text(), "BIN 1")
        self.assertEqual(self.window.columns_list.count(), 2)
        self.assertEqual(self.window.columns_list.item(0).text(), "BIN 0")
        self.assertEqual(self.window.columns_list.item(0).text(1), "Byte 0")
        self.assertEqual(self.window.columns_list.item(0).text(2), "Binary")
        self.assertEqual(self.window.columns_list.item(1).text(), "BIN 1")
        self.assertEqual(self.window.columns_list.item(1).text(1), "Byte 1")
        self.assertEqual(self.window.columns_list.item(1).text(2), "Binary")

    def test_edit_split_from_list_cancel_keeps_split_unchanged(self):
        self.window.byte_table.set_bytes(np.array([0xCA], dtype=np.uint8))
        self.window.byte_table._split_columns({0}, "binary")
        self.window.refresh_column_definitions_list()

        item = self.window.columns_list.item(0)
        with mock.patch(
            "viewer.window_columns.AddColumnDialog.exec",
            autospec=True,
            return_value=QDialog.DialogCode.Rejected,
        ):
            self.window.edit_column_definition(item)

        self.assertEqual(len(self.window.byte_table.column_definitions), 0)
        self.assertIn(0, self.window.byte_table.split_columns)
        self.assertEqual(self.window.columns_list.count(), 1)
        self.assertEqual(self.window.columns_list.item(0).text(), "BIN 0")
        self.assertEqual(self.window.columns_list.item(0).text(1), "Byte 0")
        self.assertEqual(self.window.columns_list.item(0).text(2), "Binary")

    def test_edit_split_from_list_accept_creates_regular_definition(self):
        self.window.byte_table.set_bytes(np.array([0xCA], dtype=np.uint8))
        self.window.byte_table._split_columns({0}, "binary")
        self.window.refresh_column_definitions_list()

        item = self.window.columns_list.item(0)
        with mock.patch(
            "viewer.window_columns.AddColumnDialog.exec",
            autospec=True,
            return_value=QDialog.DialogCode.Accepted,
        ):
            self.window.edit_column_definition(item)

        self.assertEqual(len(self.window.byte_table.column_definitions), 1)
        created = self.window.byte_table.column_definitions[0]
        self.assertEqual(created.label, "BIN 0")
        self.assertEqual(created.display_format, "binary")
        self.assertEqual(created.color_name, "Sky")
        self.assertEqual(created.unit, "bit")
        self.assertEqual(created.start_bit, 0)
        self.assertEqual(created.total_bits, 8)

    def test_double_clicking_split_header_uses_regular_definition_editor(self):
        self.window.byte_table.set_bytes(np.array([0xCA], dtype=np.uint8))
        self.window.byte_table._split_columns({0}, "binary")
        self.window.byte_table.update_display()

        with mock.patch.object(self.window, "edit_split_as_definition") as edit_split:
            self.window.byte_table._on_cell_double_clicked(0, 0)

        edit_split.assert_called_once_with(0)

    def test_left_panel_uses_shared_wider_default_width(self):
        self.assertEqual(self.window.left_panel_scroll.minimumWidth(), self.window.LEFT_PANEL_WIDTH)
        self.assertEqual(self.window.left_panel_scroll.maximumWidth(), self.window.LEFT_PANEL_WIDTH)

    def test_column_definition_position_column_uses_compact_ranges(self):
        self.window.byte_table.column_definitions = [
            ColumnDefinition(
                start_byte=0,
                end_byte=0,
                label="Sync",
                display_format="hex_be",
                color_name="Sky",
                unit="bit",
                start_bit=0,
                total_bits=56,
            ),
            ColumnDefinition(
                start_byte=0,
                end_byte=0,
                label="Counter",
                display_format="dec_be",
                color_name="Gold",
                unit="bit",
                start_bit=56,
                total_bits=12,
            ),
        ]
        self.window.refresh_column_definitions_list()

        self.assertEqual(self.window.columns_list.item(0).text(), "Sync")
        self.assertEqual(self.window.columns_list.item(0).text(1), "0-55")
        self.assertEqual(self.window.columns_list.item(1).text(), "Counter")
        self.assertEqual(self.window.columns_list.item(1).text(1), "56-67")

    def test_column_definition_format_dropdown_updates_definition(self):
        self.window.byte_table.column_definitions = [
            ColumnDefinition(
                start_byte=0,
                end_byte=0,
                label="Counter",
                display_format="dec_be",
                color_name="Gold",
                unit="bit",
                start_bit=8,
                total_bits=8,
            ),
        ]
        self.window.refresh_column_definitions_list()

        item = self.window.columns_list.item(0)
        combo = self.window.columns_list.itemWidget(item, 2)
        self.assertIsNotNone(combo)
        self.assertEqual(combo.currentData(), "dec_be")

        hex_index = combo.findData("hex_be")
        self.assertGreaterEqual(hex_index, 0)
        combo.setCurrentIndex(hex_index)

        self.assertEqual(self.window.byte_table.column_definitions[0].display_format, "hex_be")

        refreshed_item = self.window.columns_list.item(0)
        refreshed_combo = self.window.columns_list.itemWidget(refreshed_item, 2)
        self.assertIsNotNone(refreshed_combo)
        self.assertEqual(refreshed_combo.currentData(), "hex_be")

    def test_split_row_format_dropdown_creates_definition(self):
        self.window.byte_table.set_bytes(np.array([0xCA], dtype=np.uint8))
        self.window.byte_table._split_columns({0}, "binary")
        self.window.refresh_column_definitions_list()

        item = self.window.columns_list.item(0)
        combo = self.window.columns_list.itemWidget(item, 2)
        self.assertIsNotNone(combo)
        self.assertEqual(combo.currentData(), "binary")

        hex_index = combo.findData("hex_be")
        self.assertGreaterEqual(hex_index, 0)
        combo.setCurrentIndex(hex_index)

        self.assertEqual(len(self.window.byte_table.column_definitions), 1)
        created = self.window.byte_table.column_definitions[0]
        self.assertEqual(created.label, "BIN 0")
        self.assertEqual(created.display_format, "hex_be")
        self.assertEqual(created.start_bit, 0)
        self.assertEqual(created.total_bits, 8)

        refreshed_item = self.window.columns_list.item(0)
        refreshed_combo = self.window.columns_list.itemWidget(refreshed_item, 2)
        self.assertIsNotNone(refreshed_combo)
        self.assertEqual(refreshed_combo.currentData(), "hex_be")

    def test_column_list_shows_only_remaining_visible_split_bits(self):
        self.window.byte_table.set_bytes(np.array([0xCA], dtype=np.uint8))
        self.window.byte_table._split_columns({0}, "binary")
        self.window.byte_table.column_definitions = [
            ColumnDefinition(
                start_byte=0,
                end_byte=0,
                label="test",
                display_format="hex_be",
                color_name="Coral",
                unit="bit",
                start_bit=4,
                total_bits=4,
            ),
        ]
        self.window.refresh_column_definitions_list()

        self.assertEqual(self.window.columns_list.count(), 2)
        self.assertEqual(self.window.columns_list.item(0).text(), "test")
        self.assertEqual(self.window.columns_list.item(0).text(1), "4-7")
        self.assertEqual(self.window.columns_list.item(1).text(), "BIN 0")
        self.assertEqual(self.window.columns_list.item(1).text(1), "Byte 0 bits 4-7")
        self.assertEqual(self.window.columns_list.item(1).text(2), "Binary")

        source, error = self.window._resolve_combine_source(self.window.columns_list.item(1))
        self.assertIsNone(error)
        self.assertEqual(source["start_bit"], 0)
        self.assertEqual(source["end_bit"], 3)
        self.assertEqual(source["total_bits"], 4)

    def test_combine_selected_visible_eight_binary_bits_defaults_to_hex(self):
        self.window.byte_table.set_bytes(np.array([0xCA], dtype=np.uint8))
        self.window.byte_table._split_columns({0}, "binary")
        self.window.byte_table.update_display()

        selected_cols = set(range(8))
        self.assertTrue(self.window.can_combine_selected_table_columns(selected_cols))

        self.window.combine_selected_table_columns(selected_cols)

        self.assertEqual(len(self.window.byte_table.column_definitions), 1)
        combined = self.window.byte_table.column_definitions[0]
        self.assertEqual(combined.start_bit, 0)
        self.assertEqual(combined.total_bits, 8)
        self.assertEqual(combined.display_format, "hex_be")
        self.assertEqual(combined.label, "Combo 0")
        self.assertEqual(self.window.byte_table.item(1, 0).text(), "0")
        self.assertEqual(self.window.byte_table.columnSpan(1, 0), 8)
        self.assertEqual(self.window.byte_table.item(2, 0).text(), "0-7")
        self.assertEqual(self.window.byte_table.columnSpan(2, 0), 8)

    def test_combine_selected_multi_byte_headers_use_compact_zero_based_ranges(self):
        self.window.byte_table.set_row_size(4)
        self.window.byte_table.set_bytes(np.array([0x12, 0x34, 0x56, 0x78], dtype=np.uint8))
        self.window.byte_table.update_display()

        selected_cols = {0, 1, 2}
        self.assertTrue(self.window.can_combine_selected_table_columns(selected_cols))

        self.window.combine_selected_table_columns(selected_cols)

        table = self.window.byte_table
        self.assertEqual(table.item(0, 0).text(), "Combo 0")
        self.assertEqual(table.columnSpan(0, 0), 3)
        self.assertEqual(table.item(1, 0).text(), "0-2")
        self.assertEqual(table.columnSpan(1, 0), 3)
        self.assertEqual(table.item(2, 0).text(), "0-23")
        self.assertEqual(table.columnSpan(2, 0), 3)
        header = table.horizontalHeader()
        self.assertTrue(header.isHidden())
        self.assertLess(table.columnWidth(0), 42)
        hit_x = table.columnViewportPosition(0) + table.columnWidth(0)
        hit_y = table.rowViewportPosition(1) + (table.rowHeight(1) // 2)
        self.assertEqual(table._header_resize_hit_test(QPoint(hit_x, hit_y)), 0)
        label_hit_y = table.rowViewportPosition(0) + (table.rowHeight(0) // 2)
        self.assertIsNone(table._header_resize_hit_test(QPoint(hit_x, label_hit_y)))

    def test_live_viewer_uses_underlying_bits_for_combined_nibble_definition(self):
        self.window.byte_table.set_bytes(np.array([0xF0, 0xF1, 0xFA], dtype=np.uint8))
        self.window.byte_table.frames = [(0, 8), (8, 8), (16, 8)]
        self.window.byte_table.update_display()
        self.window.byte_table._split_columns({0}, "binary")

        self.window.combine_selected_table_columns({4, 5, 6, 7})
        self.assertEqual(self.window.byte_table.item(1, 4).text(), "0")
        self.assertEqual(self.window.byte_table.columnSpan(1, 4), 4)
        self.assertEqual(self.window.byte_table.item(2, 0).text(), "0")
        self.assertEqual(self.window.byte_table.item(2, 1).text(), "1")
        self.assertEqual(self.window.byte_table.item(2, 2).text(), "2")
        self.assertEqual(self.window.byte_table.item(2, 3).text(), "3")
        self.assertEqual(self.window.byte_table.item(2, 4).text(), "4-7")
        self.assertEqual(self.window.byte_table.columnSpan(2, 4), 4)

        self.window.byte_table.selected_columns = {4}
        self.window.byte_table._update_live_bit_viewer()

        frames_bits = self.window.live_bit_viewer_canvas.frames_bits
        self.assertEqual([bits.tolist() for bits in frames_bits], [[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]])

    def test_live_viewer_uses_underlying_bits_for_combined_middle_four_bits(self):
        self.window.byte_table.set_bytes(np.array([0x3C, 0x24], dtype=np.uint8))
        self.window.byte_table.frames = [(0, 8), (8, 8)]
        self.window.byte_table.update_display()
        self.window.byte_table._split_columns({0}, "binary")

        self.window.combine_selected_table_columns({2, 3, 4, 5})
        self.assertEqual(self.window.byte_table.item(2, 2).text(), "2-5")
        self.assertEqual(self.window.byte_table.columnSpan(2, 2), 4)

        self.window.byte_table.selected_columns = {2}
        self.window.byte_table._update_live_bit_viewer()

        frames_bits = self.window.live_bit_viewer_canvas.frames_bits
        self.assertEqual([bits.tolist() for bits in frames_bits], [[1, 1, 1, 1], [1, 0, 0, 1]])

    def test_combine_selected_visible_table_columns_accepts_multi_segment_mixed_selection(self):
        self.window.byte_table.set_row_size(3)
        self.window.byte_table.set_bytes(np.array([0xCA, 0x55, 0xF0], dtype=np.uint8))
        self.window.byte_table._split_columns({1, 2}, "nibble")
        self.window.byte_table._split_columns({0}, "binary")
        self.window.byte_table.update_display()

        selected_cols = {4, 5, 6, 7, 8, 9, 10, 11}
        self.assertTrue(self.window.can_combine_selected_table_columns(selected_cols))

        self.window.combine_selected_table_columns(selected_cols)

        self.assertEqual(len(self.window.byte_table.column_definitions), 1)
        combined = self.window.byte_table.column_definitions[0]
        self.assertEqual(combined.start_bit, 4)
        self.assertEqual(combined.total_bits, 20)
        self.assertEqual(combined.display_format, "hex_be")
        self.assertEqual(combined.color_name, "Sky")
        self.assertEqual(combined.label, "Combo 0")


if __name__ == "__main__":
    unittest.main()
