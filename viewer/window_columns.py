"""Column-definition helpers for the main bit viewer window."""

from __future__ import annotations

import re

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush
from PyQt6.QtWidgets import QComboBox, QHeaderView, QMessageBox, QTreeWidget, QTreeWidgetItem

from .column import AddColumnDialog, ColumnDefinition


LIST_KIND_ROLE = Qt.ItemDataRole.UserRole
LIST_PAYLOAD_ROLE = Qt.ItemDataRole.UserRole + 1
FORMAT_COLUMN_INDEX = 2


class ColumnDefinitionsItem(QTreeWidgetItem):
    """Tree item with QListWidgetItem-like convenience defaults for column 0."""

    def text(self, column=0):
        return super().text(column)

    def data(self, *args):
        if len(args) == 1:
            column, role = 0, args[0]
        else:
            column, role = args
        return super().data(column, role)

    def setData(self, *args):
        if len(args) == 2:
            column, role, value = 0, args[0], args[1]
        else:
            column, role, value = args
        super().setData(column, role, value)

    def setBackground(self, brush, column=0):
        super().setBackground(column, brush)

    def background(self, column=0):
        return super().background(column)


class ColumnDefinitionsTreeWidget(QTreeWidget):
    """Structured column-definition list with light QListWidget compatibility."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(3)
        self.setHeaderLabels(["Name", "Position", "Format"])
        self.setRootIsDecorated(False)
        self.setItemsExpandable(False)
        self.setUniformRowHeights(True)
        self.setIndentation(0)
        self.setAllColumnsShowFocus(True)

        header = self.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

    def addItem(self, item):
        self.addTopLevelItem(item)

    def count(self):
        return self.topLevelItemCount()

    def item(self, index):
        return self.topLevelItem(index)

    def takeItem(self, index):
        return self.takeTopLevelItem(index)


class BitViewerWindowColumnsMixin:
    DISPLAY_FORMAT_LABELS = {
        "hex_be": "Hex (MSBF)",
        "hex_le": "Hex (LSBF)",
        "hex": "Hex (MSBF)",
        "binary": "Binary",
        "dec_be": "Decimal (MSBF)",
        "dec_le": "Decimal (LSBF)",
        "tc_be": "Twos complement (MSBF)",
        "tc_le": "Twos complement (LSBF)",
        "ascii_be": "ASCII (MSBF)",
        "ascii_le": "ASCII (LSBF)",
        "ascii": "ASCII (MSBF)",
    }
    DISPLAY_FORMAT_OPTIONS = [
        ("hex_be", "Hex (MSBF)"),
        ("hex_le", "Hex (LSBF)"),
        ("binary", "Binary"),
        ("dec_be", "Decimal (MSBF)"),
        ("dec_le", "Decimal (LSBF)"),
        ("tc_be", "Twos complement (MSBF)"),
        ("tc_le", "Twos complement (LSBF)"),
        ("ascii_be", "ASCII (MSBF)"),
        ("ascii_le", "ASCII (LSBF)"),
    ]

    def _next_combo_label(self):
        next_index = 0
        for col_def in self.byte_table.column_definitions:
            match = re.match(r"^COMBO\s+(\d+)$", str(col_def.label or "").strip(), re.IGNORECASE)
            if match:
                next_index = max(next_index, int(match.group(1)) + 1)
        return f"Combo {next_index}"

    def _absolute_bit_bounds(self, byte_idx, bit_start, bit_end):
        start_bit_msb = 7 - bit_start
        end_bit_msb = 7 - bit_end
        abs_start = byte_idx * 8 + min(start_bit_msb, end_bit_msb)
        abs_end = byte_idx * 8 + max(start_bit_msb, end_bit_msb)
        return abs_start, abs_end

    def _column_definition_bit_bounds(self, col_def):
        if col_def.unit == "bit":
            start_bit = col_def.start_bit
            total_bits = col_def.total_bits
        else:
            start_bit = col_def.start_byte * 8
            total_bits = (col_def.end_byte - col_def.start_byte + 1) * 8
        return start_bit, start_bit + total_bits - 1, total_bits

    def _range_fully_defined(self, start_bit, end_bit):
        for col_def in self.byte_table.column_definitions:
            def_start, def_end, _ = self._column_definition_bit_bounds(col_def)
            if def_start <= start_bit and end_bit <= def_end:
                return True
        return False

    def _byte_display_ranges(self, byte_idx):
        """Rebuild the current visible bit ranges for one byte."""
        byte_start_bit = byte_idx * 8
        byte_end_bit = byte_start_bit + 7
        has_split = self.byte_table._byte_has_split(byte_idx)
        ranges_in_byte = []

        for col_def in self.byte_table.column_definitions:
            if col_def.unit == "bit":
                def_start_bit = col_def.start_bit
                def_end_bit = def_start_bit + col_def.total_bits - 1
                if def_start_bit <= byte_end_bit and def_end_bit >= byte_start_bit:
                    overlap_start_bit = max(def_start_bit, byte_start_bit)
                    overlap_end_bit = min(def_end_bit, byte_end_bit)
                    local_start_bit = overlap_start_bit - byte_start_bit
                    local_end_bit = overlap_end_bit - byte_start_bit
                    if has_split:
                        start_in_byte = 7 - local_end_bit
                        end_in_byte = 7 - local_start_bit
                    else:
                        start_in_byte = local_start_bit
                        end_in_byte = local_end_bit
                    ranges_in_byte.append((start_in_byte, end_in_byte, col_def, False))
            elif col_def.start_byte <= byte_idx <= col_def.end_byte:
                ranges_in_byte.append((0, 7, col_def, False))
                break

        if ranges_in_byte and not (
            len(ranges_in_byte) == 1 and ranges_in_byte[0][0] == 0 and ranges_in_byte[0][1] == 7
        ):
            ranges_in_byte.sort(key=lambda entry: entry[0])
            all_ranges = []
            current_bit = 0

            for start_in_byte, end_in_byte, col_def, is_undef in ranges_in_byte:
                if current_bit < start_in_byte:
                    all_ranges.append((current_bit, start_in_byte - 1, None, not has_split))
                all_ranges.append((start_in_byte, end_in_byte, col_def, is_undef))
                current_bit = end_in_byte + 1

            if current_bit <= 7:
                all_ranges.append((current_bit, 7, None, not has_split))
            return all_ranges

        if ranges_in_byte:
            return ranges_in_byte
        return [(0, 7, None, False)]

    def _visible_byte_segment_ranges(self, byte_idx, segment_start, segment_end):
        """Return the currently visible uncovered ranges for a split segment."""
        intersections = self.byte_table._segment_intersections_for_display(
            self._byte_display_ranges(byte_idx),
            segment_start,
            segment_end,
        )
        return [
            (range_start, range_end)
            for range_start, range_end, col_def, is_undef in intersections
            if col_def is None and not is_undef
        ]

    def _visible_ranges_for_split_payload(self, payload):
        """Return visible uncovered ranges for a split list payload."""
        if isinstance(payload, tuple):
            if len(payload) == 4 and payload[0] == "segment":
                _, split_key, segment_start, segment_end = payload
                if not isinstance(split_key, int):
                    return []
                return self._visible_byte_segment_ranges(split_key, segment_start, segment_end)

            byte_idx, _ = payload
            split_info = self.byte_table.split_columns.get(payload)
            if split_info is None or split_info.get("type") != "nibble_binary":
                return []
            bit_start = split_info.get("bit_start", 0)
            return self._visible_byte_segment_ranges(byte_idx, bit_start, bit_start + 3)

        split_info = self.byte_table.split_columns.get(payload)
        if split_info is None:
            return []
        return self._visible_byte_segment_ranges(payload, 0, 7)

    def _format_bit_range_list(self, ranges):
        """Format one or more within-byte ranges like '0-3, 6-7'."""
        normalized = sorted((min(start, end), max(start, end)) for start, end in ranges)
        parts = []
        for start, end in normalized:
            if start == end:
                parts.append(str(start))
            else:
                parts.append(f"{start}-{end}")
        return ", ".join(parts)

    def _visible_range_suffix(self, visible_ranges, original_start, original_end):
        """Return a display suffix for partial split visibility."""
        normalized = sorted((min(start, end), max(start, end)) for start, end in visible_ranges)
        original = (min(original_start, original_end), max(original_start, original_end))
        if normalized == [original]:
            return ""
        return f" bits {self._format_bit_range_list(normalized)}"

    def _split_payload_bit_bounds(self, payload):
        visible_ranges = self._visible_ranges_for_split_payload(payload)
        if len(visible_ranges) != 1:
            return None
        if isinstance(payload, tuple):
            if len(payload) == 4 and payload[0] == "segment":
                _, split_key, _, _ = payload
                if not isinstance(split_key, int):
                    return None
                return self._absolute_bit_bounds(split_key, visible_ranges[0][0], visible_ranges[0][1])
            byte_idx, _ = payload
            return self._absolute_bit_bounds(byte_idx, visible_ranges[0][0], visible_ranges[0][1])
        return self._absolute_bit_bounds(payload, visible_ranges[0][0], visible_ranges[0][1])

    def _combine_display_kind_for_format(self, display_format):
        if display_format == "binary":
            return "binary"
        if display_format in {"hex", "hex_be", "hex_le"}:
            return "hex"
        return None

    def _display_format_dialog_label(self, display_format):
        return self.DISPLAY_FORMAT_LABELS.get(display_format, "Hex (MSBF)")

    def _normalize_display_format(self, display_format):
        if display_format == "hex":
            return "hex_be"
        if display_format == "ascii":
            return "ascii_be"
        return display_format

    def _definition_supports_hex_format(self, col_def):
        return col_def.unit == "byte" or (
            (col_def.start_bit % 4) == 0 and (col_def.total_bits % 4) == 0
        )

    def _definition_supports_ascii_format(self, col_def):
        return col_def.unit == "byte" or (
            (col_def.start_bit % 8) == 0 and (col_def.total_bits % 8) == 0
        )

    def _format_options_for_definition(self, col_def):
        return self._format_options_for_span(col_def.start_bit, col_def.total_bits, col_def.unit)

    def _format_options_for_span(self, start_bit, total_bits, unit="bit"):
        byte_aligned = unit == "byte" or ((start_bit % 8) == 0 and (total_bits % 8) == 0)
        nibble_aligned = unit == "byte" or ((start_bit % 4) == 0 and (total_bits % 4) == 0)
        options = []
        for format_key, format_label in self.DISPLAY_FORMAT_OPTIONS:
            if format_key.startswith("hex") and not nibble_aligned:
                continue
            if format_key.startswith("ascii") and not byte_aligned:
                continue
            options.append((format_key, format_label))
        return options

    def _column_definition_name(self, col_def):
        return col_def.label if col_def.label else "(unnamed)"

    def _compact_range_label(self, start_value, end_value):
        if start_value == end_value:
            return str(start_value)
        return f"{start_value}-{end_value}"

    def _column_definition_position(self, col_def):
        if col_def.unit == "byte":
            return self._compact_range_label(col_def.start_byte, col_def.end_byte)
        return self._compact_range_label(
            col_def.start_bit,
            col_def.start_bit + col_def.total_bits - 1,
        )

    def _split_definition_name(self, split_info):
        return split_info.get("label") or split_info.get("type", "Split").title()

    def _split_definition_position(self, split_key, split_info, visible_ranges=None):
        if isinstance(split_key, tuple):
            byte_idx, nibble_type = split_key
            if visible_ranges:
                suffix = self._visible_range_suffix(
                    visible_ranges,
                    split_info.get("bit_start", 0),
                    split_info.get("bit_start", 0) + 3,
                )
                if suffix:
                    return f"Byte {byte_idx}{suffix}"
            nibble_label = "high nibble" if nibble_type == "high" else "low nibble"
            return f"Byte {byte_idx} {nibble_label}"

        split_type = split_info.get("type", "split")
        if split_type == "custom_bits":
            labeled_segment = next(
                (segment for segment in split_info.get("segments", []) if segment.get("label")),
                None,
            )
            if labeled_segment is not None:
                return f"Byte {split_key} bits {labeled_segment['start']}-{labeled_segment['end']}"

        suffix = self._visible_range_suffix(visible_ranges or [(0, 7)], 0, 7)
        if suffix:
            return f"Byte {split_key}{suffix}"
        return f"Byte {split_key}"

    def _custom_split_segment_name(self, segment):
        return segment.get("label") or segment.get("format", "Split").title()

    def _custom_split_segment_position(self, split_key, segment, visible_ranges=None):
        suffix = self._visible_range_suffix(
            visible_ranges or [(segment["start"], segment["end"])],
            segment["start"],
            segment["end"],
        )
        if suffix:
            return f"Byte {split_key}{suffix}"
        return f"Byte {split_key} bits {segment['start']}-{segment['end']}"

    def _split_display_format_label(self, split_info, segment=None):
        if segment is not None:
            return self._display_format_dialog_label("binary" if segment.get("format") == "binary" else "hex_be")

        split_type = split_info.get("type")
        if split_type in {"binary", "nibble_binary"}:
            return self._display_format_dialog_label("binary")
        if split_type == "nibble":
            return self._display_format_dialog_label("hex_be")
        return ""

    def _add_columns_list_item(self, name, position, format_text, color_name, kind, payload):
        item = ColumnDefinitionsItem([name, position, format_text])
        item.setData(LIST_KIND_ROLE, kind)
        item.setData(LIST_PAYLOAD_ROLE, payload)
        if color_name != "None":
            color = self.byte_table._color_from_name(color_name)
            if color:
                for column in range(self.columns_list.columnCount()):
                    item.setBackground(QBrush(color), column)
        self.columns_list.addItem(item)
        return item

    def _build_definition_format_combo(self, definition_index, col_def):
        combo = QComboBox(self.columns_list)
        current_format = self._normalize_display_format(col_def.display_format)
        for format_key, format_label in self._format_options_for_definition(col_def):
            combo.addItem(format_label, format_key)

        current_index = combo.findData(current_format)
        if current_index >= 0:
            combo.setCurrentIndex(current_index)

        combo.currentIndexChanged.connect(
            lambda _idx, idx=definition_index, widget=combo: self._on_columns_list_format_changed(idx, widget)
        )
        return combo

    def _create_definition_from_split_payload(self, payload, display_format):
        spec, error = self._split_payload_definition_spec(payload)
        if error is not None:
            QMessageBox.warning(self, "Edit Split", error)
            return False

        self.byte_table.add_column_definition(
            ColumnDefinition(
                start_byte=0,
                end_byte=0,
                label=spec["label"],
                display_format=self._normalize_display_format(display_format),
                color_name=spec["color_name"],
                unit="bit",
                start_bit=spec["start_bit"],
                total_bits=spec["total_bits"],
            )
        )
        self.refresh_column_definitions_list()
        self._refresh_byte_table_view()
        return True

    def _build_split_format_combo(self, payload):
        spec, error = self._split_payload_definition_spec(payload)
        if error is not None:
            return None

        combo = QComboBox(self.columns_list)
        current_format = self._normalize_display_format(spec["display_format"])
        for format_key, format_label in self._format_options_for_span(spec["start_bit"], spec["total_bits"], "bit"):
            combo.addItem(format_label, format_key)

        current_index = combo.findData(current_format)
        if current_index >= 0:
            combo.setCurrentIndex(current_index)

        combo.currentIndexChanged.connect(
            lambda _idx, split_payload=payload, widget=combo: self._on_split_format_changed(split_payload, widget)
        )
        return combo

    def _on_columns_list_format_changed(self, definition_index, combo):
        if definition_index < 0 or definition_index >= len(self.byte_table.column_definitions):
            return

        new_format = self._normalize_display_format(combo.currentData())
        old_def = self.byte_table.column_definitions[definition_index]
        if self._normalize_display_format(old_def.display_format) == new_format:
            return

        old_def.display_format = new_format
        self.refresh_column_definitions_list()
        self._refresh_byte_table_view()

    def _on_split_format_changed(self, payload, combo):
        new_format = self._normalize_display_format(combo.currentData())
        spec, error = self._split_payload_definition_spec(payload)
        if error is not None:
            return

        current_format = self._normalize_display_format(spec["display_format"])
        if current_format == new_format:
            return

        self._create_definition_from_split_payload(payload, new_format)

    def _split_payload_definition_spec(self, payload):
        """Return dialog-prefill data for converting a split into a definition."""
        visible_ranges = self._visible_ranges_for_split_payload(payload)
        if not visible_ranges:
            return None, "Selected split is fully covered by column definitions."
        if len(visible_ranges) != 1:
            return None, "Selected split is fragmented; define it from the table selection instead."

        split_info = None
        label = ""
        color_name = "None"
        display_format = "hex_be"
        range_start, range_end = visible_ranges[0]

        if isinstance(payload, tuple) and len(payload) == 4 and payload[0] == "segment":
            _, split_key, segment_start, segment_end = payload
            split_info = self.byte_table.split_columns.get(split_key)
            if split_info is None or not isinstance(split_key, int):
                return None, "Selected split segment is no longer available."
            segment = next(
                (
                    candidate for candidate in split_info.get("segments", [])
                    if candidate.get("start") == segment_start and candidate.get("end") == segment_end
                ),
                None,
            )
            if segment is None:
                return None, "Selected split segment is no longer available."
            label = segment.get("label", "")
            color_name = segment.get("color", split_info.get("color", "None"))
            display_format = "binary" if segment.get("format") == "binary" else "hex_be"
            start_bit, end_bit = self._absolute_bit_bounds(split_key, range_start, range_end)
        elif isinstance(payload, tuple):
            byte_idx, _ = payload
            split_info = self.byte_table.split_columns.get(payload)
            if split_info is None:
                return None, "Selected split is no longer available."
            label = split_info.get("label", "")
            color_name = split_info.get("color", "None")
            display_format = "binary"
            start_bit, end_bit = self._absolute_bit_bounds(byte_idx, range_start, range_end)
        else:
            split_info = self.byte_table.split_columns.get(payload)
            if split_info is None:
                return None, "Selected split is no longer available."
            label = split_info.get("label", "")
            color_name = split_info.get("color", "None")
            split_type = split_info.get("type")
            if split_type == "binary":
                display_format = "binary"
            elif split_type == "nibble":
                display_format = "hex_be"
            else:
                return None, "Selected split can't be edited as a single regular column definition."
            start_bit, end_bit = self._absolute_bit_bounds(payload, range_start, range_end)

        return {
            "label": label,
            "color_name": color_name,
            "display_format": display_format,
            "start_bit": start_bit,
            "total_bits": end_bit - start_bit + 1,
        }, None

    def edit_split_as_definition(self, payload):
        """Open the standard definition dialog prefilled from a split."""
        spec, error = self._split_payload_definition_spec(payload)
        if error is not None:
            QMessageBox.warning(self, "Edit Split", error)
            return

        dialog = AddColumnDialog(self)
        dialog.setWindowTitle("Edit Column Definition")
        dialog.bit_radio.setChecked(True)
        dialog.bit_pos_radio.setChecked(True)
        dialog.abs_bit_position_spin.setValue(spec["start_bit"])
        dialog.total_bits_spin.setValue(spec["total_bits"])
        dialog.label_input.setText(spec["label"])

        format_label = self._display_format_dialog_label(spec["display_format"])
        format_index = dialog.format_combo.findText(format_label)
        if format_index >= 0:
            dialog.format_combo.setCurrentIndex(format_index)

        color_index = dialog.color_combo.findText(spec["color_name"])
        if color_index >= 0:
            dialog.color_combo.setCurrentIndex(color_index)

        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        self.byte_table.add_column_definition(dialog.get_column_definition())
        self.refresh_column_definitions_list()
        self._refresh_byte_table_view()

    def _resolve_split_combine_source(self, payload):
        split_info = None
        label = ""
        color_name = "None"

        if isinstance(payload, tuple) and len(payload) == 4 and payload[0] == "segment":
            _, split_key, segment_start, segment_end = payload
            split_info = self.byte_table.split_columns.get(split_key)
            if split_info is None or not isinstance(split_key, int):
                return None, "Selected split segment is no longer available."

            segment = next(
                (
                    candidate for candidate in split_info.get("segments", [])
                    if candidate.get("start") == segment_start and candidate.get("end") == segment_end
                ),
                None,
            )
            if segment is None:
                return None, "Selected split segment is no longer available."

            visible_ranges = self._visible_ranges_for_split_payload(payload)
            if not visible_ranges:
                return None, "Selected split segment is fully covered by column definitions."
            if len(visible_ranges) != 1:
                return None, "Selected split segment is fragmented; combine it from the table selection instead."

            display_kind = "binary" if segment.get("format") == "binary" else "hex"
            label = segment.get("label", "")
            color_name = segment.get("color", split_info.get("color", "None"))
            start_bit, end_bit = self._absolute_bit_bounds(split_key, visible_ranges[0][0], visible_ranges[0][1])
            return {
                "source_kind": "split",
                "display_kind": display_kind,
                "start_bit": start_bit,
                "end_bit": end_bit,
                "total_bits": end_bit - start_bit + 1,
                "label": label,
                "color_name": color_name,
                "definition_index": None,
            }, None

        split_info = self.byte_table.split_columns.get(payload)
        if split_info is None:
            return None, "Selected split is no longer available."

        split_type = split_info.get("type")
        label = split_info.get("label", "")
        color_name = split_info.get("color", "None")
        visible_ranges = self._visible_ranges_for_split_payload(payload)
        if not visible_ranges:
            return None, "Selected split is fully covered by column definitions."
        if len(visible_ranges) != 1:
            return None, "Selected split is fragmented; combine it from the table selection instead."

        if isinstance(payload, tuple):
            byte_idx, _ = payload
            if split_type != "nibble_binary":
                return None, "Only binary and hex split columns can be combined."
            start_bit, end_bit = self._absolute_bit_bounds(byte_idx, visible_ranges[0][0], visible_ranges[0][1])
            display_kind = "binary"
        else:
            start_bit, end_bit = self._absolute_bit_bounds(payload, visible_ranges[0][0], visible_ranges[0][1])
            if split_type == "binary":
                display_kind = "binary"
            elif split_type == "nibble":
                display_kind = "hex"
            else:
                return None, "Only binary and hex split columns can be combined."

        return {
            "source_kind": "split",
            "display_kind": display_kind,
            "start_bit": start_bit,
            "end_bit": end_bit,
            "total_bits": end_bit - start_bit + 1,
            "label": label,
            "color_name": color_name,
            "definition_index": None,
        }, None

    def _resolve_combine_source(self, item):
        item_kind = item.data(LIST_KIND_ROLE)
        payload = item.data(LIST_PAYLOAD_ROLE)

        if item_kind == "definition":
            if payload < 0 or payload >= len(self.byte_table.column_definitions):
                return None, "Selected column definition is no longer available."

            col_def = self.byte_table.column_definitions[payload]
            display_kind = self._combine_display_kind_for_format(col_def.display_format)
            if display_kind is None:
                return None, "Combine currently supports binary and hex columns only."

            start_bit, end_bit, total_bits = self._column_definition_bit_bounds(col_def)
            return {
                "source_kind": "definition",
                "display_kind": display_kind,
                "start_bit": start_bit,
                "end_bit": end_bit,
                "total_bits": total_bits,
                "label": col_def.label,
                "color_name": col_def.color_name,
                "definition_index": payload,
            }, None

        if item_kind == "split":
            return self._resolve_split_combine_source(payload)

        return None, "Unsupported selection."

    def _combined_display_format(self, sources, total_bits):
        display_kinds = {source["display_kind"] for source in sources}
        if display_kinds == {"binary"}:
            if total_bits in {4, 8}:
                return "hex_be"
            return "binary"
        if total_bits % 4 != 0:
            return None
        return "hex_be"

    def _combined_label(self, sources):
        return self._next_combo_label()

    def _combine_source_from_table_column(self, table_col):
        if not self.byte_table._all_columns_info or table_col < 0 or table_col >= len(self.byte_table._all_columns_info):
            return None, "Selected columns are out of range."

        byte_idx, bit_start, bit_end, col_def, is_undef = self.byte_table._all_columns_info[table_col]
        if is_undef:
            return None, "Undefined columns cannot be combined."

        if col_def is not None:
            try:
                definition_index = self.byte_table.column_definitions.index(col_def)
            except ValueError:
                return None, "Selected column definition is no longer available."

            start_bit, end_bit, total_bits = self._column_definition_bit_bounds(col_def)
            display_kind = self._combine_display_kind_for_format(col_def.display_format)
            if display_kind is None:
                return None, "Combine currently supports binary and hex columns only."

            return {
                "source_key": ("definition", id(col_def)),
                "source_kind": "definition",
                "display_kind": display_kind,
                "start_bit": start_bit,
                "end_bit": end_bit,
                "total_bits": total_bits,
                "label": col_def.label,
                "color_name": col_def.color_name,
                "definition_index": definition_index,
            }, None

        num_bits = bit_end - bit_start + 1
        if num_bits == 1:
            display_kind = "binary"
        elif num_bits % 4 == 0:
            display_kind = "hex"
        else:
            return None, "Combine currently supports binary and hex columns only."

        color_name = "None"
        custom_segment = self.byte_table._custom_split_segment(byte_idx, bit_start, bit_end)
        if custom_segment is not None:
            color_name = custom_segment.get("color", "None")
        else:
            nibble_type = None
            if 4 <= bit_start <= bit_end <= 7:
                nibble_type = "high"
            elif 0 <= bit_start <= bit_end <= 3:
                nibble_type = "low"

            if nibble_type is not None:
                nibble_split_info = self.byte_table.split_columns.get((byte_idx, nibble_type))
                if nibble_split_info is not None:
                    color_name = nibble_split_info.get("color", "None")

            if color_name == "None":
                split_info = self.byte_table.split_columns.get(byte_idx)
                if split_info is not None:
                    color_name = split_info.get("color", "None")

        absolute_start_bit, absolute_end_bit = self._absolute_bit_bounds(byte_idx, bit_start, bit_end)
        return {
            "source_key": ("table_col", table_col),
            "source_kind": "table_col",
            "display_kind": display_kind,
            "start_bit": absolute_start_bit,
            "end_bit": absolute_end_bit,
            "total_bits": absolute_end_bit - absolute_start_bit + 1,
            "label": "",
            "color_name": color_name,
            "definition_index": None,
        }, None

    def _resolve_table_combine_sources(self, selected_cols):
        if not selected_cols:
            return None, "Select at least two columns to combine."

        expanded_cols = self.byte_table._expand_selected_definition_columns(sorted(selected_cols))
        sources_by_key = {}
        source_order = []

        for table_col in expanded_cols:
            source, error = self._combine_source_from_table_column(table_col)
            if error is not None:
                return None, error
            source_key = source["source_key"]
            if source_key not in sources_by_key:
                sources_by_key[source_key] = source
                source_order.append(source_key)

        resolved_sources = [sources_by_key[source_key] for source_key in source_order]
        if len(resolved_sources) < 2:
            return None, "Select at least two visible columns to combine."

        return resolved_sources, None

    def _validate_combine_sources(self, resolved_sources):
        if resolved_sources is None or len(resolved_sources) < 2:
            return None, None, "Select at least two columns to combine."

        sorted_sources = sorted(resolved_sources, key=lambda source: (source["start_bit"], source["end_bit"]))
        previous_source = sorted_sources[0]
        for current_source in sorted_sources[1:]:
            if previous_source["end_bit"] >= current_source["start_bit"]:
                return None, None, "Selected columns overlap and cannot be combined."
            if previous_source["end_bit"] + 1 != current_source["start_bit"]:
                return None, None, "Selected columns must be adjacent to combine."
            previous_source = current_source

        total_bits = sorted_sources[-1]["end_bit"] - sorted_sources[0]["start_bit"] + 1
        display_format = self._combined_display_format(sorted_sources, total_bits)
        if display_format is None:
            return None, None, (
                "Mixed binary and hex columns can only be combined when the total width is a multiple of 4 bits."
            )

        return sorted_sources, display_format, None

    def can_combine_selected_table_columns(self, selected_cols):
        sources, error = self._resolve_table_combine_sources(selected_cols)
        if error is not None or sources is None:
            return False

        _, _, validation_error = self._validate_combine_sources(sources)
        return validation_error is None

    def _combine_resolved_sources(self, resolved_sources, title):
        sorted_sources, display_format, validation_error = self._validate_combine_sources(resolved_sources)
        if validation_error is not None:
            QMessageBox.warning(self, title, validation_error)
            return

        first_source = sorted_sources[0]
        total_bits = sorted_sources[-1]["end_bit"] - first_source["start_bit"] + 1
        color_name = next(
            (
                source["color_name"]
                for source in sorted_sources
                if source.get("color_name") and source["color_name"] != "None"
            ),
            "None",
        )
        combined_definition = ColumnDefinition(
            start_byte=0,
            end_byte=0,
            label=self._combined_label(resolved_sources),
            display_format=display_format,
            color_name=color_name,
            unit="bit",
            start_bit=first_source["start_bit"],
            total_bits=total_bits,
        )

        definition_indices = sorted(
            {
                source["definition_index"]
                for source in resolved_sources
                if source.get("definition_index") is not None
            },
            reverse=True,
        )
        insert_at = min(definition_indices) if definition_indices else len(self.byte_table.column_definitions)

        for definition_index in definition_indices:
            self.byte_table.column_definitions.pop(definition_index)

        self.byte_table.column_definitions.insert(insert_at, combined_definition)
        self._refresh_byte_table_view()

    def refresh_column_definitions_list(self):
        self.columns_list.clear()

        for index, col_def in enumerate(self.byte_table.column_definitions):
            item = self._add_columns_list_item(
                self._column_definition_name(col_def),
                self._column_definition_position(col_def),
                "",
                col_def.color_name,
                "definition",
                index,
            )
            self.columns_list.setItemWidget(
                item,
                FORMAT_COLUMN_INDEX,
                self._build_definition_format_combo(index, col_def),
            )

        def split_sort_key(split_key):
            if isinstance(split_key, tuple):
                byte_idx, nibble_type = split_key
                nibble_order = 0 if nibble_type == "high" else 1
                return (byte_idx, 1, nibble_order, str(split_key))
            return (split_key, 0, 0, str(split_key))

        for split_key in sorted(self.byte_table.split_columns.keys(), key=split_sort_key):
            split_info = self.byte_table.split_columns[split_key]
            if split_info.get("type") == "custom_bits":
                for segment in split_info.get("segments", []):
                    payload = ("segment", split_key, segment.get("start"), segment.get("end"))
                    visible_ranges = self._visible_ranges_for_split_payload(
                        ("segment", split_key, segment.get("start", 0), segment.get("end", 0))
                    )
                    if not visible_ranges:
                        continue
                    item = self._add_columns_list_item(
                        self._custom_split_segment_name(segment),
                        self._custom_split_segment_position(split_key, segment, visible_ranges),
                        self._split_display_format_label(split_info, segment),
                        segment.get("color", split_info.get("color", "None")),
                        "split",
                        payload,
                    )
                    combo = self._build_split_format_combo(payload)
                    if combo is not None:
                        self.columns_list.setItemWidget(item, FORMAT_COLUMN_INDEX, combo)
            else:
                visible_ranges = self._visible_ranges_for_split_payload(split_key)
                if not visible_ranges:
                    continue
                item = self._add_columns_list_item(
                    self._split_definition_name(split_info),
                    self._split_definition_position(split_key, split_info, visible_ranges),
                    self._split_display_format_label(split_info),
                    split_info.get("color", "None"),
                    "split",
                    split_key,
                )
                combo = self._build_split_format_combo(split_key)
                if combo is not None:
                    self.columns_list.setItemWidget(item, FORMAT_COLUMN_INDEX, combo)

    def add_column_definition(self):
        dialog = AddColumnDialog(self)
        if dialog.exec() == dialog.DialogCode.Accepted:
            col_def = dialog.get_column_definition()
            self.byte_table.add_column_definition(col_def)
            self.refresh_column_definitions_list()

    def add_column_definition_prefilled(self, start_byte, end_byte):
        dialog = AddColumnDialog(self)
        dialog.setWindowTitle("Add Column Definition from Selection")
        dialog.byte_radio.setChecked(True)
        dialog.start_byte_spin.setValue(start_byte)
        dialog.end_byte_spin.setValue(end_byte)

        if dialog.exec() == dialog.DialogCode.Accepted:
            col_def = dialog.get_column_definition()
            self.byte_table.add_column_definition(col_def)
            self.refresh_column_definitions_list()
            self.byte_table.selected_columns.clear()
            self._refresh_byte_table_view()

    def add_column_definition_prefilled_bits(self, start_bit, total_bits):
        dialog = AddColumnDialog(self)
        dialog.setWindowTitle("Add Column Definition from Selection")
        dialog.bit_radio.setChecked(True)
        dialog.bit_pos_radio.setChecked(True)
        dialog.abs_bit_position_spin.setValue(start_bit)
        dialog.total_bits_spin.setValue(total_bits)

        if dialog.exec() == dialog.DialogCode.Accepted:
            col_def = dialog.get_column_definition()
            self.byte_table.add_column_definition(col_def)
            self.refresh_column_definitions_list()
            self.byte_table.selected_columns.clear()
            self._refresh_byte_table_view()

    def edit_column_definition(self, item):
        item_kind = item.data(LIST_KIND_ROLE)
        payload = item.data(LIST_PAYLOAD_ROLE)

        if item_kind == "split":
            self.edit_split_as_definition(payload)
            return

        if item_kind != "definition":
            return

        row = payload
        if row < 0 or row >= len(self.byte_table.column_definitions):
            return

        old_def = self.byte_table.column_definitions[row]
        dialog = AddColumnDialog(self)
        dialog.setWindowTitle("Edit Column Definition")
        dialog.label_input.setText(old_def.label)

        if old_def.unit == "byte":
            dialog.byte_radio.setChecked(True)
            dialog.start_byte_spin.setValue(old_def.start_byte)
            dialog.end_byte_spin.setValue(old_def.end_byte)
        else:
            dialog.bit_radio.setChecked(True)
            dialog.bit_pos_radio.setChecked(True)
            dialog.abs_bit_position_spin.setValue(old_def.start_bit)
            dialog.total_bits_spin.setValue(old_def.total_bits)

        format_text = self._display_format_dialog_label(old_def.display_format)
        index = dialog.format_combo.findText(format_text)
        if index >= 0:
            dialog.format_combo.setCurrentIndex(index)

        color_index = dialog.color_combo.findText(old_def.color_name)
        if color_index >= 0:
            dialog.color_combo.setCurrentIndex(color_index)

        if dialog.exec() == dialog.DialogCode.Accepted:
            self.byte_table.remove_column_definition(row)
            col_def = dialog.get_column_definition()
            self.byte_table.column_definitions.insert(row, col_def)
            self.refresh_column_definitions_list()
            self._refresh_byte_table_view()

    def remove_column_definition(self):
        item = self.columns_list.currentItem()
        if item is None:
            return

        item_kind = item.data(LIST_KIND_ROLE)
        payload = item.data(LIST_PAYLOAD_ROLE)

        if item_kind == "split":
            if hasattr(self.byte_table, "_remove_split_entry"):
                self.byte_table._remove_split_entry(payload)
            else:
                self.byte_table.split_columns.pop(payload, None)
            self.refresh_column_definitions_list()
            self._refresh_byte_table_view()
            return

        if item_kind == "definition" and 0 <= payload < len(self.byte_table.column_definitions):
            self.byte_table.remove_column_definition(payload)
            self.refresh_column_definitions_list()
            self._refresh_byte_table_view()

    def clear_column_definitions(self):
        self.byte_table.clear_column_definitions()
        self.byte_table.split_columns.clear()
        self.refresh_column_definitions_list()
        self._refresh_byte_table_view()

    def combine_selected_column_definitions(self):
        selected_items = self.columns_list.selectedItems()
        if len(selected_items) < 2:
            QMessageBox.warning(self, "Combine Selected", "Select at least two column entries to combine.")
            return

        resolved_sources = []
        for item in selected_items:
            source, error = self._resolve_combine_source(item)
            if error is not None:
                QMessageBox.warning(self, "Combine Selected", error)
                return
            resolved_sources.append(source)

        self._combine_resolved_sources(resolved_sources, "Combine Selected")

    def combine_selected_table_columns(self, selected_cols):
        resolved_sources, error = self._resolve_table_combine_sources(selected_cols)
        if error is not None:
            QMessageBox.warning(self, "Combine Selection", error)
            return
        self._combine_resolved_sources(resolved_sources, "Combine Selection")

    def combine_selected_table_columns_from_current_selection(self):
        self.combine_selected_table_columns(set(self.byte_table.selected_columns))

    def add_definition_from_undefined(self, start_bit, total_bits):
        dialog = AddColumnDialog(self)
        dialog.setWindowTitle("Define Field from Undefined Bits")
        dialog.bit_radio.setChecked(True)
        dialog.byte_radio.setChecked(False)
        dialog.bit_pos_radio.setChecked(True)
        dialog.byte_pos_radio.setChecked(False)
        dialog.abs_bit_position_spin.setValue(start_bit)
        dialog.total_bits_spin.setValue(total_bits)

        idx_fmt = dialog.format_combo.findText("Binary")
        if idx_fmt >= 0:
            dialog.format_combo.setCurrentIndex(idx_fmt)

        idx_color = dialog.color_combo.findText("None")
        if idx_color >= 0:
            dialog.color_combo.setCurrentIndex(idx_color)

        dialog.label_input.setText("?")

        if dialog.exec() == dialog.DialogCode.Accepted:
            col_def = dialog.get_column_definition()
            self.byte_table.add_column_definition(col_def)
            self.refresh_column_definitions_list()

    def parse_hex_pattern(self, text):
        s = text.strip().lower()
        if not s:
            return None
        if s.startswith("0x"):
            s = s[2:]
        s = s.replace(" ", "")
        if len(s) == 0 or len(s) % 2 != 0:
            return None
        try:
            return [int(s[i:i + 2], 16) for i in range(0, len(s), 2)]
        except ValueError:
            return None
