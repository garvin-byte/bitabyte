"""Column-definition helpers for the main bit viewer window."""

from __future__ import annotations

import re

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QListWidgetItem, QMessageBox

from .column import AddColumnDialog, ColumnDefinition


LIST_KIND_ROLE = Qt.ItemDataRole.UserRole
LIST_PAYLOAD_ROLE = Qt.ItemDataRole.UserRole + 1


class BitViewerWindowColumnsMixin:
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

    def _split_payload_bit_bounds(self, payload):
        if isinstance(payload, tuple):
            if len(payload) == 4 and payload[0] == "segment":
                _, split_key, segment_start, segment_end = payload
                if not isinstance(split_key, int):
                    return None
                return self._absolute_bit_bounds(split_key, segment_start, segment_end)

            byte_idx, nibble_type = payload
            split_info = self.byte_table.split_columns.get(payload)
            if split_info is None or split_info.get("type") != "nibble_binary":
                return None
            bit_start = split_info.get("bit_start", 0)
            return self._absolute_bit_bounds(byte_idx, bit_start, bit_start + 3)

        split_info = self.byte_table.split_columns.get(payload)
        if split_info is None:
            return None

        return self._absolute_bit_bounds(payload, 0, 7)

    def _combine_display_kind_for_format(self, display_format):
        if display_format == "binary":
            return "binary"
        if display_format in {"hex", "hex_be", "hex_le"}:
            return "hex"
        return None

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

            display_kind = "binary" if segment.get("format") == "binary" else "hex"
            label = segment.get("label", "")
            color_name = segment.get("color", split_info.get("color", "None"))
            start_bit, end_bit = self._absolute_bit_bounds(split_key, segment_start, segment_end)
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

        if isinstance(payload, tuple):
            byte_idx, _ = payload
            if split_type != "nibble_binary":
                return None, "Only binary and hex split columns can be combined."
            bit_start = split_info.get("bit_start", 0)
            start_bit, end_bit = self._absolute_bit_bounds(byte_idx, bit_start, bit_start + 3)
            display_kind = "binary"
        else:
            start_bit, end_bit = self._absolute_bit_bounds(payload, 0, 7)
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
            return None, "Select two columns to combine."

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
        if len(resolved_sources) != 2:
            return None, "Select exactly two visible columns to combine."

        return resolved_sources, None

    def can_combine_selected_table_columns(self, selected_cols):
        sources, error = self._resolve_table_combine_sources(selected_cols)
        if error is not None or sources is None:
            return False

        sorted_sources = sorted(sources, key=lambda source: (source["start_bit"], source["end_bit"]))
        first_source, second_source = sorted_sources
        if first_source["end_bit"] >= second_source["start_bit"]:
            return False
        if first_source["end_bit"] + 1 != second_source["start_bit"]:
            return False

        total_bits = second_source["end_bit"] - first_source["start_bit"] + 1
        return self._combined_display_format(sources, total_bits) is not None

    def _combine_resolved_sources(self, resolved_sources, title):
        sorted_sources = sorted(resolved_sources, key=lambda source: (source["start_bit"], source["end_bit"]))
        first_source, second_source = sorted_sources

        if first_source["end_bit"] >= second_source["start_bit"]:
            QMessageBox.warning(self, title, "Selected columns overlap and cannot be combined.")
            return

        if first_source["end_bit"] + 1 != second_source["start_bit"]:
            QMessageBox.warning(self, title, "Selected columns must be adjacent to combine.")
            return

        total_bits = second_source["end_bit"] - first_source["start_bit"] + 1
        display_format = self._combined_display_format(resolved_sources, total_bits)
        if display_format is None:
            QMessageBox.warning(
                self,
                title,
                "Mixed binary and hex columns can only be combined when the total width is a multiple of 4 bits.",
            )
            return

        color_name = next(
            (
                source["color_name"]
                for source in resolved_sources
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

    def _column_definition_description(self, col_def):
        if col_def.unit == "byte":
            return (
                f"{col_def.label if col_def.label else '(unnamed)'}: "
                f"Bytes [{col_def.start_byte}-{col_def.end_byte}] as {col_def.display_format} ({col_def.color_name})"
            )
        return (
                f"{col_def.label if col_def.label else '(unnamed)'}: "
                f"Bit {col_def.start_bit}+{col_def.total_bits} as {col_def.display_format} ({col_def.color_name})"
            )

    def _split_definition_description(self, split_key, split_info):
        label = split_info.get("label") or split_info.get("type", "Split").title()
        color_name = split_info.get("color", "None")

        if isinstance(split_key, tuple):
            byte_idx, nibble_type = split_key
            nibble_label = "High nibble" if nibble_type == "high" else "Low nibble"
            return f"{label}: Byte {byte_idx} {nibble_label} ({color_name})"

        split_type = split_info.get("type", "split")
        if split_type == "custom_bits":
            labeled_segment = next(
                (segment for segment in split_info.get("segments", []) if segment.get("label")),
                None,
            )
            if labeled_segment is not None:
                return (
                    f"{label}: Byte {split_key} bits "
                    f"{labeled_segment['start']}-{labeled_segment['end']} ({color_name})"
                )
        return f"{label}: Byte {split_key} ({color_name})"

    def _custom_split_segment_description(self, split_key, segment):
        label = segment.get("label") or segment.get("format", "Split").title()
        color_name = segment.get("color", "None")
        return f"{label}: Byte {split_key} bits {segment['start']}-{segment['end']} ({color_name})"

    def _add_columns_list_item(self, text, color_name, kind, payload):
        item = QListWidgetItem(text)
        item.setData(LIST_KIND_ROLE, kind)
        item.setData(LIST_PAYLOAD_ROLE, payload)
        if color_name != "None":
            color = self.byte_table._color_from_name(color_name)
            if color:
                item.setBackground(color)
        self.columns_list.addItem(item)

    def refresh_column_definitions_list(self):
        self.columns_list.clear()

        for index, col_def in enumerate(self.byte_table.column_definitions):
            self._add_columns_list_item(
                self._column_definition_description(col_def),
                col_def.color_name,
                "definition",
                index,
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
                    segment_start_bit, segment_end_bit = self._absolute_bit_bounds(
                        split_key,
                        segment.get("start", 0),
                        segment.get("end", 0),
                    )
                    if self._range_fully_defined(segment_start_bit, segment_end_bit):
                        continue
                    self._add_columns_list_item(
                        self._custom_split_segment_description(split_key, segment),
                        segment.get("color", split_info.get("color", "None")),
                        "split",
                        ("segment", split_key, segment.get("start"), segment.get("end")),
                    )
            else:
                split_bounds = self._split_payload_bit_bounds(split_key)
                if split_bounds is not None and self._range_fully_defined(*split_bounds):
                    continue
                self._add_columns_list_item(
                    self._split_definition_description(split_key, split_info),
                    split_info.get("color", "None"),
                    "split",
                    split_key,
                )

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
            if hasattr(self.byte_table, "_edit_split_label"):
                if isinstance(payload, tuple) and len(payload) == 4 and payload[0] == "segment":
                    self.byte_table._edit_split_label(payload[1])
                else:
                    self.byte_table._edit_split_label(payload)
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

        format_map = {
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

        format_text = format_map.get(old_def.display_format, "Hex")
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
        if len(selected_items) != 2:
            QMessageBox.warning(self, "Combine Selected", "Select exactly two column entries to combine.")
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
