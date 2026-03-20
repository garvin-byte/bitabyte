"""ColumnDefinition data class and AddColumnDialog."""

import numpy as np
from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout,
                              QLabel, QSpinBox, QComboBox, QCheckBox, QWidget,
                              QGroupBox, QButtonGroup, QRadioButton, QListWidget,
                              QListWidgetItem, QScrollArea, QSizePolicy, QPushButton,
                              QLineEdit, QMessageBox, QAbstractItemView)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QPixmap
from .colors import COLOR_OPTIONS, COLOR_NAME_TO_QCOLOR, _populate_color_combo


class ColumnDefinition:
    """
    Represents a column definition in byte mode.

    unit = "byte": start_byte..end_byte are byte-column indices and can span
                   multiple bytes (combined value).
    unit = "bit" : Bit-based definition with total_bits length.
                   start_bit is the absolute bit position.
                   Endianness determined by display_format (dec_be, dec_le, etc.)
    """

    def __init__(
            self,
            start_byte,
            end_byte,
            label,
            display_format,
            color_name=None,
            unit="byte",
            start_bit=0,
            total_bits=8,
    ):
        self.label = label
        # "hex", "binary", "dec_be", "dec_le", "tc_be", "tc_le", "ascii"
        self.display_format = display_format
        # highlight colour name: None, "Yellow", "Cyan", "Magenta", "Green"
        self.color_name = color_name or "None"
        # "byte" or "bit"
        self.unit = unit

        # Calculate byte positions based on unit
        if self.unit == "bit":
            # start_bit is absolute bit position
            self.start_bit = start_bit
            self.total_bits = total_bits

            # Calculate which bytes this spans
            self.start_byte = start_bit // 8
            end_bit_position = start_bit + total_bits - 1
            self.end_byte = end_bit_position // 8
        else:
            self.start_byte = start_byte
            self.end_byte = end_byte
            self.start_bit = 0
            self.total_bits = (end_byte - start_byte + 1) * 8


class AddColumnDialog(QDialog):
    """Dialog for adding a column definition with flexible bit/byte selection"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Column Definition")
        self.setModal(True)
        self.resize(450, 400)

        layout = QVBoxLayout(self)

        # Label
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("Label:"))
        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("e.g., Sync, Timestamp, Data")
        label_layout.addWidget(self.label_input)
        layout.addLayout(label_layout)

        # Unit: bytes vs bits
        unit_group_box = QGroupBox("Definition Unit")
        unit_layout = QVBoxLayout(unit_group_box)

        self.byte_radio = QRadioButton("Bytes")
        self.bit_radio = QRadioButton("Bits")
        self.byte_radio.setChecked(True)

        unit_layout.addWidget(self.byte_radio)
        unit_layout.addWidget(self.bit_radio)

        layout.addWidget(unit_group_box)

        # === BYTE MODE CONTROLS ===
        self.byte_controls = QWidget()
        byte_layout = QVBoxLayout(self.byte_controls)
        byte_layout.setContentsMargins(0, 0, 0, 0)

        # Start byte
        start_byte_layout = QHBoxLayout()
        start_byte_layout.addWidget(QLabel("Start Byte:"))
        self.start_byte_spin = QSpinBox()
        self.start_byte_spin.setRange(0, 1000)
        self.start_byte_spin.setValue(0)
        start_byte_layout.addWidget(self.start_byte_spin)
        byte_layout.addLayout(start_byte_layout)

        # End byte
        end_byte_layout = QHBoxLayout()
        end_byte_layout.addWidget(QLabel("End Byte:"))
        self.end_byte_spin = QSpinBox()
        self.end_byte_spin.setRange(0, 1000)
        self.end_byte_spin.setValue(0)
        end_byte_layout.addWidget(self.end_byte_spin)
        byte_layout.addLayout(end_byte_layout)

        layout.addWidget(self.byte_controls)

        # === BIT MODE CONTROLS ===
        self.bit_controls = QWidget()
        bit_layout = QVBoxLayout(self.bit_controls)
        bit_layout.setContentsMargins(0, 0, 0, 0)

        # Bit position input method
        bit_input_group = QGroupBox("Bit Position Input")
        bit_input_layout = QVBoxLayout(bit_input_group)

        self.bit_input_button_group = QButtonGroup()
        self.bit_pos_radio = QRadioButton("Bit Position (absolute)")
        self.byte_pos_radio = QRadioButton("Byte Position (byte + offset)")
        self.bit_pos_radio.setChecked(True)

        self.bit_input_button_group.addButton(self.bit_pos_radio)
        self.bit_input_button_group.addButton(self.byte_pos_radio)

        bit_input_layout.addWidget(self.bit_pos_radio)
        bit_input_layout.addWidget(self.byte_pos_radio)
        bit_layout.addWidget(bit_input_group)

        # Absolute bit position input
        self.absolute_bit_widget = QWidget()
        abs_bit_layout = QVBoxLayout(self.absolute_bit_widget)
        abs_bit_layout.setContentsMargins(0, 0, 0, 0)

        abs_pos_layout = QHBoxLayout()
        abs_pos_layout.addWidget(QLabel("Starting Bit Position:"))
        self.abs_bit_position_spin = QSpinBox()
        self.abs_bit_position_spin.setRange(0, 10000)
        self.abs_bit_position_spin.setValue(0)
        abs_pos_layout.addWidget(self.abs_bit_position_spin)
        abs_bit_layout.addLayout(abs_pos_layout)

        bit_layout.addWidget(self.absolute_bit_widget)

        # Byte + offset position input
        self.byte_offset_widget = QWidget()
        byte_off_layout = QVBoxLayout(self.byte_offset_widget)
        byte_off_layout.setContentsMargins(0, 0, 0, 0)

        byte_pos_layout = QHBoxLayout()
        byte_pos_layout.addWidget(QLabel("Starting Byte:"))
        self.bit_start_byte_spin = QSpinBox()
        self.bit_start_byte_spin.setRange(0, 1000)
        self.bit_start_byte_spin.setValue(0)
        byte_pos_layout.addWidget(self.bit_start_byte_spin)
        byte_off_layout.addLayout(byte_pos_layout)

        bit_offset_layout = QHBoxLayout()
        bit_offset_layout.addWidget(QLabel("Bit Offset in Byte:"))
        self.bit_offset_spin = QSpinBox()
        self.bit_offset_spin.setRange(0, 7)
        self.bit_offset_spin.setValue(0)
        bit_offset_layout.addWidget(self.bit_offset_spin)
        byte_off_layout.addLayout(bit_offset_layout)

        bit_layout.addWidget(self.byte_offset_widget)
        self.byte_offset_widget.hide()

        # Total bits
        total_bits_layout = QHBoxLayout()
        total_bits_layout.addWidget(QLabel("Total Bits:"))
        self.total_bits_spin = QSpinBox()
        self.total_bits_spin.setRange(1, 128)
        self.total_bits_spin.setValue(8)
        total_bits_layout.addWidget(self.total_bits_spin)
        bit_layout.addLayout(total_bits_layout)

        # Example label
        self.bit_example_label = QLabel()
        self.bit_example_label.setWordWrap(True)
        self.bit_example_label.setStyleSheet("color: #666; font-style: italic;")
        bit_layout.addWidget(self.bit_example_label)

        layout.addWidget(self.bit_controls)
        self.bit_controls.hide()

        # Display format
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Display Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems([
            "Hex (MSBF)",  # hex_be
            "Hex (LSBF)",  # hex_le
            "Binary",  # binary (no endianness)
            "Decimal (MSBF)",  # dec_be
            "Decimal (LSBF)",  # dec_le
            "Twos complement (MSBF)",  # tc_be
            "Twos complement (LSBF)",  # tc_le
            "ASCII (MSBF)",  # ascii_be
            "ASCII (LSBF)",  # ascii_le
        ])
        format_layout.addWidget(self.format_combo)
        layout.addLayout(format_layout)

        # Color choice
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Highlight Color:"))
        self.color_combo = QComboBox()
        _populate_color_combo(self.color_combo, "None")
        color_layout.addWidget(self.color_combo)
        layout.addLayout(color_layout)

        # Connect radio buttons
        # Connect radio buttons
        def update_format_enabled():
            """
            Enable / disable formats based on alignment:

            ASCII:
              - Byte mode: always allowed
              - Bit mode: only if start_bit is byte-aligned AND total_bits is multiple of 8

            Hex:
              - Byte mode: always allowed
              - Bit mode: only if start_bit is nibble-aligned AND total_bits is multiple of 4
            """
            # Defaults
            ascii_valid = False
            hex_valid = False

            # Byte mode: everything is nicely aligned
            if self.byte_radio.isChecked():
                ascii_valid = True
                hex_valid = True
            else:
                # Bit mode: compute absolute start_bit
                if self.bit_pos_radio.isChecked():
                    start_bit = self.abs_bit_position_spin.value()
                else:
                    start_byte_val = self.bit_start_byte_spin.value()
                    bit_offset = self.bit_offset_spin.value()
                    start_bit = start_byte_val * 8 + bit_offset

                total_bits = self.total_bits_spin.value()

                # ASCII -> full bytes only
                ascii_valid = ((start_bit % 8) == 0) and ((total_bits % 8) == 0)
                # Hex -> nibble aligned
                hex_valid = ((start_bit % 4) == 0) and ((total_bits % 4) == 0)

            # Enable/disable ASCII entries
            for label in ["ASCII (MSBF)", "ASCII (LSBF)"]:
                idx = self.format_combo.findText(label)
                if idx >= 0:
                    item = self.format_combo.model().item(idx)
                    if item:
                        item.setEnabled(ascii_valid)

            # Enable/disable Hex entries
            for label in ["Hex (MSBF)", "Hex (LSBF)"]:
                idx = self.format_combo.findText(label)
                if idx >= 0:
                    item = self.format_combo.model().item(idx)
                    if item:
                        item.setEnabled(hex_valid)

            # Fix current selection if it became invalid
            current_text = self.format_combo.currentText()

            # If ASCII currently selected but no longer valid
            if current_text.startswith("ASCII") and not ascii_valid:
                # Prefer Hex if that's valid, otherwise fall back to Binary
                fallback = "Hex (MSBF)" if hex_valid else "Binary"
                idx = self.format_combo.findText(fallback)
                if idx >= 0:
                    self.format_combo.setCurrentIndex(idx)

            # If Hex currently selected but no longer valid
            elif current_text.startswith("Hex") and not hex_valid:
                # Prefer ASCII if that's valid, otherwise fall back to Binary
                fallback = "ASCII (MSBF)" if ascii_valid else "Binary"
                idx = self.format_combo.findText(fallback)
                if idx >= 0:
                    self.format_combo.setCurrentIndex(idx)

        def update_mode():
            if self.byte_radio.isChecked():
                self.byte_controls.show()
                self.bit_controls.hide()
            else:
                self.byte_controls.hide()
                self.bit_controls.show()
                update_bit_input_mode()
                update_bit_example()
            update_format_enabled()

        self.byte_radio.toggled.connect(update_mode)
        self.bit_radio.toggled.connect(update_mode)

        # Connect bit input mode toggle
        def update_bit_input_mode():
            if self.bit_pos_radio.isChecked():
                self.absolute_bit_widget.show()
                self.byte_offset_widget.hide()
            else:
                self.absolute_bit_widget.hide()
                self.byte_offset_widget.show()
            update_bit_example()
            update_format_enabled()

        self.bit_pos_radio.toggled.connect(update_bit_input_mode)
        self.byte_pos_radio.toggled.connect(update_bit_input_mode)

        # Update bit example
        def update_bit_example():
            if self.bit_radio.isChecked():
                if self.bit_pos_radio.isChecked():
                    # Absolute bit position
                    start_bit = self.abs_bit_position_spin.value()
                    total_bits = self.total_bits_spin.value()

                    start_byte = start_bit // 8
                    end_bit = start_bit + total_bits - 1
                    end_byte = end_bit // 8

                    if end_byte == start_byte:
                        self.bit_example_label.setText(
                            f"Example: Extracts bits {start_bit}-{end_bit} (all in byte {start_byte})"
                        )
                    else:
                        self.bit_example_label.setText(
                            f"Example: Extracts {total_bits} bits starting at bit {start_bit}\n"
                            f"(spans bytes {start_byte}-{end_byte})"
                        )
                else:
                    # Byte + offset
                    start_byte = self.bit_start_byte_spin.value()
                    bit_offset = self.bit_offset_spin.value()
                    total_bits = self.total_bits_spin.value()

                    start_bit = start_byte * 8 + bit_offset
                    end_bit = start_bit + total_bits - 1
                    end_byte = end_bit // 8

                    if end_byte == start_byte:
                        self.bit_example_label.setText(
                            f"Example: Extracts bits {bit_offset}-{bit_offset + total_bits - 1} from byte {start_byte}\n"
                            f"(absolute bit position {start_bit}-{end_bit})"
                        )
                    else:
                        self.bit_example_label.setText(
                            f"Example: Extracts {total_bits} bits from byte {start_byte}, offset {bit_offset}\n"
                            f"(absolute bit {start_bit}-{end_bit}, spans bytes {start_byte}-{end_byte})"
                        )
            update_format_enabled()

        self.abs_bit_position_spin.valueChanged.connect(update_bit_example)
        self.bit_start_byte_spin.valueChanged.connect(update_bit_example)
        self.bit_offset_spin.valueChanged.connect(update_bit_example)
        self.total_bits_spin.valueChanged.connect(update_bit_example)

        update_mode()

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_column_definition(self):
        """Return the column definition from the dialog"""
        label = self.label_input.text().strip()

        if self.byte_radio.isChecked():
            unit = "byte"
            start_byte = self.start_byte_spin.value()
            end_byte = self.end_byte_spin.value()
            start_bit = 0
            total_bits = 8
        else:
            unit = "bit"

            # Calculate absolute bit position
            if self.bit_pos_radio.isChecked():
                start_bit = self.abs_bit_position_spin.value()
            else:
                start_byte_val = self.bit_start_byte_spin.value()
                bit_offset = self.bit_offset_spin.value()
                start_bit = start_byte_val * 8 + bit_offset

            total_bits = self.total_bits_spin.value()

            # These will be calculated by ColumnDefinition
            start_byte = 0
            end_byte = 0

        fmt_label = self.format_combo.currentText()
        fmt_map = {
            "Hex (MSBF)": "hex_be",
            "Hex (LSBF)": "hex_le",
            "Binary": "binary",
            "Decimal (MSBF)": "dec_be",
            "Decimal (LSBF)": "dec_le",
            "Twos complement (MSBF)": "tc_be",
            "Twos complement (LSBF)": "tc_le",
            "ASCII (MSBF)": "ascii_be",
            "ASCII (LSBF)": "ascii_le",
        }
        display_format = fmt_map.get(fmt_label, "hex_be")

        color_name = self.color_combo.currentText()

        return ColumnDefinition(
            start_byte,
            end_byte,
            label,
            display_format,
            color_name=color_name,
            unit=unit,
            start_bit=start_bit,
            total_bits=total_bits
        )
