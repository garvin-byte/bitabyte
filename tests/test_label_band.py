#!/usr/bin/env python3
"""Test script to debug label band creation."""

from dataclasses import dataclass

@dataclass
class ColumnDefinition:
    start_byte: int
    end_byte: int
    label: str
    display_format: str
    color_name: str = "None"
    unit: str = "byte"
    start_bit: int = 0
    total_bits: int = 8


def build_label_band(column_definitions, width: int):
    labels = ["" for _ in range(width)]
    spans = []
    print(f"DEBUG: Building label band for width={width}")
    print(f"DEBUG: Column definitions count: {len(column_definitions)}")
    for col_def in column_definitions:
        print(f"DEBUG: Col def: start={col_def.start_byte}, end={col_def.end_byte}, label={col_def.label}, unit={col_def.unit}")
        if col_def.unit != "byte":
            print(f"DEBUG: Skipping non-byte column")
            continue
        if not col_def.label:
            print(f"DEBUG: Skipping empty label")
            continue
        start = max(0, min(width - 1, col_def.start_byte))
        end = max(0, min(width - 1, col_def.end_byte))
        if end < start:
            print(f"DEBUG: Skipping invalid range")
            continue
        length = end - start + 1
        print(f"DEBUG: Adding span: start={start}, length={length}, label={col_def.label}")
        spans.append((start, length, col_def.label))
        for idx in range(start, end + 1):
            labels[idx] = col_def.label
    print(f"DEBUG: Final spans: {spans}")
    return labels, spans


# Test with a 4-byte sync pattern at start
col_defs = [
    ColumnDefinition(
        start_byte=0,
        end_byte=3,
        label="Sync",
        display_format="hex",
        color_name="Sky",
        unit="byte",
    )
]

labels, spans = build_label_band(col_defs, 16)
print(f"\nResult:")
print(f"Labels: {labels}")
print(f"Spans: {spans}")
