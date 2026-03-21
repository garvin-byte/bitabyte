"""Shared helpers for viewer bit formatting and highlight lookups."""

from __future__ import annotations

import bisect

import numpy as np


def apply_bit_order(bits, bit_order: str):
    if bit_order == "msb":
        return bits
    return bits[::-1]


def bits_to_int(bits):
    if bits is None or len(bits) == 0:
        return None
    return int("".join(str(int(bit)) for bit in bits.tolist()), 2)


def bits_to_ascii(bits) -> str:
    if bits is None or len(bits) == 0:
        return "(no data)"

    pad = (8 - (len(bits) % 8)) % 8
    padded = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)]) if pad else bits

    chars = []
    for byte_value in np.packbits(padded):
        chars.append(chr(int(byte_value)) if 32 <= byte_value <= 126 else ".")
    return "".join(chars)


def build_highlight_intervals(positions, pattern_length: int):
    if not positions or pattern_length <= 0:
        return []
    return sorted((pos, pos + pattern_length) for pos in positions)


def is_bit_highlighted(highlight_intervals, bit_index: int) -> bool:
    if not highlight_intervals:
        return False

    starts = [start for start, _ in highlight_intervals]
    interval_index = bisect.bisect_right(starts, bit_index) - 1
    if interval_index < 0:
        return False
    start, end = highlight_intervals[interval_index]
    return start <= bit_index < end
