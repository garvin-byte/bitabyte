"""Shared helpers for parsing bit patterns and tap lists."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def bits_from_hex_string(hex_string: str) -> np.ndarray | None:
    """Convert a hex string like ``0x47`` or ``47`` into a bit array."""
    cleaned = hex_string.strip()
    if cleaned.startswith(("0x", "0X")):
        cleaned = cleaned[2:]
    if not cleaned:
        return None

    try:
        value = int(cleaned, 16)
    except ValueError:
        return None

    bit_length = len(cleaned) * 4
    binary_string = format(value, f"0{bit_length}b")
    return np.fromiter((int(bit) for bit in binary_string), dtype=np.uint8)


def bits_from_binary_string(
    binary_string: str,
    *,
    aliases: dict[str, str] | None = None,
    ignored_characters: Iterable[str] = (),
) -> np.ndarray | None:
    """Convert a binary string into a bit array."""
    normalized = []
    alias_map = aliases or {}
    ignored = set(ignored_characters)

    for raw_char in binary_string.strip():
        if raw_char in ignored:
            continue
        char = alias_map.get(raw_char, raw_char)
        if char not in {"0", "1"}:
            return None
        normalized.append(int(char))

    if not normalized:
        return None
    return np.array(normalized, dtype=np.uint8)


def parse_tap_list(taps_string: str) -> list[int] | None:
    """Parse tap lists like ``0,1,7`` or ``0 1 7``."""
    try:
        return [int(token.strip()) for token in taps_string.replace(",", " ").split() if token.strip()]
    except ValueError:
        return None
