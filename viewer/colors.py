"""Color constants and palette helpers for the bit viewer."""

from common.color_palette import COLOR_NAME_TO_QCOLOR, COLOR_OPTIONS, populate_color_combo

BIT_SYNC_WARNING_BYTES = 256 * 1024  # 256 KB
BIT_SYNC_HARD_LIMIT_BYTES = 1 * 1024 * 1024  # 1 MB
MAX_SYNC_FRAMES = 10_000
PATTERN_SCAN_MAX_ELEMENTS = 4_000_000  # Limit 2D comparison blocks to ~4M elements


def _populate_color_combo(combo, current_name="None"):
    populate_color_combo(combo, current_name)

