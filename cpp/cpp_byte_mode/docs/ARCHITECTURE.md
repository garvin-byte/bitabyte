# Bitabyte C++ Architecture

The C++ app is split by responsibility so feature logic stays out of the main window as much as possible.

## Top-Level Structure

- `src/core`
  Small shared helpers for formatting, parsing, and lightweight utilities.
- `src/data`
  File-backed byte/bit access and shared data ownership.
- `src/features`
  Framing, frame discovery, field inspection, grouping, export, and field-definition logic.
- `src/models`
  Qt models that adapt raw and framed data to the table and tree views.
- `src/ui`
  Windows, dialogs, widgets, and controller-style orchestration code.

## Main Window Shape

`MainWindow` is intended to stay as the composition root for the app:

- menu and dock construction
- central widget and view wiring
- high-level orchestration between controllers

Feature workflows are being moved out of `MainWindow` into focused controllers in `src/ui`, including:

- file/session handling
- framing actions
- frame browser/grouping behavior
- inspection and live-viewer refresh

## Important Feature Areas

- `features/framing`
  Frame spans, row ordering, and framing layout rules.
- `features/frame_sync`
  Direct sync-pattern framing from user input.
- `features/bitstream_sync_discovery`
  Ranked frame/sync candidate discovery and analysis.
- `features/frame_browser`
  Grouping keys, grouping values, and frame-tree support.
- `features/inspector`
  Field statistics, value decoding, and selection analysis.

## Model/View Split

- `models/byte_table_model`
  Source model for raw and framed table display.
- `models/frame_group_tree_model`
  Tree model for grouped frame navigation/filtering.
- `ui/byte_table_view`
  The table presentation layer and interaction surface.

## Current Design Rules

- keep reusable logic in `core`, `data`, `features`, or `models`
- keep UI orchestration in `ui`
- prefer narrow controllers over adding more behavior to `MainWindow`
- keep framing and discovery logic independent from widget code
- use `(row, col)` for table/grid coordinates and `(x, y)` only for pixels
