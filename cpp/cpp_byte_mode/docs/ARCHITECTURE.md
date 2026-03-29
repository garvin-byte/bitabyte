# C++ Byte Mode Architecture

This folder is intentionally split by responsibility:

- `src/app`
  Startup and application entry point only.
- `src/core`
  Small shared helpers for formatting and parsing.
- `src/data`
  File-backed or memory-backed byte sources.
- `src/features`
  Isolated feature slices that sit above the data/model layer.
- `src/models`
  Qt models that adapt data into `QTableView`.
- `src/ui`
  Windows, views, menus, dialogs, and presentation behavior.

Current implemented slice:

- file load
- file reload
- drag and drop file load
- byte table view
- bytes-per-row control
- bit-accurate sync-pattern framing from one match to the next
- frame clear back to raw layout
- live selection status
- CSV export

Planned next feature folders:

- `src/features/columns`
- `src/features/inspector`
- `src/features/statistics`

Code conventions for this folder:

- descriptive names over shorthand
- `(row, col)` for grid positions
- `(x, y)` for pixel positions only
- no `(col, row)` ordering in internal code
- keep `MainWindow` orchestration-focused and move reusable logic into `core`, `data`, `models`, or `features`
