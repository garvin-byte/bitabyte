# Bitabyte C++ App

This is the main Bitabyte application.

## What It Does

Bitabyte is focused on reverse-engineering and inspecting binary data with a workflow built around:

- raw byte browsing
- bit-accurate framing
- sync and frame candidate discovery
- field definition and split views
- grouped frame navigation and filtering
- field inspection and live bit preview
- CSV export

## Main UI Areas

- center table view for raw bytes or framed rows
- left dock for column definitions and frame grouping
- right dock for the live bit viewer and field inspector

## Current Capabilities

- open, reload, and drag-drop files
- change bytes per row
- apply framing from a typed sync pattern
- frame from a selected table region
- run `Find Frames...` to discover likely framing candidates
- define columns from bit selections
- split fields into binary or nibble views
- group/filter framed rows by field values
- inspect selected values in multiple decoded formats
- highlight constant columns
- export the visible table to CSV

## Build Requirements

- CMake 3.21+
- Qt 6 Widgets development files
- a C++20 compiler

## Build

Generic example:

```powershell
cmake -S cpp/cpp_byte_mode -B cpp/build/cpp_byte_mode
cmake --build cpp/build/cpp_byte_mode
```

Windows Qt/MinGW example:

```powershell
$env:PATH = "C:\Qt\6.11.0\mingw_64\bin;C:\Qt\Tools\mingw1310_64\bin;$env:PATH"

& "C:\Program Files\CMake\bin\cmake.exe" `
  -S "cpp/cpp_byte_mode" `
  -B "cpp/build/cpp_byte_mode_mingw" `
  -G "MinGW Makefiles" `
  -DCMAKE_PREFIX_PATH="C:\Qt\6.11.0\mingw_64"

& "C:\Program Files\CMake\bin\cmake.exe" `
  --build "cpp/build/cpp_byte_mode_mingw"

& ".\cpp\build\cpp_byte_mode_mingw\bitabyte_cpp_byte_mode.exe"
```

## Project Layout

- `src/core`: small shared helpers for parsing and formatting
- `src/data`: file-backed byte/bit access
- `src/features`: framing, discovery, inspector, grouping, export, and field logic
- `src/models`: Qt models and table adapters
- `src/ui`: windows, dialogs, controllers, and widgets
- `docs`: architecture, style, and roadmap notes

## Related Docs

- [docs/ARCHITECTURE.md](c:/Users/thoma/PycharmProjects/bitabyte/cpp/cpp_byte_mode/docs/ARCHITECTURE.md)
- [docs/STYLE.md](c:/Users/thoma/PycharmProjects/bitabyte/cpp/cpp_byte_mode/docs/STYLE.md)
- [docs/ROADMAP.md](c:/Users/thoma/PycharmProjects/bitabyte/cpp/cpp_byte_mode/docs/ROADMAP.md)
