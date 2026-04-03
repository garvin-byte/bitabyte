# Bitabyte

Bitabyte is a desktop tool for exploring binary files, defining fields, discovering frame boundaries, and inspecting structured data at bit and byte level.

The active application is the C++/Qt version in [cpp/cpp_byte_mode](cpp/cpp_byte_mode).

## Current Features

- open, reload, and drag-drop binary files into the app
- browse raw bytes with configurable bytes-per-row
- apply bit-accurate framing from a typed sync pattern or a table selection
- run `Find Frames...` to rank candidate frame layouts and preview them before applying
- inspect detected constant and counter-like fields after framing
- define columns and split views for bytes, nibbles, and individual bits
- group and filter framed rows from the left dock
- inspect selected values in the live bit viewer, current value panel, and framed distribution view
- highlight constant columns and export the visible table to CSV

## Build

Requirements:

- CMake 3.21+
- Qt 6 Widgets development files
- a C++20 compiler

Example build on the current Windows Qt/MinGW setup:

```powershell
$env:PATH = "C:\Qt\6.11.0\mingw_64\bin;C:\Qt\Tools\mingw1310_64\bin;$env:PATH"

& "C:\Program Files\CMake\bin\cmake.exe" `
  -S "cpp/cpp_byte_mode" `
  -B "cpp/build/cpp_byte_mode_mingw" `
  -G "MinGW Makefiles" `
  -DCMAKE_PREFIX_PATH="C:\Qt\6.11.0\mingw_64"

& "C:\Program Files\CMake\bin\cmake.exe" `
  --build "cpp/build/cpp_byte_mode_mingw"
```

Run:

```powershell
.\cpp\build\cpp_byte_mode_mingw\bitabyte_cpp_byte_mode.exe
```

## Docs

- [cpp/cpp_byte_mode/README.md](cpp/cpp_byte_mode/README.md)
- [cpp/cpp_byte_mode/docs/ARCHITECTURE.md](cpp/cpp_byte_mode/docs/ARCHITECTURE.md)
- [cpp/cpp_byte_mode/docs/ROADMAP.md](cpp/cpp_byte_mode/docs/ROADMAP.md)
- [cpp/cpp_byte_mode/docs/STYLE.md](cpp/cpp_byte_mode/docs/STYLE.md)
