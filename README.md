# Bitabyte

Bitabyte is a desktop tool for exploring binary files, defining fields, discovering frame boundaries, and inspecting structured data at bit and byte level.

The active application is the C++/Qt version in [cpp/cpp_byte_mode](c:/Users/thoma/PycharmProjects/bitabyte/cpp/cpp_byte_mode).

## Current Features

- open files, reload files, and drag-and-drop data into the app
- byte-table browsing with configurable bytes-per-row
- bit-accurate sync framing from a typed pattern or table selection
- `Find Frames...` candidate discovery with ranked framing results
- column definitions and split views for bytes, nibbles, and individual bits
- frame grouping and branch filtering in the left dock
- live bit viewer and field inspector
- constant-column highlighting and CSV export

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

- [cpp/cpp_byte_mode/README.md](c:/Users/thoma/PycharmProjects/bitabyte/cpp/cpp_byte_mode/README.md)
- [cpp/cpp_byte_mode/docs/ARCHITECTURE.md](c:/Users/thoma/PycharmProjects/bitabyte/cpp/cpp_byte_mode/docs/ARCHITECTURE.md)
- [cpp/cpp_byte_mode/docs/ROADMAP.md](c:/Users/thoma/PycharmProjects/bitabyte/cpp/cpp_byte_mode/docs/ROADMAP.md)
- [cpp/cpp_byte_mode/docs/STYLE.md](c:/Users/thoma/PycharmProjects/bitabyte/cpp/cpp_byte_mode/docs/STYLE.md)
