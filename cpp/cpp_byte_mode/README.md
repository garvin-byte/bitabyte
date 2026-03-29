# Bitabyte C++ Byte Mode

This is the primary Bitabyte application.

Current scope:

- byte mode only
- open a file
- reload the current file
- drag and drop a file onto the window
- view bytes in a `QTableView`
- change bytes per row
- apply framing from repeated bit-accurate sync matches
- clear framing and return to the raw byte layout
- show row start positions in the vertical header
- show the current selection in the status bar
- export the visible byte table as CSV

Build requirements:

- CMake 3.21+
- Qt 6 Widgets development files
- a C++20 compiler

Example build:

```powershell
cmake -S cpp/cpp_byte_mode -B cpp/build/cpp_byte_mode
cmake --build cpp/build/cpp_byte_mode
```

Example build on this Windows setup with Qt MinGW:

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

Design rules for this folder:

- Prefer descriptive names over short names.
- Use `(row, col)` for grid positions.
- Use `(x, y)` only for pixel coordinates.
- Never store or return `(col, row)`.
- Keep byte-mode behavior simple first, then add framing, definitions, and inspectors after the table is stable.

Folder layout:

- `src/app`
- `src/core`
- `src/data`
- `src/features`
- `src/models`
- `src/ui`
- `docs`

Useful docs:

- `docs/ARCHITECTURE.md`
- `docs/STYLE.md`
- `docs/ROADMAP.md`
