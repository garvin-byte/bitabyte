# Bitabyte

Bitabyte is now a C++/Qt desktop application for exploring binary files, defining fields, framing records, and discovering sync patterns.

Primary app:

- `cpp/cpp_byte_mode`

Build on this Windows Qt/MinGW setup:

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

Useful docs:

- `cpp/cpp_byte_mode/README.md`
- `cpp/cpp_byte_mode/docs/ARCHITECTURE.md`
- `cpp/cpp_byte_mode/docs/ROADMAP.md`
- `cpp/cpp_byte_mode/docs/STYLE.md`
