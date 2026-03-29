# C++ Byte Mode Style Rules

These rules are the baseline for the C++ rewrite path.

Naming:

- Avoid single-letter and double-letter variable names whenever possible.
- Prefer descriptive names over clever shorthand.
- Acceptable short names are limited to conventional cases like `i`, `x`, `y`, `row`, and `col`.
- Abbreviations should still be readable. `row`, `col`, and `adj` are fine. Cryptic names are not.

Coordinate conventions:

- Use `(row, col)` for grid and table positions.
- Row always comes before column.
- Function parameters, tuples, and stored fields should follow `(row, col)`.
- Never use `(col, row)` in internal code.
- Use `(x, y)` only for pixel coordinates.
- Convert from pixels to cells as:
  - `col = x / square_size`
  - `row = y / square_size`
- When returning or storing that result, keep it as `(row, col)`.

Architecture rules:

- Keep UI orchestration in `src/ui`.
- Keep file/data ownership in `src/data`.
- Keep reusable formatting or parsing helpers in `src/core`.
- Keep feature-specific logic in `src/features`.
- Keep model/view adaptation in `src/models`.

Readability rules:

- Prefer explicit helper names over comments that decode cryptic variables.
- Split UI wiring from feature logic when a window method starts doing too much.
- Make the folder name reflect the responsibility of the code inside it.
