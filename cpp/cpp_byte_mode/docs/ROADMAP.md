# Bitabyte C++ Roadmap

The C++ app is already beyond the original "byte mode only" starting point. The current roadmap is about deepening the workflow and continuing cleanup.

## Landed

- raw byte browsing
- bytes-per-row control
- drag-drop and reload flow
- CSV export
- framing from typed or selected sync patterns
- `Find Frames...` discovery dialog
- column definitions and split views
- field inspector and live bit viewer
- frame grouping and branch filtering
- controller-based reduction of `MainWindow` responsibilities

## Next Likely Work

- continue shrinking `MainWindow` by pulling more session/file behavior into focused controllers
- improve public documentation and examples
- keep refining `Find Frames...` ranking and readability
- add more inspector/decode helpers where they materially help binary analysis

## Longer-Term Ideas

- workspace save/load
- import/export of user-defined layouts or field definitions
- richer multi-file comparison workflows
- more advanced anomaly/statistics tooling once the core framed workflow is stable
