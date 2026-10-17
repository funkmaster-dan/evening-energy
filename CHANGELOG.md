# Changelog

## v1.0.0

- Local Home Assistant consumption, solar and battery forecasting with separate LightGBM models for home use and each panel orientation.
- Overview with live readings, consumption/solar uncertainty bands, temperature, synchronized chart inspection and battery charge projections.
- Conservative, middle and optimistic day/night energy scenarios with separate stored battery excess and achievable grid export.
- Multiple charging and export windows, overnight schedules, shared charging import budgets and per-window rates capped by inverter limits.
- Minimum charge and additional reserve checks across the complete forecast horizon, with zero allowances when inputs or calibration are not ready.
- Prepare/train/review/activate workflow with history coverage, quality reports, validation charts and explicit candidate selection.
- Sensor roles for battery and inverter calibration, history-only consumption sources, nighttime solar checks and map-based location settings.
- Read-only forecast reporting through the separate Energy Excess Home Assistant integration.
- Rewritten documentation and current screenshots captured from the app running in local Podman, with location and connection details excluded.
