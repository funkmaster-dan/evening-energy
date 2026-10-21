# Changelog

## v1.1.0

- Simulate every charging and export window in the forecast, with separate export allowances allocated in time order while protecting household needs and reserve.
- Show 24 hours of recorded consumption, solar and battery charge, plus Open-Meteo recent-past temperature, before the forecast boundary.
- Add cumulative consumption, solar, export and grid-import summaries at 24-hour steps through the configured horizon.
- Compact the Overview, Models and Settings layouts, with chart summaries alongside plots on wide screens.
- Keep scenario labels and zero-export values in sync, and correct partial-hour power-chart interpolation.
- Default Models to configured live sensors, remove duplicate Settings controls, and require a 24–72-hour forecast length.
- Refresh the README and screenshots for the current interface.

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
