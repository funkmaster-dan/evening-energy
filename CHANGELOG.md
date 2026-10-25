# Changelog

## v1.2.2

- Show consumption and solar history as interpolated hourly averages, matching the forecast presentation.
- Keep completed-hour averages fixed and connect the latest hour's average so far to its measurement cutoff using the same solid line.
- Preserve five-minute battery history and clarify the power averaging basis in chart tooltips.

## v1.2.1

- Replace routine model searches with bounded LightGBM fitting and energy-aware stopping on chronological development forecasts.
- Save stopping and validation forecasts during preparation so training and candidate activation can run offline.
- Preserve prepared data and pending candidates across code updates; keep active model inputs independent of new preparations.
- Keep forecasts refreshing during training and compare active/candidate metrics on identical validation issues.

## v1.2.0

- Add combined solar production models using one measured AC sensor, panel counts and bank orientations.
- Compare solar mean and median forecasts with physical and residual alternatives, with chronological uncertainty calibration and clearer validation reports.
- Share data-preparation logic, exclude history before recorded readings begin, and detect stuck power readings while preserving normal nighttime zeros and full battery charge.
- Remove the inverter-efficiency calibration model and its forecast multiplier.
- Reorganize Settings into Basic settings, Live sensor data, Battery, Solar array, Time of use and Advanced cards.
- Keep candidate-only validation separate from paired model comparisons and respect training cutoffs during filtering.
- Clear outdated prepared datasets and pending candidates after code updates; recorded history, settings and active models are retained. Prepare data again before training.

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
