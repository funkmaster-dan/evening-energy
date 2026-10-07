# Optional export automation

Evening Energy only reports forecasts. Battery control belongs in Home Assistant and requires a controller compatible with your inverter.

[Energy Excess](https://github.com/funkmaster-dan/ha-energy-excess) exposes stored battery excess and achievable peak export as separate sensors. Install it through HACS using the forecasting app's URL.

For supported AlphaESS systems, the separate [AlphaESS Portal Export](https://github.com/funkmaster-dan/ha-alphaess-portal) integration provides a force-discharge switch and power, target SoC and duration controls. It creates no telemetry sensors.

## Example: 6–9 pm export when excess is positive

[portal_middle_forecast_automation.yaml](portal_middle_forecast_automation.yaml) is a minimal example:

- Turn on between 18:00 and 21:00 when middle stored excess is above 0 kWh.
- Turn off when excess reaches zero or at 21:00.
- Resume if excess becomes positive again within the window.

Set the controller's discharge duration to **180 minutes** and choose power/SoC settings for your own battery. Update the entity IDs and times in the example to match your installation. It avoids repeatedly starting an already-on switch.

The example uses **stored excess**, not the smaller achievable-export allowance. The app itself never sends a battery command.
