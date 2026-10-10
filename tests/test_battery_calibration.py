import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.config import Settings
from app.models import train


class BatteryCalibrationTests(unittest.TestCase):
    def test_rounded_soc_retains_energy_between_percentage_changes(self):
        settings = Settings(
            battery_capacity_kwh=49,
            battery_power_entity='sensor.battery_kw',
            battery_soc_entity='sensor.soc',
        )
        index = pd.date_range('2026-09-01', periods=240, freq='5min', tz='UTC')
        power = np.array([-5. if i % 60 < 30 else 5. for i in range(len(index))])
        true_soc = 40.
        observed_soc = []
        for value in power:
            true_soc += ((-value * .94) if value < 0 else (-value / .94)) / 12 / 49 * 100
            observed_soc.append(round(true_soc / 2) * 2)
        frame = pd.DataFrame({'sensor.battery_kw': power, 'sensor.soc': observed_soc}, index=index)
        self.assertGreater(int((frame['sensor.soc'].diff() == 0).sum()), 100)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.to_pickle({'frame': frame, 'train_end': index[120], 'val_start': index[120]}, root / 'battery.pkl')
            summary = {'sufficient': True, 'kind': 'battery', 'training_rows': 120, 'validation_rows': 120,
                       'created_at': '2026-09-01T00:00:00Z'}
            with patch('app.models.DATA', root), \
                    patch('app.models.read_json', return_value=summary), \
                    patch('app.models.write_json'):
                result = train(settings, 'battery', lambda message: None)

        self.assertAlmostEqual(result['charge_efficiency'], .94, delta=.1)
        self.assertAlmostEqual(result['discharge_efficiency'], .94, delta=.1)
        self.assertGreater(result['charge_samples'], 40)
        self.assertGreater(result['discharge_samples'], 40)


if __name__ == '__main__':
    unittest.main()
