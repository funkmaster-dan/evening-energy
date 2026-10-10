import unittest
from unittest.mock import patch

import pandas as pd

from app.config import Settings
from app.datasets import build_dataset
from app.forecast import Forecaster
from app.sources import Calendar, HomeAssistant


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(ha_token='test-token')
        self.ha = HomeAssistant(self.settings)
        self.start = pd.Timestamp('2025-01-01', tz='UTC')
        self.end = self.start + pd.Timedelta(hours=3)
        for method, result in (('states', []), ('statistics', {}), ('history', [])):
            patcher = patch.object(HomeAssistant, method, return_value=result)
            mock = patcher.start()
            setattr(self, method, mock)
            self.addCleanup(patcher.stop)

    def test_empty_history_has_utc_time_index_and_requested_columns(self):
        for entities in (['sensor.missing'], ['sensor.first', 'sensor.second'], []):
            with self.subTest(entities=entities):
                frame = self.ha.frame(entities, self.start, self.end)
                self.assertTrue(frame.empty)
                self.assertIsInstance(frame.index, pd.DatetimeIndex)
                self.assertEqual(str(frame.index.tz), 'UTC')
                self.assertEqual(list(frame.columns), entities)
                grid = pd.date_range(self.start, self.end, freq='1h', inclusive='left')
                self.assertTrue(frame.reindex(grid).isna().all().all())

    def test_statistics_keep_values_and_exclude_end_boundary(self):
        self.states.return_value = [{
            'entity_id': 'sensor.power',
            'attributes': {'unit_of_measurement': 'W'},
        }]
        times = [self.start - pd.Timedelta(hours=1), self.start,
                 self.start + pd.Timedelta(hours=1), self.end]
        self.statistics.return_value = {'sensor.power': [
            {'start': t.timestamp(), 'mean': value}
            for t, value in zip(times, [500, 1000, 2000, 3000])
        ]}
        frame = self.ha.frame(['sensor.power', 'sensor.missing'], self.start, self.end)
        self.assertEqual(frame['sensor.power'].tolist(), [1., 2.])
        self.assertTrue(frame['sensor.missing'].isna().all())
        self.assertEqual(list(frame.index), times[1:3])
        self.assertEqual(str(frame.index.tz), 'UTC')

    def test_rest_history_fallback_keeps_time_index_and_units(self):
        self.states.return_value = [{
            'entity_id': 'sensor.power',
            'attributes': {'unit_of_measurement': 'W'},
        }]
        self.history.return_value = [[
            {'last_changed': self.start.isoformat(), 'state': '1500'},
            {'last_changed': (self.start + pd.Timedelta(hours=1)).isoformat(), 'state': '2500'},
        ]]
        frame = self.ha.frame(['sensor.power'], self.start, self.end)
        self.assertEqual(frame['sensor.power'].tolist(), [1.5, 2.5, 2.5])
        self.assertEqual(str(frame.index.tz), 'UTC')

    def test_empty_dataset_reports_no_recorded_data(self):
        request = {
            'kind': 'load', 'entities': ['sensor.missing'],
            'train_start': '2025-01-01', 'train_end': '2025-01-07',
            'validation_start': '2025-01-08', 'validation_end': '2025-01-18',
        }
        with self.assertRaisesRegex(ValueError, 'Home Assistant has no recorded data'):
            build_dataset(self.settings, request, lambda message: None)

    def test_forecast_without_history_returns_setup_warning_and_zero_export(self):
        def weather(start, end, historical=False):
            return pd.DataFrame({
                'temperature_2m': 20., 'shortwave_radiation': 0.,
                'direct_normal_irradiance': 0., 'diffuse_radiation': 0.,
            }, index=pd.date_range(start, end, freq='1h'))

        self.settings.horizon_hours = 1
        with patch('app.forecast.Weather.fetch', side_effect=weather), \
                patch('app.forecast.forecast_weather', side_effect=lambda settings, start, end, model: weather(start, end)), \
                patch.object(Calendar, 'load', lambda calendar: calendar), \
                patch('app.forecast.model_ready', return_value=False), \
                patch('app.forecast.read_json', side_effect=lambda name, default=None: default), \
                patch('app.forecast.allowance_used', return_value={}), \
                patch('app.forecast_measurement.archive_issue'), \
                patch('app.forecast.write_json'):
            result = Forecaster().run(self.settings, {})
        self.assertFalse(result['ready'])
        self.assertTrue(result['forecast'])
        self.assertTrue(any('No recorded consumption history' in warning for warning in result['warnings']))
        self.assertEqual(result['achievable_export_kwh'], dict.fromkeys(['low', 'middle', 'high'], 0.))


if __name__ == '__main__':
    unittest.main()
