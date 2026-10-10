import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from app.config import Bank, Settings
from app.setup import inspect_setup, _coverage
from app import main


NOW = pd.Timestamp('2026-10-11T02:00:00Z')


class FakeHA:
    def __init__(self, missing=(), wrong_unit=()):
        self.missing = set(missing)
        self.wrong_unit = set(wrong_unit)
        self.history_calls = []

    def states(self):
        return [
            {'entity_id': entity, 'state': '50',
             'attributes': {'unit_of_measurement': 'kWh' if entity in self.wrong_unit else unit}}
            for entity, unit in [('sensor.home', 'kW'), ('sensor.battery', 'kW'),
                                 ('sensor.soc', '%'), ('sensor.solar', 'kW'),
                                 ('sensor.pv', 'W')]
            if entity not in self.missing
        ]

    def statistics(self, entities, start, end, period):
        earliest = NOW - pd.Timedelta(days=30 if period == 'hour' else 13)
        times = pd.date_range(earliest, NOW, freq='1h' if period == 'hour' else '5min')
        return {entity: [{'start': t.timestamp() * 1000, 'mean': 1.0} for t in times]
                for entity in entities if entity not in self.missing}

    def history(self, entity, start, end):
        self.history_calls.append(entity)
        return []


def settings(**overrides):
    return Settings(ha_token='test-token', timezone='Australia/Melbourne',
                    consumption_entity='sensor.home',
                    historical_consumption_entities=['sensor.home'],
                    battery_power_entity='sensor.battery', battery_soc_entity='sensor.soc',
                    solar_ct_entity='sensor.solar',
                    banks=[Bank(id='north', name='North', entity='sensor.pv', azimuth=0)],
                    **overrides)


class SetupTests(unittest.TestCase):
    def test_matching_sensors_report_units_history_and_valid_date_ranges(self):
        report = inspect_setup(settings(), FakeHA(), NOW)
        self.assertTrue(all(role['status'] == 'ok' for role in report['roles']))
        self.assertTrue(all(role['history']['rows'] > 0 for role in report['roles']))
        load = report['suggestions']['load']
        self.assertEqual(load['validation_end'], '2026-10-10')
        self.assertGreaterEqual((pd.Timestamp(load['train_end']) - pd.Timestamp(load['train_start'])).days, 6)
        self.assertEqual((pd.Timestamp(load['validation_end']) - pd.Timestamp(load['validation_start'])).days, 9)
        self.assertEqual(report['suggestions']['solar:north']['validation_end'], '2026-10-10')
        self.assertEqual(report['suggestions']['battery']['validation_end'], '2026-10-10')
        self.assertEqual(report['suggestions']['inverter']['validation_end'], '2026-10-10')

    def test_missing_live_sensor_is_flagged_without_suggesting_unavailable_history(self):
        report = inspect_setup(settings(), FakeHA(missing={'sensor.home'}), NOW)
        self.assertEqual(report['roles'][0]['status'], 'missing')
        self.assertEqual(report['roles'][0]['history']['rows'], 0)
        self.assertIn('reason', report['suggestions']['load'])

    def test_wrong_unit_is_reported_even_if_history_exists(self):
        report = inspect_setup(settings(), FakeHA(wrong_unit={'sensor.home'}), NOW)
        self.assertEqual(report['roles'][0]['status'], 'wrong_unit')
        self.assertEqual(report['roles'][0]['unit'], 'kWh')

    def test_non_numeric_live_reading_is_flagged(self):
        ha = FakeHA()
        original = ha.states
        def with_bad_value():
            states = original()
            states[0]['state'] = 'not-a-number'
            return states
        ha.states = with_bad_value
        report = inspect_setup(settings(), ha, NOW)
        self.assertEqual(report['roles'][0]['status'], 'invalid_value')

    def test_historical_replacement_can_be_absent_from_live_state(self):
        s = settings()
        s.historical_consumption_entities = ['sensor.old', 'sensor.home']
        ha = FakeHA(missing={'sensor.old'})
        original = ha.statistics

        def with_old_history(entities, start, end, period):
            result = original(entities, start, end, period)
            if 'sensor.old' in entities:
                result['sensor.old'] = [{'start': (NOW - pd.Timedelta(days=20)).timestamp() * 1000, 'mean': 2.}]
            return result

        ha.statistics = with_old_history
        report = inspect_setup(s, ha, NOW)
        role = next(item for item in report['roles'] if item['entity'] == 'sensor.old')
        self.assertEqual(role['status'], 'historical_only')
        self.assertIn('train_start', report['suggestions']['load'])

    def test_insufficient_history_explains_why_dates_are_unavailable(self):
        ha = FakeHA()
        original = ha.statistics

        def short_history(entities, start, end, period):
            rows = original(entities, start, end, period)
            return {entity: samples[-24:] for entity, samples in rows.items()}

        ha.statistics = short_history
        report = inspect_setup(settings(), ha, NOW)
        self.assertIn('reason', report['suggestions']['load'])

    def test_history_fallback_when_recorder_statistics_fail(self):
        ha = FakeHA()
        def unavailable(*args):
            raise RuntimeError('Recorder disabled')
        def history(entity, start, end):
            return [[{'state': 'unavailable', 'last_updated': (NOW - pd.Timedelta(days=20)).isoformat()},
                     {'state': '1.2', 'last_updated': (NOW - pd.Timedelta(days=19)).isoformat()}]]
        ha.statistics = unavailable
        ha.history = history
        report = inspect_setup(settings(), ha, NOW)
        self.assertEqual(report['roles'][0]['history']['source'], 'state history')
        self.assertEqual(report['roles'][0]['history']['rows'], 19 * 24)
        self.assertEqual(report['roles'][0]['history']['first'], (NOW - pd.Timedelta(days=19)).isoformat())

    def test_unchanged_history_covers_the_requested_window_and_suggests_dates(self):
        ha = FakeHA()
        ha.statistics = lambda *args: {}
        ha.history = lambda *args: [[{'state': '1.2', 'last_changed': (NOW - pd.Timedelta(days=31)).isoformat()}]]
        report = inspect_setup(settings(), ha, NOW)
        load = report['roles'][0]['history']
        battery = report['roles'][1]['history']
        self.assertEqual(load['rows'], 30 * 24)
        self.assertEqual(load['first'], (NOW - pd.Timedelta(days=30)).isoformat())
        self.assertEqual(load['last'], (NOW - pd.Timedelta(hours=1)).isoformat())
        self.assertEqual(battery['rows'], 30 * 24 * 12)
        self.assertEqual(battery['last'], (NOW - pd.Timedelta(minutes=5)).isoformat())
        self.assertEqual(report['suggestions']['load']['validation_end'], '2026-10-10')

    def test_unavailable_history_is_not_filled_and_limits_coverage(self):
        ha = FakeHA()
        ha.statistics = lambda *args: {}
        start = NOW - pd.Timedelta(days=30)
        ha.history = lambda *args: [[
            {'state': '1.2', 'last_changed': start.isoformat()},
            {'state': 'unavailable', 'last_changed': (start + pd.Timedelta(days=10)).isoformat()},
            {'state': '2.4', 'last_changed': (start + pd.Timedelta(days=20)).isoformat()},
            {'state': 'unknown', 'last_changed': (start + pd.Timedelta(days=25)).isoformat()},
        ]]
        for period, per_hour in [('hour', 1), ('5minute', 12)]:
            with self.subTest(period=period):
                coverage = _coverage(ha, ['sensor.home'], start, NOW, period)['sensor.home']
                self.assertEqual(coverage['rows'], 15 * 24 * per_hour)
                self.assertEqual(coverage['last'], (start + pd.Timedelta(days=25) - pd.Timedelta(hours=1 / per_hour)).isoformat())

    def test_endpoint_returns_report_without_token(self):
        with patch.object(main, 'settings', Settings()):
            client = TestClient(main.app)
            response = client.get('/api/setup')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['roles'], [])
        self.assertIn('token', response.json()['message'])


if __name__ == '__main__':
    unittest.main()
