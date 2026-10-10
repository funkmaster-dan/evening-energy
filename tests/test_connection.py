import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app import config, main
from app.config import Settings, load_settings
from app.sources import HomeAssistant


class ConnectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        for target, value in (
            ('app.config.DATA', Path(temporary.name)),
            ('app.main.settings', Settings()),
            ('app.main.entities', []),
            ('app.main.live', {}),
            ('app.main.forecast', {}),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # Do not start the background scheduler during API tests.
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)

    def test_missing_token_returns_actionable_error_without_network(self):
        with patch('app.sources.httpx.Client') as network:
            response = self.client.post('/api/connect')
        self.assertEqual(response.status_code, 400)
        self.assertIn('long-lived access token', response.json()['detail'])
        self.assertNotIn('Bearer', response.text)
        network.assert_not_called()

    def test_save_then_connect_uses_entered_credentials(self):
        response = self.client.put('/api/config', json={
            'ha_url': ' http://ha.example:8123/ ',
            'ha_token': '  test-token\n',
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['ha_token'], '')
        self.assertTrue(response.json()['token_configured'])
        self.assertEqual(load_settings().ha_token, 'test-token')
        self.assertEqual((config.DATA / 'config.json').stat().st_mode & 0o777, 0o600)

        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json=[{
                'entity_id': 'sensor.example', 'state': '100',
                'attributes': {'unit_of_measurement': 'W'},
            }])

        real_client = httpx.Client
        with patch('app.sources.httpx.Client', side_effect=lambda **kwargs:
                   real_client(transport=httpx.MockTransport(respond), **kwargs)), \
                patch('app.main.store_live'):
            response = self.client.post('/api/connect')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['connected'])
        self.assertEqual(response.json()['entities'], 1)
        self.assertEqual(len(requests), 1)
        self.assertEqual(str(requests[0].url), 'http://ha.example:8123/api/states')
        self.assertEqual(requests[0].headers['Authorization'], 'Bearer test-token')

    def test_blank_or_omitted_token_keeps_saved_token(self):
        self.client.put('/api/config', json={'ha_token': 'saved-token'})
        for body in ({'ha_token': ''}, {'ha_token': ' \n\t '}, {'latitude': -37}):
            with self.subTest(body=body):
                response = self.client.put('/api/config', json=body)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(load_settings().ha_token, 'saved-token')
                self.assertEqual(response.json()['ha_token'], '')
                self.assertTrue(response.json()['token_configured'])

    def test_connection_save_preserves_other_settings(self):
        self.client.put('/api/config', json={'battery_capacity_kwh': 27})
        self.client.put('/api/config', json={'ha_token': 'first-token'})
        response = self.client.put('/api/config', json={'ha_token': 'replacement-token'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(load_settings().ha_token, 'replacement-token')
        self.assertEqual(load_settings().battery_capacity_kwh, 27)

    def test_whitespace_only_token_is_missing(self):
        settings = Settings(ha_token=' \n\t ')
        self.assertFalse(settings.public()['token_configured'])
        with self.assertRaisesRegex(ValueError, 'long-lived access token'):
            HomeAssistant(settings)


if __name__ == '__main__':
    unittest.main()
