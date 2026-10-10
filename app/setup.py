"""Home Assistant setup checks and history-based dataset date suggestions."""
from datetime import timedelta
import pandas as pd
from .sources import HomeAssistant
from .ha_inputs import numeric_value, history_values

POWER_UNITS = {'W', 'kW'}


def _timestamp(value):
    if isinstance(value, (int, float)):
        return pd.to_datetime(value, unit='ms' if value > 1e11 else 's', utc=True)
    return pd.to_datetime(value, utc=True)


def _coverage(ha, ids, start, end, period):
    if not ids:
        return {}
    try:
        raw = ha.statistics(ids, start, end, period)
    except Exception:
        # Recorder statistics may be disabled; live-state checks can still work.
        raw = {}
    result = {}
    for entity in ids:
        rows = raw.get(entity, [])
        times = [_timestamp(row['start']) for row in rows
                 if row.get('start') is not None and (row.get('mean') is not None or row.get('sum') is not None)]
        source = 'recorder statistics'
        if not times:
            # Some valid power sensors have state history but no long-term statistics.
            source = 'state history'
            try:
                groups = ha.history(entity, start, end)
                pieces = [(_timestamp(row.get('last_updated') or row.get('last_changed')), numeric_value(row))
                          for group in groups for row in group
                          if (row.get('last_updated') or row.get('last_changed'))]
                if pieces:
                    # States hold between changes; unavailable transitions end coverage.
                    samples = history_values(pieces, start, end, '5min' if period == '5minute' else '1h')
                    samples = samples.loc[(samples.index >= start) & (samples.index < end)]
                    times = samples.dropna().index.tolist()
            except Exception:
                source = 'history unavailable'
        result[entity] = {
            'rows': len(times),
            'first': min(times).isoformat() if times else None,
            'last': max(times).isoformat() if times else None,
            'period': period,
            'source': source,
        }
    return result


def _suggestion(ids, coverage, now, timezone, validation_days, max_days):
    histories = [coverage.get(entity) for entity in ids]
    if not histories or any(not item or not item['rows'] for item in histories):
        return {'reason': 'Recorded history is missing for one or more required sensors.'}
    first = max(_timestamp(item['first']) for item in histories)
    last = min(_timestamp(item['last']) for item in histories)
    first_local = first.tz_convert(timezone)
    first_day = first_local.date() + (timedelta(days=1) if first_local.time().isoformat() != '00:00:00' else timedelta())
    last_day = min(last.tz_convert(timezone).date(), now.tz_convert(timezone).date() - timedelta(days=1))
    validation_start = last_day - timedelta(days=validation_days - 1)
    train_start = max(first_day, last_day - timedelta(days=max_days - 1))
    if (validation_start - train_start).days < 7:
        return {'reason': 'At least seven training days before the validation period are needed.'}
    return {
        'train_start': train_start.isoformat(),
        'train_end': (validation_start - timedelta(days=1)).isoformat(),
        'validation_start': validation_start.isoformat(),
        'validation_end': last_day.isoformat(),
    }


def inspect_setup(settings, ha=None, now=None):
    if not settings.ha_token:
        return {'roles': [], 'suggestions': {}, 'message': 'Enter a Home Assistant token to check sensors.'}
    ha = ha or HomeAssistant(settings)
    now = now if now is not None else pd.Timestamp.now(tz='UTC')
    states = {row['entity_id']: row for row in ha.states()}
    specs = [
        ('consumption', 'Live home consumption', settings.consumption_entity, POWER_UNITS, True),
        ('battery_power', 'Battery I/O power', settings.battery_power_entity, POWER_UNITS, True),
        ('battery_soc', 'Battery state of charge', settings.battery_soc_entity, {'%'}, True),
        ('solar_ct', 'Solar monitoring', settings.solar_ct_entity, POWER_UNITS, True),
    ]
    specs.extend((f'history:{i}', 'Consumption history source', entity, POWER_UNITS, False)
                 for i, entity in enumerate(settings.historical_consumption_entities) if entity)
    specs.extend((f'bank:{bank.id}', f'Solar bank: {bank.name}', bank.entity, POWER_UNITS, True)
                 for bank in settings.banks)
    unique = list(dict.fromkeys(entity for _, _, entity, _, _ in specs if entity))
    battery_ids = [entity for entity in (settings.battery_power_entity, settings.battery_soc_entity) if entity]
    hourly_ids = [entity for entity in unique if entity not in battery_ids]
    start = now - pd.Timedelta(days=30)
    coverage = {**_coverage(ha, hourly_ids, start, now, 'hour'),
                **_coverage(ha, list(dict.fromkeys(battery_ids)), start, now, '5minute')}
    roles = []
    for key, label, entity, expected, live in specs:
        state = states.get(entity)
        unit = state.get('attributes', {}).get('unit_of_measurement', '') if state else ''
        value = state.get('state') if state else None
        history = coverage.get(entity, {'rows': 0, 'first': None, 'last': None, 'period': None, 'source': None})
        if not entity or not state:
            status = 'historical_only' if not live and history['rows'] else 'missing'
        elif unit not in expected:
            status = 'wrong_unit'
        elif value in ('unknown', 'unavailable', None, ''):
            status = 'unavailable'
        elif numeric_value(state) is None:
            status = 'invalid_value'
        else:
            status = 'ok'
        roles.append({'key': key, 'label': label, 'entity': entity, 'status': status,
                      'value': value, 'unit': unit, 'history': history})
    suggestions = {}
    load_ids = [entity for entity in settings.historical_consumption_entities if entity]
    # Spliced replacement sensors can contribute different parts of the date range.
    if load_ids:
        available = [coverage[entity] for entity in load_ids if coverage.get(entity, {}).get('rows')]
        if available:
            combined = {'rows': sum(item['rows'] for item in available),
                        'first': min(item['first'] for item in available),
                        'last': max(item['last'] for item in available)}
            suggestions['load'] = _suggestion(['combined'], {'combined': combined}, now, settings.timezone, 10, 30)
    suggestions.setdefault('load', {'reason': 'No consumption history was found in the last 30 days.'})
    for bank in settings.banks:
        suggestions[f'solar:{bank.id}'] = _suggestion([bank.entity], coverage, now, settings.timezone, 10, 30)
    suggestions['battery'] = _suggestion(battery_ids, coverage, now, settings.timezone, 3, 14)
    suggestions['inverter'] = _suggestion([*[bank.entity for bank in settings.banks], settings.solar_ct_entity],
                                         coverage, now, settings.timezone, 3, 14)
    return {'roles': roles, 'suggestions': suggestions, 'window_days': 30,
            'solar_ct_proxy': settings.solar_ct_proxy}
