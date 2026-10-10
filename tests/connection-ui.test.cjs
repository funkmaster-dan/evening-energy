const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function setup({saveFails = false, connectFails = false, token = 'entered-token'} = {}) {
  const elements = new Map();
  const element = selector => {
    if (!elements.has(selector)) elements.set(selector, {textContent: '', classList: {add() {}, remove() {}}});
    return elements.get(selector);
  };
  const form = element('#settings-form');
  form.elements = {ha_url: {value: 'http://entered-ha:8123'}, ha_token: {value: token}, latitude: {value: '-37.8'}};
  const sensor = {value: 'sensor.unsaved', dataset: {context: 'power'}, innerHTML: ''};
  const calls = [];
  const context = vm.createContext({
    document: {
      querySelector: element,
      querySelectorAll: selector => selector.startsWith('#settings-form select') ? [sensor] : [],
      addEventListener() {},
      createElement: () => ({setAttribute() {}}),
      body: {append() {}},
    },
    window: {addEventListener() {}},
    setTimeout() {}, clearTimeout() {},
    fetch: async (url, options = {}) => {
      // Leave the page bootstrap pending; exercise the real connection handler directly.
      if (url === '/api/config' && !options.method) return new Promise(() => {});
      calls.push({url, ...options});
      let ok = true, data;
      if (url === '/api/config') {
        ok = !saveFails;
        data = ok ? {ha_url: 'http://entered-ha:8123', ha_token: '', token_configured: true} : {detail: 'Save failed'};
      } else if (url === '/api/connect') {
        ok = !connectFails;
        data = ok ? {entities: 3} : {detail: 'Connection failed'};
      } else data = [{id: 'sensor.available', name: 'Available sensor', unit: 'W'}];
      return {ok, json: async () => data};
    },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../app/static/app.js'), 'utf8'), context);
  return {element, form, sensor, calls};
}

test('connection saves entered credentials before connecting and preserves unsaved edits', async () => {
  const {element, form, sensor, calls} = setup();
  await element('#connect-ha').onclick();
  assert.equal(calls[0].url, '/api/config');
  assert.equal(calls[0].method, 'PUT');
  assert.deepEqual(JSON.parse(calls[0].body), {ha_url: 'http://entered-ha:8123', ha_token: 'entered-token'});
  assert.equal(calls[1].url, '/api/connect');
  assert.equal(calls[1].method, 'POST');
  assert.equal(calls.length, 5);
  assert.equal(form.elements.ha_token.value, '');
  assert.equal(form.elements.latitude.value, '-37.8');
  assert.match(sensor.innerHTML, /sensor.unsaved/);
  assert.match(sensor.innerHTML, /sensor.available/);
  assert.equal(element('#connection-result').textContent, 'Connected · 3 entities available');
  assert.equal(element('#connect-ha').disabled, false);
});

test('save failure keeps entered token and does not check stale credentials', async () => {
  const {element, form, calls} = setup({saveFails: true});
  await element('#connect-ha').onclick();
  assert.equal(calls.length, 1);
  assert.equal(form.elements.ha_token.value, 'entered-token');
  assert.equal(element('#connection-result').textContent, 'Save failed');
  assert.equal(element('#connect-ha').disabled, false);
});

test('connection failure leaves unrelated edits intact and enables retry', async () => {
  const {element, form, calls} = setup({connectFails: true});
  await element('#connect-ha').onclick();
  assert.equal(calls.length, 2);
  assert.equal(form.elements.latitude.value, '-37.8');
  assert.equal(element('#connection-result').textContent, 'Connection failed');
  assert.equal(element('#connect-ha').disabled, false);
});

test('blank token is submitted so the backend can retain the saved token', async () => {
  const {element, calls} = setup({token: ''});
  await element('#connect-ha').onclick();
  assert.equal(JSON.parse(calls[0].body).ha_token, '');
  assert.equal(calls[1].url, '/api/connect');
});
