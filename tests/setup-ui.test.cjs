const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function app() {
  const elements = new Map();
  const element = selector => {
    if (!elements.has(selector)) elements.set(selector, {
      textContent: '', innerHTML: '', value: '', classList: {add() {}, remove() {}, toggle() {}},
    });
    return elements.get(selector);
  };
  const form = element('#dataset-form');
  form.elements = Object.fromEntries(['train_start', 'train_end', 'validation_start', 'validation_end']
    .map(key => [key, {value: ''}]));
  element('#dataset-model').value = 'load';
  const context = vm.createContext({
    document: {querySelector: element, querySelectorAll: () => [], addEventListener() {},
      createElement: () => ({setAttribute() {}}), body: {append() {}}},
    window: {addEventListener() {}},
    fetch: () => new Promise(() => {}),
    setTimeout() {}, clearTimeout() {},
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../app/static/app.js'), 'utf8'), context);
  return {context, element, form};
}

test('missing saved sensors and history gaps are visible', () => {
  const {context, element} = app();
  vm.runInContext(`setup={roles:[{label:'Live home consumption',entity:'sensor.missing',status:'missing',value:null,unit:'',history:{rows:0}},{label:'Solar bank',entity:'sensor.pv',status:'ok',value:'1',unit:'kW',history:{rows:24,period:'hour',first:'2026-10-09T00:00:00Z',last:'2026-10-10T00:00:00Z'}}]};renderSetup();`, context);
  assert.match(element('#setup-report').innerHTML, /sensor.missing/);
  assert.match(element('#setup-report').innerHTML, /Not found in Home Assistant/);
  assert.match(element('#setup-report').innerHTML, /No recorded readings found/);
});

test('recorded-history suggestions fill dates but preserve a later user edit', () => {
  const {context, element, form} = app();
  vm.runInContext(`setup={suggestions:{load:{train_start:'2026-09-15',train_end:'2026-09-30',validation_start:'2026-10-01',validation_end:'2026-10-10'}}};applySuggestedDates();`, context);
  assert.equal(form.elements.train_start.value, '2026-09-15');
  assert.equal(form.elements.validation_end.value, '2026-10-10');
  form.elements.train_start.value = '2026-09-20';
  vm.runInContext('datesEdited=true;applySuggestedDates();', context);
  assert.equal(form.elements.train_start.value, '2026-09-20');
  assert.match(element('#date-suggestion').textContent, /Suggested from recent recorded history/);
});
