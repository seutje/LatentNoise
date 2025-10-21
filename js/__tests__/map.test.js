import { configure, reset, update, getParamNames, getParamSpec } from '../map.js';

describe('map module', () => {
  beforeEach(() => {
    configure({ safeMode: false, silenceThreshold: 0.03 });
    reset();
  });

  afterEach(() => {
    configure({ safeMode: false, silenceThreshold: 0.03 });
    reset();
  });

  test('produces finite baseline parameters with zero outputs', () => {
    const outputs = new Float32Array(getParamNames().length);
    const params = update(outputs, { dt: 1 / 60, activity: 0 });
    for (const name of getParamNames()) {
      expect(Number.isFinite(params[name])).toBe(true);
    }
  });

  test('safe mode enforces safe maxima', () => {
    configure({ safeMode: true });
    reset();
    const outputs = new Float32Array(getParamNames().length);
    outputs.fill(1);

    let params = {};
    for (let i = 0; i < 24; i += 1) {
      params = update(outputs, { dt: 1 / 30, activity: 1 });
    }

    for (const name of getParamNames()) {
      const spec = getParamSpec(name);
      if (!spec) {
        continue;
      }
      if (typeof spec.safeMax === 'number') {
        expect(params[name]).toBeLessThanOrEqual(spec.safeMax + 1e-3);
        if (spec.symmetric) {
          expect(params[name]).toBeGreaterThanOrEqual(-spec.safeMax - 1e-3);
        }
      }
    }
  });
});
