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

  test('respects custom baselines passed to reset', () => {
    const outputs = new Float32Array(getParamNames().length);
    const custom = {
      spawnRate: 0.72,
      glow: 0.8,
      sparkleDensity: 0.18,
    };

    reset(custom);

    let params = {};
    for (let i = 0; i < 3; i += 1) {
      params = update(outputs, { dt: 1 / 60, activity: 1 });
    }

    expect(params.spawnRate).toBeCloseTo(custom.spawnRate, 1e-3);
    expect(params.glow).toBeCloseTo(custom.glow, 1e-3);
    expect(params.sparkleDensity).toBeCloseTo(custom.sparkleDensity, 1e-3);

    params = update(outputs, { dt: 1 / 60, activity: 0 });
    expect(params.spawnRate).toBeLessThan(custom.spawnRate);
    expect(params.spawnRate).toBeGreaterThan(0);
    expect(params.sparkleDensity).toBeLessThanOrEqual(custom.sparkleDensity);
  });

  test('derives activity from RMS features to avoid false silence', () => {
    const outputs = new Float32Array(getParamNames().length);
    outputs.fill(0);

    const features = new Float32Array(24);
    features[5] = 0.25;
    features[20] = 0.25;

    let params = {};
    for (let i = 0; i < 5; i += 1) {
      params = update(outputs, { dt: 1 / 60, features });
    }

    expect(params.spawnRate).toBeGreaterThan(0.3);
  });

  test('treats very low RMS features as silence', () => {
    const outputs = new Float32Array(getParamNames().length);
    outputs.fill(0);

    const features = new Float32Array(24);
    features[5] = 0.005;
    features[20] = 0.005;

    let params = {};
    for (let i = 0; i < 30; i += 1) {
      params = update(outputs, { dt: 1 / 60, features });
    }

    expect(params.spawnRate).toBeLessThan(0.27);
  });
});
