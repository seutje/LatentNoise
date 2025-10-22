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

    expect(params.spawnOffset).toBeCloseTo(0, 1e-5);
    expect(params.glowOffset).toBeCloseTo(0, 1e-5);
    expect(params.sparkleOffset).toBeCloseTo(0, 1e-5);

    params = update(outputs, { dt: 1 / 60, activity: 0 });
    expect(params.spawnRate).toBeLessThan(custom.spawnRate);
    expect(params.spawnRate).toBeGreaterThanOrEqual(0);
    expect(params.sparkleDensity).toBeLessThanOrEqual(custom.sparkleDensity);
  });

  test('reports NN-driven offsets alongside absolute parameters', () => {
    const outputs = new Float32Array(getParamNames().length);
    const spawnSpec = getParamSpec('spawnRate');
    const glowSpec = getParamSpec('glow');
    const sparkleSpec = getParamSpec('sparkleDensity');
    const hueSpec = getParamSpec('hueShift');
    const repelSpec = getParamSpec('repelImpulse');

    outputs[spawnSpec.index] = 1;
    outputs[glowSpec.index] = -1;
    outputs[sparkleSpec.index] = 1;
    outputs[hueSpec.index] = 0.5;
    outputs[repelSpec.index] = 1;

    let params = {};
    for (let i = 0; i < 6; i += 1) {
      params = update(outputs, { dt: 1 / 60, activity: 1 });
    }

    expect(params.spawnOffset).toBeCloseTo(params.spawnRate - spawnSpec.baseline, 1e-3);
    expect(params.glowOffset).toBeCloseTo(params.glow - glowSpec.baseline, 1e-3);
    expect(params.sparkleOffset).toBeCloseTo(params.sparkleDensity - sparkleSpec.baseline, 1e-3);
    expect(params.hueOffset).toBeCloseTo(params.hueShift - hueSpec.baseline, 1e-3);
    expect(params.spawnOffset).toBeGreaterThan(0);
    expect(params.glowOffset).toBeLessThan(0);
    expect(params.sparkleOffset).toBeGreaterThan(0);
    expect(params.hueOffset).not.toBe(0);
    expect(params.repelImpulse).toBeGreaterThan(0);
  });

  test('forceSilence drops spawn rate immediately', () => {
    const outputs = new Float32Array(getParamNames().length);
    outputs.fill(0.5);

    let params = update(outputs, { dt: 1 / 60, activity: 1 });
    expect(params.spawnRate).toBeGreaterThan(0.3);

    params = update(outputs, { dt: 1 / 60, activity: 0, forceSilence: true });
    expect(params.spawnRate).toBe(0);
  });
});
