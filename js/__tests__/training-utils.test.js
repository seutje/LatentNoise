import {
  sanitizeFiniteNumber,
  ensureFloat32Array,
  translateRewardToTargets,
  normalizeInto,
  prepareReinforcementBatch,
} from '../training-utils.js';

describe('training-utils', () => {
  test('translateRewardToTargets blends outputs based on score magnitude', () => {
    const sample = {
      score: 0.5,
      outputs: new Float32Array([0.1, -0.2]),
      targetOutputs: new Float32Array([0.5, 0.2]),
    };
    const { targets, weight } = translateRewardToTargets(sample, 2);
    expect(targets).toBeInstanceOf(Float32Array);
    expect(targets.length).toBe(2);
    expect(targets[0]).toBeCloseTo(0.3);
    expect(targets[1]).toBeCloseTo(0);
    expect(weight).toBeGreaterThan(0);
  });

  test('normalizeInto produces finite values', () => {
    const features = new Float32Array([1, Number.NaN, Infinity]);
    const mean = new Float32Array([0.5, 0, 0]);
    const invStd = new Float32Array([2, 1, 1]);
    const out = normalizeInto(features, mean, invStd);
    expect(out).toBeInstanceOf(Float32Array);
    expect(out.length).toBe(3);
    out.forEach((value) => {
      expect(Number.isFinite(value)).toBe(true);
    });
  });

  test('prepareReinforcementBatch sanitizes samples and generates transfer buffers', () => {
    const sample = {
      score: 1,
      timestamp: 10,
      frameTimestamp: 5,
      features: [0.1, 0.2, 0.3],
      outputs: [0, 0.1],
      targetOutputs: [0.4, -0.2],
    };
    const { samples, transfers, totalWeight } = prepareReinforcementBatch([sample], { inputSize: 3, outputSize: 2 });
    expect(samples.length).toBe(1);
    expect(samples[0].features).toBeInstanceOf(Float32Array);
    expect(samples[0].targets).toBeInstanceOf(Float32Array);
    expect(transfers.length).toBe(2);
    expect(totalWeight).toBeGreaterThan(0);
  });

  test('ensureFloat32Array resizes and sanitizes sources', () => {
    const array = ensureFloat32Array([1, Number.NaN], 3, 0);
    expect(array).toBeInstanceOf(Float32Array);
    expect(array.length).toBe(3);
    expect(array[1]).toBe(0);
  });

  test('sanitizeFiniteNumber falls back for invalid input', () => {
    expect(sanitizeFiniteNumber('foo', 1)).toBe(1);
    expect(sanitizeFiniteNumber(2, 0)).toBe(2);
  });
});
