import { loadModel, normalize, forward, applyWeightDelta } from '../nn.js';

function createDummyModel() {
  return {
    input: 3,
    normalization: {
      mean: [0.1, -0.2, 0.3],
      std: [1.1, 0.8, 1.5],
    },
    layers: [
      {
        activation: 'relu',
        weights: [
          0.5, -0.25, 0.75,
          -0.4, 0.95, 0.2,
        ],
        bias: [0.05, -0.1],
      },
      {
        activation: 'tanh',
        weights: [1.2, -0.6],
        bias: [0.02],
      },
    ],
  };
}

describe('nn runtime', () => {
  test('normalizes features and produces bounded output', async () => {
    await loadModel(createDummyModel());
    const features = new Float32Array([0.9, -0.4, 0.1]);
    const normalized = normalize(features);
    expect(normalized).toBeInstanceOf(Float32Array);
    expect(normalized.length).toBe(3);

    const outputs = forward(normalized);
    expect(outputs).toBeInstanceOf(Float32Array);
    expect(outputs.length).toBe(1);
    expect(Number.isFinite(outputs[0])).toBe(true);
    expect(Math.abs(outputs[0])).toBeLessThanOrEqual(1);
  });

  test('forward writes into provided buffer', async () => {
    await loadModel(createDummyModel());
    const normalized = normalize(new Float32Array([0.1, 0.2, -0.3]));
    const outBuffer = new Float32Array(1);
    const result = forward(normalized, outBuffer);
    expect(result).toBe(outBuffer);
    expect(Number.isFinite(result[0])).toBe(true);
  });
  test('applyWeightDelta mutates loaded model weights', async () => {
    await loadModel(createDummyModel());
    const features = new Float32Array([0.2, -0.1, 0.4]);
    const normalized = normalize(features);
    const before = forward(normalized, new Float32Array(1))[0];
    const delta = {
      layers: [
        {
          weights: new Float32Array([0.01, 0.02, -0.01, 0, 0, 0.03]),
          biases: new Float32Array([0.005, -0.002]),
        },
        {
          weights: new Float32Array([0.015, -0.01]),
          biases: new Float32Array([0.004]),
        },
      ],
    };
    const applied = applyWeightDelta(delta);
    expect(applied).toBe(true);
    const after = forward(normalized, new Float32Array(1))[0];
    expect(after).not.toBeCloseTo(before);
  });

  test('applyWeightDelta returns false for invalid deltas', async () => {
    await loadModel(createDummyModel());
    const applied = applyWeightDelta({ layers: [{ weights: [Infinity], biases: [Number.NaN] }] });
    expect(applied).toBe(false);
  });

});
