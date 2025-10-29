import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import * as adaptiveFeedback from '../adaptive-feedback.js';

function mockNow(value) {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return jest.spyOn(performance, 'now').mockReturnValue(value);
  }
  global.performance = {
    now: jest.fn().mockReturnValue(value),
  };
  return global.performance.now;
}

describe('adaptive-feedback module', () => {
  let nowSpy;

  beforeEach(() => {
    adaptiveFeedback.clear();
    adaptiveFeedback.setDefaultLatency(100);
    nowSpy = mockNow(10_000);
  });

  afterEach(() => {
    if (nowSpy && typeof nowSpy.mockRestore === 'function') {
      nowSpy.mockRestore();
    }
    jest.restoreAllMocks();
  });

  test('aligns feedback samples to the closest buffered frame', () => {
    adaptiveFeedback.recordFrame({
      timestamp: 9_000,
      features: new Float32Array([0.1, 0.2]),
      outputs: new Float32Array([0.05, -0.2]),
      mappedParams: {
        spawnRate: 0.4,
        glow: 0.5,
      },
    });
    adaptiveFeedback.recordFrame({
      timestamp: 11_000,
      features: new Float32Array([0.3, 0.4]),
      outputs: new Float32Array([0.25, 0.1]),
      mappedParams: {
        spawnRate: 0.45,
        glow: 0.55,
      },
    });

    const sample = adaptiveFeedback.recordFeedback(1, { timestamp: 12_000 });
    expect(sample).not.toBeNull();
    expect(sample.frameTimestamp).toBe(11_000);
    expect(sample.features[0]).toBeCloseTo(0.3, 5);
    expect(sample.features[1]).toBeCloseTo(0.4, 5);
    expect(sample.outputs[0]).toBeCloseTo(0.25, 5);
    expect(sample.outputs[1]).toBeCloseTo(0.1, 5);
    expect(sample.direction).toBe(1);
    expect(sample.targetOutputs[0]).toBeCloseTo(0.37, 5);
    expect(sample.targetParams.spawnRate).toBeCloseTo(0.558, 3);
    expect(sample.targetParams.glow).toBe(0.6); // clamped at safe max
  });

  test('respects explicit latency override for negative feedback', () => {
    adaptiveFeedback.recordFrame({
      timestamp: 2_000,
      features: new Float32Array([1, 2, 3]),
      outputs: new Float32Array([0.6, 0.4, 0.2]),
      mappedParams: {
        spawnRate: 0.6,
        glow: 0.42,
      },
    });
    adaptiveFeedback.recordFrame({
      timestamp: 2_320,
      features: new Float32Array([4, 5, 6]),
      outputs: new Float32Array([0.5, 0.25, 0.1]),
      mappedParams: {
        spawnRate: 0.62,
        glow: 0.44,
      },
    });

    const sample = adaptiveFeedback.recordFeedback(-1, {
      timestamp: 2_360,
      latencyMs: 20,
      metadata: { source: 'hud' },
    });

    expect(sample).not.toBeNull();
    expect(sample.direction).toBe(-1);
    expect(sample.frameTimestamp).toBe(2_320);
    expect(sample.latencyMs).toBe(20);
    expect(sample.targetOutputs[0]).toBeCloseTo(0.38, 5);
    expect(sample.targetOutputs[1]).toBeCloseTo(0.13, 5);
    expect(sample.targetOutputs[2]).toBeCloseTo(-0.02, 5);
    expect(sample.metadata).toEqual({ source: 'hud' });
  });

  test('emits batched payloads after reaching the batch size', () => {
    const batches = [];
    const unsubscribe = adaptiveFeedback.on('batch', (detail) => {
      batches.push(detail);
    });

    adaptiveFeedback.setDefaultLatency(0);

    for (let i = 0; i < 8; i += 1) {
      const timestamp = 5_000 + i * 16;
      adaptiveFeedback.recordFrame({
        timestamp,
        features: new Float32Array([i]),
        outputs: new Float32Array([i * 0.1]),
        mappedParams: {
          spawnRate: 0.4 + i * 0.01,
          glow: 0.5,
        },
      });
      adaptiveFeedback.recordFeedback(1, {
        timestamp,
        latencyMs: 0,
      });
    }

    expect(batches).toHaveLength(1);
    expect(batches[0].samples).toHaveLength(8);
    unsubscribe();
  });
});
