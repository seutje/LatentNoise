import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';

const mockPrepareReinforcementBatch = jest.fn();
const mockAdaptiveOn = jest.fn();

function createModelDefinition() {
  return {
    input: 2,
    normalization: {
      mean: [0, 0],
      std: [1, 1],
    },
    layers: [
      {
        activation: 'relu',
        weights: [0.1, -0.2, 0.3, 0.4],
        bias: [0.05, -0.05],
      },
      {
        activation: 'tanh',
        weights: [0.25, -0.35],
        bias: [0.1],
      },
    ],
  };
}

function createCallbacks() {
  return {
    onStatus: jest.fn(),
    onProgress: jest.fn(),
    onComplete: jest.fn(),
    onCancelled: jest.fn(),
    onError: jest.fn(),
    onWarmup: jest.fn(),
  };
}

describe('training controller adaptive reinforcement', () => {
  let controller;
  let worker;

  beforeEach(() => {
    jest.resetModules();
    mockPrepareReinforcementBatch.mockReset();
    mockAdaptiveOn.mockReset();

    worker = {
      postMessage: jest.fn(),
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      terminate: jest.fn(),
    };

    global.Worker = jest.fn(() => worker);
  });

  afterEach(() => {
    if (controller && typeof controller.destroy === 'function') {
      controller.destroy();
    }
    controller = undefined;
    delete global.Worker;
  });

  async function setupController() {
    jest.unstable_mockModule('../adaptive-feedback.js', () => ({
      __esModule: true,
      on: (...args) => {
        mockAdaptiveOn(...args);
        return () => {};
      },
    }));
    jest.unstable_mockModule('../training-utils.js', () => ({
      __esModule: true,
      prepareReinforcementBatch: mockPrepareReinforcementBatch,
    }));

    const { createController } = await import('../training.js');
    controller = createController(createCallbacks());
    return controller;
  }

  function extractBatchHandler() {
    const call = mockAdaptiveOn.mock.calls.find(([eventName]) => eventName === 'batch');
    return call ? call[1] : undefined;
  }

  test('forwards adaptive batches to the worker with sanitized payloads', async () => {
    const batchedFeatures = new Float32Array([0.25, -0.1]);
    const batchedTargets = new Float32Array([0.42]);
    const batchedSamples = [
      {
        features: batchedFeatures,
        targets: batchedTargets,
        weight: 0.75,
      },
    ];
    const transfers = [batchedFeatures.buffer, batchedTargets.buffer];

    mockPrepareReinforcementBatch.mockReturnValue({ samples: batchedSamples, transfers });

    await setupController();

    const startPromise = controller.start({
      mode: 'adaptive',
      modelDefinition: createModelDefinition(),
      hyperparameters: {
        learningRate: 0.01,
        epochs: 10,
        batchSize: 8,
        l2: 0.001,
      },
      stepsPerBatch: 3,
    });
    startPromise.catch(() => {});
    await Promise.resolve();

    const batchHandler = extractBatchHandler();
    expect(typeof batchHandler).toBe('function');

    const rawSamples = [
      {
        features: new Float32Array([0.1, 0.2]),
        outputs: new Float32Array([0.05]),
        score: 1,
      },
    ];

    batchHandler({ samples: rawSamples });

    expect(mockPrepareReinforcementBatch).toHaveBeenCalledWith(rawSamples, {
      inputSize: 2,
      outputSize: 1,
    });

    expect(worker.postMessage).toHaveBeenCalled();
    const lastCall = worker.postMessage.mock.calls[worker.postMessage.mock.calls.length - 1];
    expect(lastCall[0]).toEqual({
      type: 'reinforce',
      payload: {
        samples: batchedSamples,
        hyperparameters: expect.objectContaining({
          epochs: 10,
          learningRate: expect.any(Number),
          batchSize: 8,
          l2: expect.any(Number),
          steps: 3,
        }),
        options: {
          learningRateDecay: 1,
          minLearningRate: 0,
          gradientClipNorm: 0,
        },
      },
    });
    expect(lastCall[1]).toBe(transfers);
  });

  test('ignores adaptive batches while paused', async () => {
    mockPrepareReinforcementBatch.mockReturnValue({
      samples: [
        {
          features: new Float32Array([0.5, 0.25]),
          targets: new Float32Array([0.1]),
          weight: 1,
        },
      ],
      transfers: [],
    });

    await setupController();

    const startPromise = controller.start({
      mode: 'adaptive',
      modelDefinition: createModelDefinition(),
      hyperparameters: {
        learningRate: 0.02,
        epochs: 5,
        batchSize: 4,
        l2: 0,
      },
    });
    startPromise.catch(() => {});
    await Promise.resolve();

    const batchHandler = extractBatchHandler();
    expect(typeof batchHandler).toBe('function');

    expect(controller.pause()).toBe(true);

    worker.postMessage.mockClear();
    mockPrepareReinforcementBatch.mockClear();

    batchHandler({ samples: [{ features: new Float32Array([0.3, 0.4]) }] });

    expect(mockPrepareReinforcementBatch).not.toHaveBeenCalled();
    expect(worker.postMessage).not.toHaveBeenCalled();
  });
});
