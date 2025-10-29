import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';

const VALID_MODEL = {
  input: 1,
  normalization: {
    mean: [0],
    std: [1],
  },
  layers: [
    {
      activation: 'tanh',
      weights: [0.2],
      bias: [0],
    },
  ],
};

let postMessageMock;
let messageHandler;
let listeners;

async function importWorkerModule() {
  jest.resetModules();
  listeners = {};
  postMessageMock = jest.fn();
  global.self = {
    postMessage: postMessageMock,
    addEventListener: jest.fn((type, handler) => {
      listeners[type] = handler;
    }),
    removeEventListener: jest.fn(),
  };
  await import('../workers/train-worker.js');
  messageHandler = listeners.message;
}

describe('train worker reinforce handling', () => {
  beforeEach(async () => {
    await importWorkerModule();
  });

  afterEach(() => {
    postMessageMock?.mockClear();
    messageHandler = undefined;
    listeners = undefined;
    delete global.self;
  });

  test('ignores reinforce messages with no samples', () => {
    const payload = {
      type: 'reinforce',
      payload: {
        reset: true,
        model: VALID_MODEL,
        hyperparameters: {
          learningRate: 0.05,
          epochs: 1,
          batchSize: 1,
          steps: 1,
        },
        options: {
          learningRateDecay: 1,
          minLearningRate: 0,
          gradientClipNorm: 0,
        },
        samples: [],
      },
    };

    postMessageMock.mockClear();
    messageHandler({ data: payload });
    expect(postMessageMock).not.toHaveBeenCalled();
  });

  test('processes reinforcement samples and emits weight deltas', () => {
    const features = new Float32Array([0.15]);
    const targets = new Float32Array([0.35]);
    const payload = {
      type: 'reinforce',
      payload: {
        reset: true,
        model: VALID_MODEL,
        hyperparameters: {
          learningRate: 0.05,
          epochs: 1,
          batchSize: 1,
          steps: 1,
        },
        options: {
          learningRateDecay: 1,
          minLearningRate: 0,
          gradientClipNorm: 0,
        },
        samples: [
          {
            features,
            targets,
            weight: 1,
          },
        ],
      },
    };

    postMessageMock.mockClear();
    messageHandler({ data: payload });

    const weightsDeltaCall = postMessageMock.mock.calls.find(([message]) => message.type === 'weights-delta');
    expect(weightsDeltaCall).toBeDefined();
    const [message, transfers] = weightsDeltaCall;
    expect(message.changed).toBe(true);
    expect(message.stats).toEqual(
      expect.objectContaining({
        samples: 1,
        stepsApplied: 1,
        learningRate: expect.any(Number),
      }),
    );
    expect(Array.isArray(transfers)).toBe(true);
  });
});
