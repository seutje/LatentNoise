import { loadModelDefinition, createModel, infer } from './nn.js';
import { isFreshModelId } from './byom-constants.js';

const DEFAULT_OPTIONS = Object.freeze({
  learningRateDecay: 0.92,
  minLearningRate: 1e-5,
  gradientClipNorm: 5,
  progressThrottleMs: 120,
});

const FRESH_MODEL_HIDDEN_SIZE = 16;
const MIN_STD = 1e-6;

const TRAINING_STATUS = Object.freeze({
  IDLE: 'idle',
  PREPARING: 'preparing',
  RUNNING: 'running',
  PAUSED: 'paused',
  CANCELLING: 'cancelling',
  COMPLETED: 'completed',
  CANCELLED: 'cancelled',
  ERROR: 'error',
});

function defaultNow() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function clampNumber(value, min, max, fallback) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  if (numeric < min) {
    return min;
  }
  if (numeric > max) {
    return max;
  }
  return numeric;
}

function sanitizeHyperparameters(raw) {
  const epochs = clampNumber(raw?.epochs, 1, 500, 40);
  const learningRate = clampNumber(raw?.learningRate, 1e-5, 0.1, 0.001);
  const batchSize = clampNumber(raw?.batchSize, 8, 2048, 256);
  const l2 = clampNumber(raw?.l2 ?? 0, 0, 0.05, 0);
  return {
    epochs,
    learningRate,
    batchSize,
    l2,
  };
}

function sanitizeFiniteNumber(value, fallback = 0) {
  return Number.isFinite(value) ? value : fallback;
}

function computeDatasetNormalization(dataset) {
  const featureSize = Number(dataset?.featureSize);
  const frameCount = Number(dataset?.frameCount);
  const features = dataset?.features;
  const size = Number.isInteger(featureSize) && featureSize > 0 ? featureSize : 0;
  const mean = new Float32Array(size);
  const std = new Float32Array(size);
  if (!(features instanceof Float32Array) || size === 0 || frameCount <= 0) {
    for (let i = 0; i < size; i += 1) {
      std[i] = 1;
    }
    return { mean, std };
  }
  const count = Math.max(1, frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    const offset = frame * featureSize;
    for (let i = 0; i < featureSize; i += 1) {
      mean[i] += sanitizeFiniteNumber(features[offset + i]);
    }
  }
  for (let i = 0; i < featureSize; i += 1) {
    mean[i] /= frameCount > 0 ? frameCount : 1;
  }
  for (let frame = 0; frame < frameCount; frame += 1) {
    const offset = frame * featureSize;
    for (let i = 0; i < featureSize; i += 1) {
      const value = sanitizeFiniteNumber(features[offset + i]);
      const diff = value - mean[i];
      std[i] += diff * diff;
    }
  }
  for (let i = 0; i < featureSize; i += 1) {
    const variance = std[i] / count;
    const sigma = Math.sqrt(variance);
    std[i] = Number.isFinite(sigma) && sigma > MIN_STD ? sigma : 1;
  }
  return { mean, std };
}

function randomSigned() {
  return Math.random() * 2 - 1;
}

function createFreshModelDefinition(dataset) {
  const inputSize = Number(dataset?.featureSize);
  if (!Number.isFinite(inputSize) || inputSize <= 0) {
    throw new Error('Fresh training requires a dataset with a positive featureSize.');
  }
  const outputSize = Number(dataset?.targetSize);
  if (!Number.isFinite(outputSize) || outputSize <= 0) {
    throw new Error('Fresh training requires a dataset with a positive targetSize.');
  }
  const hiddenSize = FRESH_MODEL_HIDDEN_SIZE;
  const normalization = computeDatasetNormalization(dataset);
  const layer1Weights = new Float32Array(hiddenSize * inputSize);
  const layer1Bias = new Float32Array(hiddenSize);
  const layer2Weights = new Float32Array(outputSize * hiddenSize);
  const layer2Bias = new Float32Array(outputSize);

  const scale1 = Math.sqrt(2 / Math.max(inputSize, 1));
  const scale2 = Math.sqrt(2 / Math.max(hiddenSize, 1));

  for (let i = 0; i < layer1Weights.length; i += 1) {
    layer1Weights[i] = Math.fround(randomSigned() * scale1);
  }
  for (let i = 0; i < layer2Weights.length; i += 1) {
    layer2Weights[i] = Math.fround(randomSigned() * scale2);
  }

  return {
    input: inputSize,
    normalization,
    layers: [
      {
        activation: 'relu',
        weights: layer1Weights,
        bias: layer1Bias,
      },
      {
        activation: 'tanh',
        weights: layer2Weights,
        bias: layer2Bias,
      },
    ],
  };
}

function createWorker() {
  const url = new URL('./workers/train-worker.js', import.meta.url);
  return new Worker(url, { type: 'module' });
}

function ensureCallbacks(callbacks = {}) {
  return {
    onStatus: typeof callbacks.onStatus === 'function' ? callbacks.onStatus : () => {},
    onProgress: typeof callbacks.onProgress === 'function' ? callbacks.onProgress : () => {},
    onComplete: typeof callbacks.onComplete === 'function' ? callbacks.onComplete : () => {},
    onCancelled: typeof callbacks.onCancelled === 'function' ? callbacks.onCancelled : () => {},
    onError: typeof callbacks.onError === 'function' ? callbacks.onError : () => {},
    onWarmup: typeof callbacks.onWarmup === 'function' ? callbacks.onWarmup : () => {},
  };
}

/**
 * @typedef {object} TrainingStartOptions
 * @property {import('./byom-intake.js').AnalyzeResult['dataset']} dataset
 * @property {import('./byom-intake.js').AnalyzeResult['summary']} summary
 * @property {string} [modelUrl]
 * @property {object} [modelDefinition]
 * @property {object} hyperparameters
 * @property {string} [mode]
 */

export function createController(callbacks) {
  const cb = ensureCallbacks(callbacks);
  const state = {
    status: TRAINING_STATUS.IDLE,
    worker: null,
    lastProgressAt: 0,
    pending: null,
    activeDataset: null,
    activeSummary: null,
    activeHyper: null,
    activeCorrelations: [],
    warmupSample: null,
    cancelRequested: false,
    options: {
      ...DEFAULT_OPTIONS,
    },
  };

  function updateStatus(nextStatus, detail) {
    state.status = nextStatus;
    cb.onStatus({
      status: nextStatus,
      detail,
    });
  }

  function getWorker() {
    if (!state.worker) {
      state.worker = createWorker();
      state.worker.addEventListener('message', handleWorkerMessage);
      state.worker.addEventListener('error', handleWorkerError);
    }
    return state.worker;
  }

  function resetWorker() {
    if (state.worker) {
      state.worker.removeEventListener('message', handleWorkerMessage);
      state.worker.removeEventListener('error', handleWorkerError);
      state.worker.terminate();
      state.worker = null;
    }
  }

  function resolvePending(value) {
    if (state.pending) {
      state.pending.resolve(value);
      state.pending = null;
    }
  }

  function rejectPending(error) {
    if (state.pending) {
      state.pending.reject(error);
      state.pending = null;
    }
  }

  async function start(options) {
    if (state.status !== TRAINING_STATUS.IDLE && state.status !== TRAINING_STATUS.COMPLETED) {
      throw new Error('Training already in progress.');
    }
    if (!options || typeof options !== 'object') {
      throw new TypeError('Training options must be an object.');
    }
    const {
      dataset,
      summary,
      modelUrl,
      modelDefinition: providedModelDefinition,
      hyperparameters,
      correlations,
      mode,
    } = options;
    const inlineDefinition =
      providedModelDefinition && typeof providedModelDefinition === 'object'
        ? providedModelDefinition
        : null;
    const requestedMode = typeof mode === 'string' ? mode : null;
    const freshRequested = requestedMode === 'fresh' || isFreshModelId(modelUrl);
    if (!freshRequested && !inlineDefinition) {
      if (typeof modelUrl !== 'string' || modelUrl.length === 0) {
        throw new Error('Training requires a modelUrl string when tuning an existing model.');
      }
    }
    if (!dataset || !(dataset.features instanceof Float32Array)) {
      throw new Error('Training dataset missing Float32Array features.');
    }
    if (!dataset.targets || !(dataset.targets instanceof Float32Array)) {
      throw new Error('Training dataset missing Float32Array targets.');
    }
    if (!Array.isArray(correlations) || correlations.length === 0) {
      throw new Error('Training requires at least one correlation.');
    }

    state.activeDataset = dataset;
    state.activeSummary = summary ?? null;
    state.activeHyper = sanitizeHyperparameters(hyperparameters);
    state.activeCorrelations = correlations.map((correlation) => ({ ...correlation }));
    state.warmupSample = dataset.features.slice(0, dataset.featureSize);
    state.cancelRequested = false;
    state.lastProgressAt = 0;

    const startTime = defaultNow();
    const promise = new Promise((resolve, reject) => {
      state.pending = { resolve, reject };
    });

    updateStatus(TRAINING_STATUS.PREPARING, { summary, hyperparameters: state.activeHyper });

    let modelDefinition;
    try {
      if (inlineDefinition) {
        modelDefinition = inlineDefinition;
      } else if (freshRequested) {
        modelDefinition = createFreshModelDefinition(dataset);
      } else {
        modelDefinition = await loadModelDefinition(modelUrl);
      }
    } catch (error) {
      state.status = TRAINING_STATUS.ERROR;
      rejectPending(error);
      cb.onError(error);
      cb.onStatus({ status: TRAINING_STATUS.ERROR, detail: error });
      return promise;
    }

    if (state.cancelRequested) {
      state.status = TRAINING_STATUS.CANCELLED;
      resolvePending({ cancelled: true, detail: { reason: 'cancelled-before-start' } });
      cb.onCancelled({ reason: 'cancelled-before-start' });
      return promise;
    }

    const worker = getWorker();
    const clonedFeatures = dataset.features.slice();
    const clonedTargets = dataset.targets.slice();
    const datasetPayload = {
      features: clonedFeatures,
      targets: clonedTargets,
      featureSize: dataset.featureSize,
      targetSize: dataset.targetSize,
      frameCount: dataset.frameCount,
      hopSamples: dataset.hopSamples,
      sampleRate: dataset.sampleRate,
      frameSize: dataset.frameSize,
      splits: {
        train: {
          start: Number(dataset.splits?.train?.start ?? 0),
          count: Number(dataset.splits?.train?.count ?? 0),
        },
        validation: {
          start: Number(dataset.splits?.validation?.start ?? 0),
          count: Number(dataset.splits?.validation?.count ?? 0),
        },
      },
      metadata: dataset.metadata ? { ...dataset.metadata } : null,
    };

    const correlationPayload = state.activeCorrelations.map((correlation) => ({ ...correlation }));

    worker.postMessage(
      {
        type: 'train',
        payload: {
          dataset: datasetPayload,
          model: modelDefinition,
          hyperparameters: state.activeHyper,
          correlations: correlationPayload,
          options: {
            learningRateDecay: state.options.learningRateDecay,
            minLearningRate: state.options.minLearningRate,
            gradientClipNorm: state.options.gradientClipNorm,
            startedAt: startTime,
          },
        },
      },
      [clonedFeatures.buffer, clonedTargets.buffer],
    );
    updateStatus(TRAINING_STATUS.RUNNING, { startedAt: startTime });
    return promise;
  }

  function pause() {
    if (state.status !== TRAINING_STATUS.RUNNING) {
      return false;
    }
    const worker = state.worker;
    if (!worker) {
      return false;
    }
    worker.postMessage({ type: 'pause' });
    updateStatus(TRAINING_STATUS.PAUSED);
    return true;
  }

  function resume() {
    if (state.status !== TRAINING_STATUS.PAUSED) {
      return false;
    }
    const worker = state.worker;
    if (!worker) {
      return false;
    }
    worker.postMessage({ type: 'resume' });
    updateStatus(TRAINING_STATUS.RUNNING);
    return true;
  }

  function cancel() {
    if (state.status === TRAINING_STATUS.PREPARING) {
      state.cancelRequested = true;
      updateStatus(TRAINING_STATUS.CANCELLING);
      return true;
    }
    if (state.status !== TRAINING_STATUS.RUNNING && state.status !== TRAINING_STATUS.PAUSED) {
      return false;
    }
    state.cancelRequested = true;
    const worker = state.worker;
    if (worker) {
      worker.postMessage({ type: 'cancel' });
    }
    updateStatus(TRAINING_STATUS.CANCELLING);
    return true;
  }

  function destroy() {
    resetWorker();
    state.status = TRAINING_STATUS.IDLE;
    state.pending = null;
    state.activeDataset = null;
    state.activeSummary = null;
    state.activeHyper = null;
    state.activeCorrelations = [];
    state.warmupSample = null;
  }

  async function runWarmup(modelDefinition, stats) {
    if (!state.activeDataset || !state.warmupSample) {
      return null;
    }
    let warmupResult = null;
    try {
      const model = createModel(modelDefinition);
      const outputs = infer(model, state.warmupSample, new Float32Array(model.outputSize));
      const clamped = new Float32Array(outputs.length);
      for (let i = 0; i < outputs.length; i += 1) {
        const value = outputs[i];
        if (!Number.isFinite(value)) {
          clamped[i] = 0;
        } else if (value > 1) {
          clamped[i] = 1;
        } else if (value < -1) {
          clamped[i] = -1;
        } else {
          clamped[i] = value;
        }
      }
      warmupResult = {
        outputs: clamped,
        stats,
      };
      cb.onWarmup(warmupResult);
    } catch (error) {
      console.error('[training] Warm-up inference failed', error);
    }
    return warmupResult;
  }

  function handleWorkerMessage(event) {
    const message = event.data;
    if (!message || typeof message !== 'object') {
      return;
    }
    switch (message.type) {
      case 'status': {
        const { state: workerState, detail } = message;
        if (workerState === 'paused') {
          state.status = TRAINING_STATUS.PAUSED;
        } else if (workerState === 'running') {
          state.status = TRAINING_STATUS.RUNNING;
        } else if (workerState === 'cancelled') {
          state.status = TRAINING_STATUS.CANCELLED;
          state.cancelRequested = false;
          resolvePending({ cancelled: true, detail });
          cb.onCancelled(detail);
        } else if (workerState === 'error') {
          state.status = TRAINING_STATUS.ERROR;
          state.cancelRequested = false;
          const error = detail instanceof Error ? detail : new Error(detail?.message ?? 'Training worker error');
          rejectPending(error);
          cb.onError(error);
        }
        cb.onStatus({ status: state.status, detail });
        break;
      }
      case 'progress': {
        const now = defaultNow();
        if (now - state.lastProgressAt >= state.options.progressThrottleMs) {
          state.lastProgressAt = now;
          cb.onProgress(message.payload ?? message);
        }
        break;
      }
      case 'result': {
        state.status = TRAINING_STATUS.COMPLETED;
        state.cancelRequested = false;
        resolvePending({ cancelled: false, result: message.result, stats: message.stats });
        cb.onStatus({ status: TRAINING_STATUS.COMPLETED, detail: message.stats });
        runWarmup(message.result.model, message.stats).then((warmup) => {
          cb.onComplete({
            modelDefinition: message.result.model,
            stats: message.stats,
            warmup,
          });
        });
        break;
      }
      case 'error': {
        state.status = TRAINING_STATUS.ERROR;
        const error = message.error instanceof Error ? message.error : new Error(message.error?.message ?? 'Training failed');
        rejectPending(error);
        cb.onError(error);
        break;
      }
      default:
        break;
    }
  }

  function handleWorkerError(event) {
    const error = event?.error ?? new Error('Training worker crashed.');
    state.status = TRAINING_STATUS.ERROR;
    rejectPending(error);
    cb.onError(error);
    resetWorker();
  }

  function getState() {
    return {
      status: state.status,
      hyperparameters: state.activeHyper,
      summary: state.activeSummary,
      correlations: state.activeCorrelations.slice(),
    };
  }

  return {
    start,
    pause,
    resume,
    cancel,
    destroy,
    getState,
    status: () => state.status,
  };
}
