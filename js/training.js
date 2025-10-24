import { loadModelDefinition, createModel, infer } from './nn.js';

const DEFAULT_OPTIONS = Object.freeze({
  learningRateDecay: 0.92,
  minLearningRate: 1e-5,
  gradientClipNorm: 5,
  progressThrottleMs: 120,
});

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
 * @property {string} modelUrl
 * @property {object} hyperparameters
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
    const { dataset, summary, modelUrl, hyperparameters } = options;
    if (!modelUrl || typeof modelUrl !== 'string') {
      throw new Error('Training requires a modelUrl string.');
    }
    if (!dataset || !(dataset.features instanceof Float32Array)) {
      throw new Error('Training dataset missing Float32Array features.');
    }
    if (!dataset.targets || !(dataset.targets instanceof Float32Array)) {
      throw new Error('Training dataset missing Float32Array targets.');
    }

    state.activeDataset = dataset;
    state.activeSummary = summary ?? null;
    state.activeHyper = sanitizeHyperparameters(hyperparameters);
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
      modelDefinition = await loadModelDefinition(modelUrl);
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

    worker.postMessage(
      {
        type: 'train',
        payload: {
          dataset: datasetPayload,
          model: modelDefinition,
          hyperparameters: state.activeHyper,
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
