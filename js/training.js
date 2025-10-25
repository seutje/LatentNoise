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

function sanitizeFiniteNumber(value, fallback = 0) {
  return Number.isFinite(value) ? value : fallback;
}

function projectCorrelationTarget(rawValue, featureType, orientationSign) {
  const sign = orientationSign === -1 ? -1 : 1;
  const safeValue = sanitizeFiniteNumber(rawValue, 0);
  if (featureType === 'signed') {
    const clamped = Math.max(-1, Math.min(1, safeValue));
    return clamped * sign;
  }
  const positive = Math.max(0, Math.min(1, safeValue));
  const centered = positive * 2 - 1;
  return centered * sign;
}

function sanitizeCorrelations(raw, dataset) {
  if (!Array.isArray(raw) || raw.length === 0 || !dataset) {
    return [];
  }
  const featureSize = Number(dataset.featureSize);
  const targetSize = Number(dataset.targetSize);
  if (!Number.isInteger(featureSize) || featureSize <= 0 || !Number.isInteger(targetSize) || targetSize <= 0) {
    return [];
  }
  const seen = new Set();
  const sanitized = [];
  raw.forEach((entry) => {
    if (!entry) {
      return;
    }
    const featureIndex = Number(entry.featureIndex);
    const outputIndex = Number(entry.outputIndex);
    if (!Number.isInteger(featureIndex) || featureIndex < 0 || featureIndex >= featureSize) {
      return;
    }
    if (!Number.isInteger(outputIndex) || outputIndex < 0 || outputIndex >= targetSize) {
      return;
    }
    const orientationSign = entry.orientationSign === -1 ? -1 : 1;
    const id =
      typeof entry.id === 'string' && entry.id.length > 0
        ? entry.id
        : `${featureIndex}:${outputIndex}:${orientationSign < 0 ? 'inv' : 'dir'}`;
    if (seen.has(id)) {
      return;
    }
    seen.add(id);
    const featureType = typeof entry.featureType === 'string' ? entry.featureType.toLowerCase() : 'positive';
    sanitized.push({
      id,
      featureIndex,
      featureName: typeof entry.featureName === 'string' ? entry.featureName : '',
      featureType: featureType === 'signed' ? 'signed' : 'positive',
      outputIndex,
      outputName: typeof entry.outputName === 'string' ? entry.outputName : '',
      orientationSign,
    });
  });
  return sanitized;
}

function applyCorrelationTargets(targets, features, dataset, correlations) {
  if (!Array.isArray(correlations) || correlations.length === 0) {
    return;
  }
  const featureSize = Number(dataset.featureSize);
  const targetSize = Number(dataset.targetSize);
  const frameCount = Number(dataset.frameCount);
  if (!Number.isInteger(featureSize) || featureSize <= 0 || !Number.isInteger(targetSize) || targetSize <= 0) {
    return;
  }
  for (let frame = 0; frame < frameCount; frame += 1) {
    const featureOffset = frame * featureSize;
    const targetOffset = frame * targetSize;
    correlations.forEach((correlation) => {
      const rawValue = features[featureOffset + correlation.featureIndex];
      targets[targetOffset + correlation.outputIndex] = projectCorrelationTarget(
        rawValue,
        correlation.featureType,
        correlation.orientationSign,
      );
    });
  }
}

function evaluateCorrelationMetrics(modelDefinition, dataset, correlations) {
  if (!dataset || !Array.isArray(correlations) || correlations.length === 0) {
    return [];
  }
  let model;
  try {
    model = createModel(modelDefinition);
  } catch (error) {
    console.error('[training] Failed to build model for correlation evaluation', error);
    return [];
  }
  const featureSize = Number(dataset.featureSize);
  const frameCount = Number(dataset.frameCount);
  if (!Number.isInteger(featureSize) || featureSize <= 0 || !Number.isInteger(frameCount) || frameCount <= 0) {
    return [];
  }
  const featureBuffer = new Float32Array(featureSize);
  const outputBuffer = new Float32Array(model.outputSize);
  const aggregates = correlations.map(() => ({
    sumFeature: 0,
    sumOutput: 0,
    sumFeatureSq: 0,
    sumOutputSq: 0,
    sumFeatureOutput: 0,
    count: 0,
  }));

  for (let frame = 0; frame < frameCount; frame += 1) {
    const featureOffset = frame * featureSize;
    featureBuffer.set(dataset.features.subarray(featureOffset, featureOffset + featureSize));
    const outputs = infer(model, featureBuffer, outputBuffer);
    correlations.forEach((correlation, index) => {
      const featureValue = sanitizeFiniteNumber(featureBuffer[correlation.featureIndex], 0);
      const outputValue = sanitizeFiniteNumber(outputs[correlation.outputIndex], 0);
      const agg = aggregates[index];
      agg.sumFeature += featureValue;
      agg.sumOutput += outputValue;
      agg.sumFeatureSq += featureValue * featureValue;
      agg.sumOutputSq += outputValue * outputValue;
      agg.sumFeatureOutput += featureValue * outputValue;
      agg.count += 1;
    });
  }

  return correlations.map((correlation, index) => {
    const agg = aggregates[index];
    const n = agg.count || 0;
    if (n === 0) {
      return {
        id: correlation.id,
        featureIndex: correlation.featureIndex,
        outputIndex: correlation.outputIndex,
        orientationSign: correlation.orientationSign,
        correlation: 0,
      };
    }
    const numerator = n * agg.sumFeatureOutput - agg.sumFeature * agg.sumOutput;
    const denomFeature = n * agg.sumFeatureSq - agg.sumFeature * agg.sumFeature;
    const denomOutput = n * agg.sumOutputSq - agg.sumOutput * agg.sumOutput;
    const denominator = Math.sqrt(Math.max(denomFeature, 0) * Math.max(denomOutput, 0));
    const corrValue = denominator > 0 ? numerator / denominator : 0;
    return {
      id: correlation.id,
      featureIndex: correlation.featureIndex,
      outputIndex: correlation.outputIndex,
      orientationSign: correlation.orientationSign,
      correlation: Number.isFinite(corrValue) ? corrValue : 0,
    };
  });
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
    const { dataset, summary, modelUrl, hyperparameters, correlations: requestedCorrelations } = options;
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
    state.activeCorrelations = sanitizeCorrelations(requestedCorrelations, dataset);
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
    if (state.activeCorrelations.length > 0) {
      applyCorrelationTargets(clonedTargets, clonedFeatures, dataset, state.activeCorrelations);
    }
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
        const correlationMetrics = evaluateCorrelationMetrics(
          message.result.model,
          state.activeDataset,
          state.activeCorrelations,
        );
        const statsWithCorrelations = {
          ...message.stats,
          correlations: correlationMetrics,
        };
        resolvePending({ cancelled: false, result: message.result, stats: statsWithCorrelations });
        cb.onStatus({ status: TRAINING_STATUS.COMPLETED, detail: statsWithCorrelations });
        runWarmup(message.result.model, statsWithCorrelations).then((warmup) => {
          cb.onComplete({
            modelDefinition: message.result.model,
            stats: statsWithCorrelations,
            warmup,
            correlationMetrics,
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
