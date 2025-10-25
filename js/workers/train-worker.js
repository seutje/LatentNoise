const ACTIVATIONS = {
  relu: {
    activate: (x) => (x > 0 ? x : 0),
    derivative: (pre) => (pre > 0 ? 1 : 0),
  },
  elu: {
    activate: (x) => (x >= 0 ? x : Math.expm1(x)),
    derivative: (pre, out) => (pre >= 0 ? 1 : out + 1),
  },
  tanh: {
    activate: (x) => Math.tanh(x),
    derivative: (pre, out) => 1 - out * out,
  },
  linear: {
    activate: (x) => x,
    derivative: () => 1,
  },
};

const TRAINING_CONTROL = {
  idle: 'idle',
  running: 'running',
  paused: 'paused',
};

const DEFAULT_OPTIONS = {
  learningRateDecay: 0.92,
  minLearningRate: 1e-5,
  gradientClipNorm: 5,
};

const state = {
  control: TRAINING_CONTROL.idle,
  paused: false,
  cancelRequested: false,
  pauseResolvers: [],
  trainingPromise: null,
};

function postStatus(stateName, detail) {
  self.postMessage({ type: 'status', state: stateName, detail });
}

function postProgress(payload) {
  self.postMessage({ type: 'progress', payload });
}

function postResult(result, stats) {
  self.postMessage({ type: 'result', result, stats });
}

function postError(error) {
  const payload = error instanceof Error ? { message: error.message, stack: error.stack } : { message: String(error) };
  self.postMessage({ type: 'error', error: payload });
}

function sanitizeNumber(value, fallback, min = -Infinity, max = Infinity) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return Math.min(Math.max(numeric, min), max);
}

function sanitizeOptions(options) {
  const lrDecay = sanitizeNumber(options?.learningRateDecay, DEFAULT_OPTIONS.learningRateDecay, 0.5, 0.999);
  const minLr = sanitizeNumber(options?.minLearningRate, DEFAULT_OPTIONS.minLearningRate, 1e-6, 0.05);
  const clipNorm = sanitizeNumber(options?.gradientClipNorm, DEFAULT_OPTIONS.gradientClipNorm, 0.1, 50);
  return {
    learningRateDecay: lrDecay,
    minLearningRate: minLr,
    gradientClipNorm: clipNorm,
  };
}

function sanitizeHyper(raw) {
  return {
    epochs: Math.max(1, Math.floor(Number(raw?.epochs ?? 1))),
    learningRate: sanitizeNumber(raw?.learningRate, 0.001, 1e-6, 0.5),
    batchSize: Math.max(1, Math.floor(Number(raw?.batchSize ?? 64))),
    l2: Math.max(0, Number(raw?.l2 ?? 0)),
  };
}

function clampSignedUnit(value) {
  if (!Number.isFinite(value)) {
    return 0;
  }
  if (value > 1) {
    return 1;
  }
  if (value < -1) {
    return -1;
  }
  return value;
}

function clampPositiveUnit(value) {
  if (!Number.isFinite(value) || value <= 0) {
    return 0;
  }
  if (value >= 1) {
    return 1;
  }
  return value;
}

function sanitizeCorrelationFeature(value, type) {
  return type === 'signed' ? clampSignedUnit(value) : clampPositiveUnit(value);
}

function projectCorrelationTarget(rawValue, featureType, orientationSign) {
  if (featureType === 'signed') {
    return clampSignedUnit(rawValue) * orientationSign;
  }
  const normalized = clampPositiveUnit(rawValue) * 2 - 1;
  return clampSignedUnit(normalized * orientationSign);
}

function ensureActivation(name) {
  const key = typeof name === 'string' ? name.toLowerCase() : '';
  if (!Object.prototype.hasOwnProperty.call(ACTIVATIONS, key)) {
    throw new Error(`Unsupported activation "${name}".`);
  }
  return { name: key, ...ACTIVATIONS[key] };
}

function toFloat32Array(source, expectedLength, label) {
  const arr =
    source instanceof Float32Array
      ? new Float32Array(source)
      : Array.isArray(source)
        ? Float32Array.from(source)
        : null;
  if (!arr) {
    throw new Error(`Layer field "${label}" must be an array.`);
  }
  if (typeof expectedLength === 'number' && arr.length !== expectedLength) {
    throw new Error(
      `Layer field "${label}" expected length ${expectedLength}, received ${arr.length}.`,
    );
  }
  return arr;
}

function buildRuntimeModel(rawModel) {
  if (!rawModel || typeof rawModel !== 'object') {
    throw new Error('Model definition must be an object.');
  }
  const inputSize = Number(rawModel.input);
  if (!Number.isFinite(inputSize) || inputSize <= 0) {
    throw new Error('Model definition missing valid input size.');
  }
  const layersRaw = Array.isArray(rawModel.layers) ? rawModel.layers : [];
  if (layersRaw.length === 0) {
    throw new Error('Model definition requires at least one layer.');
  }
  const normMean = toFloat32Array(rawModel.normalization?.mean ?? rawModel.norm?.mean ?? [], inputSize, 'normalization.mean');
  const stdSource = rawModel.normalization?.std ?? rawModel.norm?.std;
  const normStd =
    stdSource !== undefined
      ? toFloat32Array(stdSource, inputSize, 'normalization.std')
      : (() => {
          const arr = new Float32Array(inputSize);
          arr.fill(1);
          return arr;
        })();
  const normInvStd = new Float32Array(inputSize);
  for (let i = 0; i < inputSize; i += 1) {
    const std = normStd[i];
    normInvStd[i] = std > 0 && Number.isFinite(std) ? 1 / std : 1;
  }

  const layers = [];
  let prevSize = inputSize;
  for (let layerIndex = 0; layerIndex < layersRaw.length; layerIndex += 1) {
    const layerRaw = layersRaw[layerIndex];
    const biases = toFloat32Array(layerRaw.bias ?? layerRaw.biases, undefined, `layer[${layerIndex}].bias`);
    const outputSize = biases.length;
    if (outputSize === 0) {
      throw new Error(`Layer ${layerIndex} must have non-empty bias array.`);
    }
    const weights = toFloat32Array(layerRaw.weights, prevSize * outputSize, `layer[${layerIndex}].weights`);
    const activation = ensureActivation(layerRaw.activation ?? 'linear');
    const layer = {
      activation,
      weights,
      biases,
      inputSize: prevSize,
      outputSize,
      preActivations: new Float32Array(outputSize),
      outputs: new Float32Array(outputSize),
      deltas: new Float32Array(outputSize),
      weightGrads: new Float32Array(weights.length),
      biasGrads: new Float32Array(outputSize),
    };
    layers.push(layer);
    prevSize = outputSize;
  }

  return {
    inputSize,
    normMean,
    normInvStd,
    layers,
    inputBuffer: new Float32Array(inputSize),
    outputBuffer: new Float32Array(layers[layers.length - 1].outputSize),
  };
}

function prepareDataset(dataset) {
  if (!dataset || typeof dataset !== 'object') {
    throw new Error('Dataset missing.');
  }
  const { features, targets, featureSize, targetSize, frameCount, splits } = dataset;
  if (!(features instanceof Float32Array) || !(targets instanceof Float32Array)) {
    throw new Error('Dataset features/targets must be Float32Array.');
  }
  if (!Number.isFinite(featureSize) || featureSize <= 0) {
    throw new Error('Dataset featureSize must be positive.');
  }
  if (!Number.isFinite(targetSize) || targetSize <= 0) {
    throw new Error('Dataset targetSize must be positive.');
  }
  const expectedFeatureLength = featureSize * frameCount;
  const expectedTargetLength = targetSize * frameCount;
  if (features.length < expectedFeatureLength || targets.length < expectedTargetLength) {
    throw new Error('Dataset buffers shorter than expected length.');
  }
  const trainStart = Math.max(0, Math.floor(Number(splits?.train?.start ?? 0)));
  const trainCount = Math.max(0, Math.floor(Number(splits?.train?.count ?? 0)));
  const valStart = Math.max(0, Math.floor(Number(splits?.validation?.start ?? trainStart + trainCount)));
  const valCount = Math.max(0, Math.floor(Number(splits?.validation?.count ?? 0)));

  function clampCount(start, count) {
    if (start >= frameCount) {
      return 0;
    }
    return Math.max(0, Math.min(count, frameCount - start));
  }

  const cappedTrainCount = clampCount(trainStart, trainCount);
  const cappedValCount = clampCount(valStart, valCount);

  const trainIndices = new Uint32Array(cappedTrainCount);
  for (let i = 0; i < cappedTrainCount; i += 1) {
    trainIndices[i] = trainStart + i;
  }
  const valIndices = new Uint32Array(cappedValCount);
  for (let i = 0; i < cappedValCount; i += 1) {
    valIndices[i] = valStart + i;
  }

  if (trainIndices.length === 0) {
    throw new Error('Training split is empty; cannot train model.');
  }

  return {
    features,
    targets,
    featureSize,
    targetSize,
    frameCount,
    trainIndices,
    valIndices,
  };
}

function prepareCorrelations(rawCorrelations, dataset) {
  if (!Array.isArray(rawCorrelations) || rawCorrelations.length === 0) {
    return [];
  }
  const sanitized = [];
  const { featureSize, targetSize } = dataset;
  rawCorrelations.forEach((entry, index) => {
    const featureIndex = Number(entry?.featureIndex);
    const outputIndex = Number(entry?.outputIndex);
    if (!Number.isInteger(featureIndex) || featureIndex < 0 || featureIndex >= featureSize) {
      return;
    }
    if (!Number.isInteger(outputIndex) || outputIndex < 0 || outputIndex >= targetSize) {
      return;
    }
    const orientationSign = Number(entry?.orientationSign) === -1 ? -1 : 1;
    const weight = Number(entry?.weight);
    sanitized.push({
      id:
        typeof entry?.id === 'string' && entry.id.length > 0
          ? entry.id
          : `${featureIndex}:${outputIndex}:${orientationSign === -1 ? 'inv' : 'dir'}:${index}`,
      featureIndex,
      featureType: entry?.featureType === 'signed' ? 'signed' : 'positive',
      outputIndex,
      orientationSign,
      inverse: Boolean(entry?.inverse) || orientationSign === -1,
      weight: Number.isFinite(weight) && weight > 0 ? Number(weight) : 1,
    });
  });
  return sanitized;
}

function createTrainingMode(dataset, correlations) {
  if (Array.isArray(correlations) && correlations.length > 0) {
    const weightSum = correlations.reduce((sum, correlation) => sum + correlation.weight, 0) || 1;
    return {
      kind: 'correlation',
      correlations,
      weightSum,
      invWeightSum: 1 / weightSum,
    };
  }
  const invTargetSize = dataset.targetSize > 0 ? 1 / dataset.targetSize : 1;
  return {
    kind: 'supervised',
    invTargetSize,
  };
}

function shuffleIndices(indices) {
  for (let i = indices.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    const temp = indices[i];
    indices[i] = indices[j];
    indices[j] = temp;
  }
}

function waitForResume() {
  if (!state.paused) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    state.pauseResolvers.push(resolve);
  });
}

function resumeFromPause() {
  if (!state.paused) {
    return;
  }
  state.paused = false;
  const resolvers = state.pauseResolvers.slice();
  state.pauseResolvers.length = 0;
  resolvers.forEach((resolve) => {
    try {
      resolve();
    } catch {
      // ignore resolver errors
    }
  });
}

function cancelTraining() {
  state.cancelRequested = true;
  resumeFromPause();
}

function maybeYield(iteration) {
  if ((iteration & 0xf) === 0) {
    return new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
  }
  return null;
}

function sanitizeFinite(value, fallback = 0) {
  return Number.isFinite(value) ? value : fallback;
}

function forwardPass(runtime, inputVector) {
  let currentInput = inputVector;
  const { layers } = runtime;
  for (let layerIndex = 0; layerIndex < layers.length; layerIndex += 1) {
    const layer = layers[layerIndex];
    const { weights, biases, inputSize, outputSize, preActivations, outputs, activation } = layer;
    for (let outIndex = 0; outIndex < outputSize; outIndex += 1) {
      let sum = biases[outIndex];
      const weightOffset = outIndex * inputSize;
      for (let inIndex = 0; inIndex < inputSize; inIndex += 1) {
        sum += weights[weightOffset + inIndex] * currentInput[inIndex];
      }
      const activated = activation.activate(sum);
      preActivations[outIndex] = sum;
      outputs[outIndex] = sanitizeFinite(activated);
    }
    currentInput = outputs;
  }
  runtime.outputBuffer.set(currentInput);
  return runtime.outputBuffer;
}

function normalizeInput(runtime, dataset, frameIndex) {
  const { features } = dataset;
  const { normMean, normInvStd, inputBuffer, inputSize } = runtime;
  const offset = frameIndex * dataset.featureSize;
  for (let i = 0; i < inputSize; i += 1) {
    const raw = features[offset + i];
    const centered = sanitizeFinite(raw) - normMean[i];
    inputBuffer[i] = sanitizeFinite(centered * normInvStd[i]);
  }
  return inputBuffer;
}

function computeSupervisedLossAndGradients(runtime, dataset, frameIndex, invTargetSize) {
  const input = normalizeInput(runtime, dataset, frameIndex);
  const output = forwardPass(runtime, input);
  const { targets, targetSize } = dataset;
  const targetOffset = frameIndex * targetSize;
  const layers = runtime.layers;
  const lastLayer = layers[layers.length - 1];

  let sampleLoss = 0;
  for (let i = 0; i < targetSize; i += 1) {
    const prediction = output[i];
    const target = sanitizeFinite(targets[targetOffset + i]);
    const diff = prediction - target;
    sampleLoss += diff * diff;
    const gradient = diff * invTargetSize;
    lastLayer.deltas[i] = gradient * lastLayer.activation.derivative(lastLayer.preActivations[i], prediction);
  }
  const lossValue = 0.5 * sampleLoss * invTargetSize;

  for (let layerIndex = layers.length - 2; layerIndex >= 0; layerIndex -= 1) {
    const layer = layers[layerIndex];
    const nextLayer = layers[layerIndex + 1];
    const { outputSize } = layer;
    for (let outIndex = 0; outIndex < outputSize; outIndex += 1) {
      let sum = 0;
      for (let nextOut = 0; nextOut < nextLayer.outputSize; nextOut += 1) {
        const weight = nextLayer.weights[nextOut * layer.outputSize + outIndex];
        sum += weight * nextLayer.deltas[nextOut];
      }
      const derivative = layer.activation.derivative(layer.preActivations[outIndex], layer.outputs[outIndex]);
      layer.deltas[outIndex] = sum * derivative;
    }
  }

  for (let layerIndex = 0; layerIndex < layers.length; layerIndex += 1) {
    const layer = layers[layerIndex];
    const prevOutputs = layerIndex === 0 ? input : layers[layerIndex - 1].outputs;
    const { inputSize, outputSize, deltas, weightGrads, biasGrads } = layer;
    for (let outIndex = 0; outIndex < outputSize; outIndex += 1) {
      const delta = deltas[outIndex];
      biasGrads[outIndex] += delta;
      const weightOffset = outIndex * inputSize;
      for (let inIndex = 0; inIndex < inputSize; inIndex += 1) {
        weightGrads[weightOffset + inIndex] += delta * prevOutputs[inIndex];
      }
    }
  }
  return lossValue;
}

function computeCorrelationLossAndGradients(runtime, dataset, frameIndex, mode) {
  const input = normalizeInput(runtime, dataset, frameIndex);
  const output = forwardPass(runtime, input);
  const layers = runtime.layers;
  const lastLayer = layers[layers.length - 1];
  const deltas = lastLayer.deltas;
  const featureOffset = frameIndex * dataset.featureSize;

  for (let i = 0; i < deltas.length; i += 1) {
    deltas[i] = 0;
  }

  let weightedLoss = 0;
  for (let i = 0; i < mode.correlations.length; i += 1) {
    const correlation = mode.correlations[i];
    const rawFeature = dataset.features[featureOffset + correlation.featureIndex];
    const featureValue = sanitizeCorrelationFeature(rawFeature, correlation.featureType);
    const target = projectCorrelationTarget(featureValue, correlation.featureType, correlation.orientationSign);
    const prediction = output[correlation.outputIndex];
    const error = prediction - target;
    weightedLoss += correlation.weight * error * error;
    const derivative = lastLayer.activation.derivative(
      lastLayer.preActivations[correlation.outputIndex],
      prediction,
    );
    deltas[correlation.outputIndex] += correlation.weight * error * derivative;
  }

  for (let layerIndex = layers.length - 2; layerIndex >= 0; layerIndex -= 1) {
    const layer = layers[layerIndex];
    const nextLayer = layers[layerIndex + 1];
    const { outputSize } = layer;
    for (let outIndex = 0; outIndex < outputSize; outIndex += 1) {
      let sum = 0;
      for (let nextOut = 0; nextOut < nextLayer.outputSize; nextOut += 1) {
        const weight = nextLayer.weights[nextOut * layer.outputSize + outIndex];
        sum += weight * nextLayer.deltas[nextOut];
      }
      const derivative = layer.activation.derivative(layer.preActivations[outIndex], layer.outputs[outIndex]);
      layer.deltas[outIndex] = sum * derivative;
    }
  }

  for (let layerIndex = 0; layerIndex < layers.length; layerIndex += 1) {
    const layer = layers[layerIndex];
    const prevOutputs = layerIndex === 0 ? input : layers[layerIndex - 1].outputs;
    const { inputSize, outputSize, deltas: layerDeltas, weightGrads, biasGrads } = layer;
    for (let outIndex = 0; outIndex < outputSize; outIndex += 1) {
      const delta = layerDeltas[outIndex];
      if (delta === 0) {
        continue;
      }
      biasGrads[outIndex] += delta;
      const weightOffset = outIndex * inputSize;
      for (let inIndex = 0; inIndex < inputSize; inIndex += 1) {
        weightGrads[weightOffset + inIndex] += delta * prevOutputs[inIndex];
      }
    }
  }

  return 0.5 * weightedLoss * mode.invWeightSum;
}

function computeLossAndGradients(runtime, dataset, frameIndex, mode) {
  if (mode.kind === 'correlation') {
    return computeCorrelationLossAndGradients(runtime, dataset, frameIndex, mode);
  }
  return computeSupervisedLossAndGradients(runtime, dataset, frameIndex, mode.invTargetSize);
}

function applyGradients(runtime, hyper, options, sampleCount, epochLearningRate) {
  if (sampleCount <= 0) {
    return;
  }
  const { layers } = runtime;
  const scale = 1 / sampleCount;
  let gradNormSq = 0;

  for (let layerIndex = 0; layerIndex < layers.length; layerIndex += 1) {
    const layer = layers[layerIndex];
    const { weightGrads, biasGrads } = layer;
    for (let i = 0; i < weightGrads.length; i += 1) {
      const scaled = weightGrads[i] * scale;
      gradNormSq += scaled * scaled;
    }
    for (let i = 0; i < biasGrads.length; i += 1) {
      const scaled = biasGrads[i] * scale;
      gradNormSq += scaled * scaled;
    }
  }

  let clipScale = 1;
  if (gradNormSq > 0) {
    const norm = Math.sqrt(gradNormSq);
    if (norm > options.gradientClipNorm) {
      clipScale = options.gradientClipNorm / norm;
    }
  }

  const lr = Math.max(options.minLearningRate, epochLearningRate);

  for (let layerIndex = 0; layerIndex < layers.length; layerIndex += 1) {
    const layer = layers[layerIndex];
    const { weights, biases, weightGrads, biasGrads } = layer;
    for (let i = 0; i < weights.length; i += 1) {
      const grad = (weightGrads[i] * scale) * clipScale + hyper.l2 * weights[i];
      weights[i] -= lr * grad;
      weightGrads[i] = 0;
    }
    for (let i = 0; i < biases.length; i += 1) {
      const grad = (biasGrads[i] * scale) * clipScale;
      biases[i] -= lr * grad;
      biasGrads[i] = 0;
    }
  }
}

function resetGradients(runtime) {
  const { layers } = runtime;
  for (let layerIndex = 0; layerIndex < layers.length; layerIndex += 1) {
    const layer = layers[layerIndex];
    layer.weightGrads.fill(0);
    layer.biasGrads.fill(0);
  }
}

function computeSupervisedValidationLoss(runtime, dataset) {
  const { valIndices, targetSize } = dataset;
  if (!valIndices || valIndices.length === 0) {
    return null;
  }
  const invTargetSize = 1 / targetSize;
  let totalLoss = 0;
  for (let i = 0; i < valIndices.length; i += 1) {
    const frameIndex = valIndices[i];
    const input = normalizeInput(runtime, dataset, frameIndex);
    const output = forwardPass(runtime, input);
    const targetOffset = frameIndex * targetSize;
    let sampleLoss = 0;
    for (let j = 0; j < targetSize; j += 1) {
      const diff = output[j] - dataset.targets[targetOffset + j];
      sampleLoss += diff * diff;
    }
    totalLoss += 0.5 * sampleLoss * invTargetSize;
  }
  return totalLoss / valIndices.length;
}
function computeCorrelationValidationLoss(runtime, dataset, mode) {
  const { valIndices } = dataset;
  if (!valIndices || valIndices.length === 0) {
    return null;
  }
  let totalLoss = 0;
  for (let i = 0; i < valIndices.length; i += 1) {
    const frameIndex = valIndices[i];
    const input = normalizeInput(runtime, dataset, frameIndex);
    const output = forwardPass(runtime, input);
    const featureOffset = frameIndex * dataset.featureSize;
    let frameLoss = 0;
    for (let j = 0; j < mode.correlations.length; j += 1) {
      const correlation = mode.correlations[j];
      const rawFeature = dataset.features[featureOffset + correlation.featureIndex];
      const featureValue = sanitizeCorrelationFeature(rawFeature, correlation.featureType);
      const target = projectCorrelationTarget(featureValue, correlation.featureType, correlation.orientationSign);
      const prediction = output[correlation.outputIndex];
      const error = prediction - target;
      frameLoss += correlation.weight * error * error;
    }
    totalLoss += 0.5 * frameLoss * mode.invWeightSum;
  }
  return totalLoss / valIndices.length;
}

function computeValidationLoss(runtime, dataset, mode) {
  if (mode.kind === 'correlation') {
    return computeCorrelationValidationLoss(runtime, dataset, mode);
  }
  return computeSupervisedValidationLoss(runtime, dataset);
}
function computeCorrelationMetrics(runtime, dataset, mode) {
  if (!Array.isArray(mode.correlations) || mode.correlations.length === 0) {
    return [];
  }
  const frameCount = dataset.frameCount || 0;
  if (frameCount <= 0) {
    return mode.correlations.map((correlation) => ({
      id: correlation.id,
      featureIndex: correlation.featureIndex,
      outputIndex: correlation.outputIndex,
      inverse: correlation.inverse,
      orientationSign: correlation.orientationSign,
      weight: correlation.weight,
      correlation: 0,
      mse: 0,
    }));
  }
  const aggregators = mode.correlations.map(() => ({
    sumFeature: 0,
    sumOutput: 0,
    sumFeatureSq: 0,
    sumOutputSq: 0,
    sumFeatureOutput: 0,
    mse: 0,
  }));
  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    const input = normalizeInput(runtime, dataset, frameIndex);
    const output = forwardPass(runtime, input);
    const featureOffset = frameIndex * dataset.featureSize;
    for (let i = 0; i < mode.correlations.length; i += 1) {
      const correlation = mode.correlations[i];
      const featureValue = sanitizeCorrelationFeature(
        dataset.features[featureOffset + correlation.featureIndex],
        correlation.featureType,
      );
      const prediction = output[correlation.outputIndex];
      const aggregator = aggregators[i];
      aggregator.sumFeature += featureValue;
      aggregator.sumOutput += prediction;
      aggregator.sumFeatureSq += featureValue * featureValue;
      aggregator.sumOutputSq += prediction * prediction;
      aggregator.sumFeatureOutput += featureValue * prediction;
      const target = projectCorrelationTarget(featureValue, correlation.featureType, correlation.orientationSign);
      const error = prediction - target;
      aggregator.mse += error * error;
    }
  }
  return mode.correlations.map((correlation, index) => {
    const aggregator = aggregators[index];
    const n = frameCount || 1;
    const numerator = n * aggregator.sumFeatureOutput - aggregator.sumFeature * aggregator.sumOutput;
    const denomFeature = n * aggregator.sumFeatureSq - aggregator.sumFeature * aggregator.sumFeature;
    const denomOutput = n * aggregator.sumOutputSq - aggregator.sumOutput * aggregator.sumOutput;
    const denominator = Math.sqrt(Math.max(denomFeature, 0) * Math.max(denomOutput, 0));
    const rawCorrelation = denominator > 0 ? numerator / denominator : 0;
    const clamped = Math.max(-1, Math.min(1, rawCorrelation));
    return {
      id: correlation.id,
      featureIndex: correlation.featureIndex,
      outputIndex: correlation.outputIndex,
      inverse: correlation.inverse,
      orientationSign: correlation.orientationSign,
      weight: correlation.weight,
      correlation: clamped,
      mse: aggregator.mse / n,
    };
  });
}

function buildUpdatedModelDefinition(baseModel, runtime) {
  const layers = runtime.layers.map((layer, index) => {
    const source = baseModel.layers[index] ?? {};
    return {
      activation: source.activation ?? 'linear',
      weights: Array.from(layer.weights),
      bias: Array.from(layer.biases),
    };
  });
  return {
    input: baseModel.input,
    normalization: {
      mean: Array.from(baseModel.normalization?.mean ?? baseModel.norm?.mean ?? []),
      std: Array.from(baseModel.normalization?.std ?? baseModel.norm?.std ?? []),
    },
    layers,
  };
}

async function trainModel(payload) {
  const dataset = prepareDataset(payload.dataset);
  const runtime = buildRuntimeModel(payload.model);
  const hyper = sanitizeHyper(payload.hyperparameters);
  const options = sanitizeOptions(payload.options);
  const correlations = prepareCorrelations(payload.correlations, dataset);
  const trainingMode = createTrainingMode(dataset, correlations);

  const trainIndices = dataset.trainIndices;
  const batchesPerEpoch = Math.max(1, Math.ceil(trainIndices.length / hyper.batchSize));
  const totalBatches = batchesPerEpoch * hyper.epochs;
  let processedBatches = 0;
  let samplesProcessedTotal = 0;
  let lastTrainLoss = null;

  const startedAt = typeof payload.options?.startedAt === 'number' ? payload.options.startedAt : performance.now();
  const loopStartedAt = performance.now();
  let lastValidationLoss = null;

  for (let epoch = 0; epoch < hyper.epochs; epoch += 1) {
    if (state.cancelRequested) {
      break;
    }

    shuffleIndices(trainIndices);
    resetGradients(runtime);

    const epochLearningRate = hyper.learningRate * Math.pow(options.learningRateDecay, epoch);
    const trainSamples = trainIndices.length;
    let epochLoss = 0;
    let samplesAccumulated = 0;

    for (let batchIndex = 0; batchIndex < batchesPerEpoch; batchIndex += 1) {
      if (state.cancelRequested) {
        break;
      }
      await waitForResume();

      const startIndex = batchIndex * hyper.batchSize;
      const endIndex = Math.min(startIndex + hyper.batchSize, trainSamples);
      if (startIndex >= endIndex) {
        continue;
      }

      let batchLoss = 0;
      for (let i = startIndex; i < endIndex; i += 1) {
        const frameIndex = trainIndices[i];
        batchLoss += computeLossAndGradients(runtime, dataset, frameIndex, trainingMode);
      }

      applyGradients(runtime, hyper, options, endIndex - startIndex, epochLearningRate);
      samplesAccumulated += endIndex - startIndex;
      epochLoss += batchLoss;

      processedBatches += 1;
      const progress = processedBatches / totalBatches;
      const elapsedMs = performance.now() - loopStartedAt;
      const etaMs = progress > 0 ? elapsedMs * (1 / progress - 1) : null;
      postProgress({
        epoch: epoch + 1,
        epochs: hyper.epochs,
        batch: batchIndex + 1,
        batches: batchesPerEpoch,
        progress,
        trainLoss: samplesAccumulated > 0 ? epochLoss / samplesAccumulated : null,
        valLoss: lastValidationLoss,
        learningRate: Math.max(options.minLearningRate, epochLearningRate),
        elapsedMs,
        etaMs,
      });

      await maybeYield(processedBatches);
      if (state.cancelRequested) {
        break;
      }
    }

    lastTrainLoss = samplesAccumulated > 0 ? epochLoss / samplesAccumulated : null;
    samplesProcessedTotal += samplesAccumulated;

    if (state.cancelRequested) {
      break;
    }

    lastValidationLoss = computeValidationLoss(runtime, dataset, trainingMode);
    postProgress({
      epoch: epoch + 1,
      epochs: hyper.epochs,
      batch: batchesPerEpoch,
      batches: batchesPerEpoch,
      progress: processedBatches / totalBatches,
      trainLoss: lastTrainLoss,
      valLoss: lastValidationLoss,
      learningRate: Math.max(options.minLearningRate, hyper.learningRate * Math.pow(options.learningRateDecay, epoch + 1)),
      elapsedMs: performance.now() - loopStartedAt,
      etaMs: 0,
    });
  }

  if (state.cancelRequested) {
    postStatus('cancelled', { reason: 'cancelled' });
    return;
  }

  const elapsedMs = performance.now() - loopStartedAt;
  const stats = {
    epochsCompleted: Math.min(hyper.epochs, Math.round(processedBatches / batchesPerEpoch)),
    trainLoss: lastTrainLoss,
    valLoss: lastValidationLoss,
    elapsedMs,
    startedAt,
    samplesProcessed: samplesProcessedTotal,
  };

  const correlationMetrics =
    trainingMode.kind === 'correlation'
      ? computeCorrelationMetrics(runtime, dataset, trainingMode)
      : [];

  const model = buildUpdatedModelDefinition(payload.model, runtime);
  postResult({ model }, { ...stats, correlationMetrics });
}

self.addEventListener('message', (event) => {
  const message = event.data;
  if (!message || typeof message !== 'object') {
    return;
  }
  const { type, payload } = message;

  switch (type) {
    case 'train': {
      if (state.control === TRAINING_CONTROL.running) {
        postStatus('error', { message: 'Training already in progress.' });
        return;
      }
      state.control = TRAINING_CONTROL.running;
      state.paused = false;
      state.cancelRequested = false;
      postStatus('running', { stage: 'starting' });
      try {
        const promise = trainModel(payload);
        state.trainingPromise = promise;
        promise
          .catch((error) => {
            if (state.cancelRequested) {
              postStatus('cancelled', { reason: 'cancelled' });
            } else {
              postError(error);
            }
          })
          .finally(() => {
            state.control = TRAINING_CONTROL.idle;
            state.paused = false;
            state.cancelRequested = false;
            state.trainingPromise = null;
            resumeFromPause();
          });
      } catch (error) {
        postError(error);
        state.control = TRAINING_CONTROL.idle;
        state.trainingPromise = null;
      }
      break;
    }
    case 'pause': {
      if (state.control !== TRAINING_CONTROL.running || state.paused) {
        return;
      }
      state.paused = true;
      postStatus('paused', { stage: 'paused' });
      break;
    }
    case 'resume': {
      if (state.control !== TRAINING_CONTROL.running || !state.paused) {
        return;
      }
      resumeFromPause();
      postStatus('running', { stage: 'resumed' });
      break;
    }
    case 'cancel': {
      if (state.control === TRAINING_CONTROL.idle) {
        postStatus('cancelled', { reason: 'cancelled-before-start' });
        return;
      }
      cancelTraining();
      break;
    }
    default:
      break;
  }
});
