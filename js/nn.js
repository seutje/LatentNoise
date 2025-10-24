const ACTIVATIONS = /** @type {const} */ ({
  relu: (x) => (x > 0 ? x : 0),
  elu: (x) => (x >= 0 ? x : Math.expm1(x)),
  tanh: (x) => Math.tanh(x),
  linear: (x) => x,
});

let currentModel = null;

function assertFinite(value, fallback = 0) {
  return Number.isFinite(value) ? value : fallback;
}

function toFloat32Array(source, expectedLength, label) {
  if (!Array.isArray(source) && !(source instanceof Float32Array)) {
    throw new TypeError(`Model field "${label}" must be an array of numbers.`);
  }
  const array = source instanceof Float32Array ? new Float32Array(source) : Float32Array.from(source);
  if (typeof expectedLength === 'number' && array.length !== expectedLength) {
    throw new Error(`Model field "${label}" expected length ${expectedLength}, received ${array.length}.`);
  }
  return array;
}

function createLayer(rawLayer, inputSize) {
  const activationName = String(rawLayer.activation || 'linear').toLowerCase();
  if (!(activationName in ACTIVATIONS)) {
    throw new Error(`Unsupported activation "${rawLayer.activation}".`);
  }

  const biases = toFloat32Array(rawLayer.bias || rawLayer.biases, undefined, 'bias');
  const outputSize = biases.length;
  if (outputSize === 0) {
    throw new Error('Layer bias array must not be empty.');
  }

  const expectedWeightsLength = inputSize * outputSize;
  const weights = toFloat32Array(rawLayer.weights, expectedWeightsLength, 'weights');

  return {
    activation: activationName,
    activationFn: ACTIVATIONS[activationName],
    weights,
    biases,
    inputSize,
    outputSize,
    buffer: new Float32Array(outputSize),
  };
}

function buildModel(raw) {
  if (!raw || typeof raw !== 'object') {
    throw new TypeError('Model definition must be an object.');
  }

  const rawLayers = Array.isArray(raw.layers) ? raw.layers : [];
  if (rawLayers.length === 0) {
    throw new Error('Model must contain at least one layer.');
  }

  const inputSize = Number.isFinite(raw.input) ? raw.input : Number(rawLayers[0]?.input ?? rawLayers[0]?.in);
  if (!Number.isFinite(inputSize) || inputSize <= 0) {
    throw new Error('Model must provide a positive "input" size.');
  }

  const meanSource = raw.normalization?.mean ?? raw.norm?.mean;
  const stdSource = raw.normalization?.std ?? raw.norm?.std;

  const normMean =
    meanSource !== undefined
      ? toFloat32Array(meanSource, inputSize, 'normalization.mean')
      : new Float32Array(inputSize);
  const normStd =
    stdSource !== undefined
      ? toFloat32Array(stdSource, inputSize, 'normalization.std')
      : (() => {
          const defaults = new Float32Array(inputSize);
          defaults.fill(1);
          return defaults;
        })();

  const invStd = new Float32Array(inputSize);
  for (let i = 0; i < inputSize; i += 1) {
    const std = normStd[i];
    invStd[i] = std > 0 && Number.isFinite(std) ? 1 / std : 1;
  }

  let prevSize = inputSize;
  const layers = rawLayers.map((layerRaw) => {
    const layer = createLayer(layerRaw, prevSize);
    prevSize = layer.outputSize;
    return layer;
  });

  const outputSize = layers[layers.length - 1].outputSize;

  return {
    inputSize,
    outputSize,
    layers,
    normMean,
    normInvStd: invStd,
    normBuffer: new Float32Array(inputSize),
    inputBuffer: new Float32Array(inputSize),
    outputBuffer: new Float32Array(outputSize),
  };
}

function normalizeWithModel(model, features) {
  const { normMean, normInvStd, normBuffer, inputSize } = model;
  for (let i = 0; i < inputSize; i += 1) {
    const value = assertFinite(i < features.length ? features[i] : 0);
    const centered = value - normMean[i];
    const normalized = centered * normInvStd[i];
    normBuffer[i] = assertFinite(normalized);
  }
  return normBuffer;
}

function forwardWithModel(model, normalizedFeatures, outBuffer) {
  const { layers, inputBuffer, outputBuffer, inputSize } = model;
  const source = inputBuffer;
  const len = normalizedFeatures.length;
  for (let i = 0; i < inputSize; i += 1) {
    source[i] = assertFinite(i < len ? normalizedFeatures[i] : 0);
  }

  let current = source;
  for (let layerIndex = 0; layerIndex < layers.length; layerIndex += 1) {
    const layer = layers[layerIndex];
    const { weights, biases, inputSize: layerInputSize, outputSize, activationFn, buffer } = layer;

    for (let outIndex = 0; outIndex < outputSize; outIndex += 1) {
      let sum = biases[outIndex];
      const weightOffset = outIndex * layerInputSize;
      for (let inIndex = 0; inIndex < layerInputSize; inIndex += 1) {
        sum += weights[weightOffset + inIndex] * current[inIndex];
      }
      buffer[outIndex] = assertFinite(activationFn(sum));
    }
    current = buffer;
  }

  const target = outBuffer || outputBuffer;
  if (target.length !== current.length) {
    throw new Error(`Output buffer length mismatch: expected ${current.length}, received ${target.length}.`);
  }
  for (let i = 0; i < current.length; i += 1) {
    target[i] = current[i];
  }
  return target;
}

async function fetchModelDefinition(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load model from "${url}" (${response.status} ${response.statusText}).`);
  }
  return response.json();
}

export async function loadModel(urlOrObject) {
  const rawDefinition =
    typeof urlOrObject === 'string'
      ? await fetchModelDefinition(urlOrObject)
      : urlOrObject;

  const model = buildModel(rawDefinition);
  currentModel = model;
  return {
    inputSize: model.inputSize,
    outputSize: model.outputSize,
    layers: model.layers.length,
  };
}

export async function loadModelDefinition(url) {
  return fetchModelDefinition(url);
}

export function createModel(rawDefinition) {
  return buildModel(rawDefinition);
}

export function infer(model, features, outBuffer) {
  if (!model) {
    throw new Error('infer() requires a model instance.');
  }
  const normalized = normalizeWithModel(model, features);
  return forwardWithModel(model, normalized, outBuffer || model.outputBuffer);
}

export function normalize(features) {
  if (!currentModel) {
    throw new Error('No model loaded. Call loadModel() before normalize().');
  }
  return normalizeWithModel(currentModel, features);
}

export function forward(normalizedFeatures, outBuffer) {
  if (!currentModel) {
    throw new Error('No model loaded. Call loadModel() before forward().');
  }
  return forwardWithModel(currentModel, normalizedFeatures, outBuffer || currentModel.outputBuffer);
}

function runSelfTest() {
  try {
    const rawModel = {
      input: 3,
      normalization: {
        mean: [0, 0, 0],
        std: [1, 1, 1],
      },
      layers: [
        {
          activation: 'relu',
          weights: [
            1, 0.5, -1,
            -0.5, 1, 0.25,
          ],
          bias: [0.25, -0.5],
        },
        {
          activation: 'tanh',
          weights: [2, -1],
          bias: [0.1],
        },
      ],
    };

    const model = buildModel(rawModel);
    const features = new Float32Array([1, -1, 0.5]);
    const normalized = normalizeWithModel(model, features);
    const result = forwardWithModel(model, normalized, model.outputBuffer);
    const expected = 0.5370496;
    const delta = Math.abs(result[0] - expected);
    const passed = delta < 1e-4;
    const logger = passed ? console.info : console.error;
    logger(`[nn] self-test ${passed ? 'passed' : 'failed'} (delta=${delta.toFixed(6)})`);
  } catch (error) {
    console.error('[nn] self-test errored', error);
  }
}

if (typeof window !== 'undefined' && !window.__LN_NN_TESTED__) {
  window.__LN_NN_TESTED__ = true;
  runSelfTest();
}

export function getCurrentModelInfo() {
  if (!currentModel) {
    return null;
  }
  return {
    inputSize: currentModel.inputSize,
    outputSize: currentModel.outputSize,
    layers: currentModel.layers.length,
  };
}
