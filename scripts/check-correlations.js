#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  FEATURE_LABELS,
  FEATURE_TYPES,
  INPUT_SIZE,
  OUTPUT_LABELS,
  OUTPUT_SIZE,
  createRandom,
  randomPositive,
  randomSigned,
} from './correlation-common.js';
import { findTrack, formatTrackList } from './tracks.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PROJECT_ROOT = dirname(__dirname);
const MODELS_DIR = join(PROJECT_ROOT, 'models');

const DEFAULT_SAMPLES = 8192;
const DEFAULT_TOP = 10;

const ACTIVATIONS = new Map([
  ['relu', (x) => (x > 0 ? x : 0)],
  ['elu', (x) => (x >= 0 ? x : Math.expm1(x))],
  ['tanh', (x) => Math.tanh(x)],
  ['linear', (x) => x],
]);

function parseArgs(rawArgs) {
  const positionals = [];
  const options = {};

  rawArgs.forEach((arg) => {
    if (arg.startsWith('--')) {
      const [key, value] = arg.slice(2).split('=');
      options[key] = value === undefined ? true : value;
    } else {
      positionals.push(arg);
    }
  });

  if (positionals.length === 0) {
    throw new Error(
      'Usage: node scripts/check-correlations.js <track> [--samples=8192] [--top=10] [--seed=42]',
    );
  }

  const trackRef = positionals[0];
  const track = findTrack(trackRef);
  if (!track) {
    throw new Error(`Unknown track reference "${trackRef}". Available: ${formatTrackList()}`);
  }

  const samples = options.samples !== undefined ? Number(options.samples) : DEFAULT_SAMPLES;
  if (!Number.isFinite(samples) || samples <= 0) {
    throw new Error(`Invalid sample count: ${options.samples}`);
  }

  const top = options.top !== undefined ? Number(options.top) : DEFAULT_TOP;
  if (!Number.isFinite(top) || top <= 0) {
    throw new Error(`Invalid top count: ${options.top}`);
  }

  const seed = options.seed !== undefined ? Number(options.seed) : null;

  return {
    track,
    samples: Math.trunc(samples),
    top: Math.trunc(top),
    seed,
  };
}

function toFloat32ArrayStrict(source, expectedLength, label) {
  if (!Array.isArray(source) && !(source instanceof Float32Array)) {
    throw new Error(`Model definition is missing ${label}.`);
  }
  const values = Array.from(source, (value) => Number(value));
  if (values.length !== expectedLength) {
    throw new Error(`Expected ${label} length ${expectedLength}, received ${values.length}.`);
  }
  const result = new Float32Array(expectedLength);
  for (let i = 0; i < expectedLength; i += 1) {
    const value = values[i];
    result[i] = Number.isFinite(value) ? value : 0;
  }
  return result;
}

function toFloat32Vector(source, expectedLength, fallback) {
  const values = Array.isArray(source) || ArrayBuffer.isView(source)
    ? Array.from(source, (value) => Number(value))
    : [];
  const result = new Float32Array(expectedLength);
  for (let i = 0; i < expectedLength; i += 1) {
    const value = values[i];
    result[i] = Number.isFinite(value) ? value : fallback;
  }
  return result;
}

function buildLayer(rawLayer, inputSize) {
  const activationName = String(rawLayer?.activation || 'linear').toLowerCase();
  const activation = ACTIVATIONS.get(activationName);
  if (!activation) {
    throw new Error(`Unsupported activation "${rawLayer?.activation}".`);
  }

  const biasSource = rawLayer?.bias ?? rawLayer?.biases;
  const biasLength =
    (Array.isArray(biasSource) || ArrayBuffer.isView(biasSource)) && typeof biasSource.length === 'number'
      ? biasSource.length
      : 0;
  const biases = toFloat32ArrayStrict(biasSource, biasLength, 'layer.bias');
  const outputSize = biases.length;
  if (outputSize === 0) {
    throw new Error('Layer bias array must not be empty.');
  }

  const weights = toFloat32ArrayStrict(rawLayer?.weights, inputSize * outputSize, 'layer.weights');

  return {
    activation,
    inputSize,
    outputSize,
    weights,
    biases,
    buffer: new Float32Array(outputSize),
  };
}

function loadModel(track) {
  const targetPath = join(MODELS_DIR, track.file);
  let raw;
  try {
    raw = JSON.parse(readFileSync(targetPath, 'utf8'));
  } catch (error) {
    throw new Error(
      `Unable to read model for track "${track.name}" at ${targetPath}: ${
        error instanceof Error ? error.message : error
      }`,
    );
  }

  const inputSize = Number(raw?.input);
  if (!Number.isFinite(inputSize) || inputSize !== INPUT_SIZE) {
    throw new Error(`Model input size mismatch. Expected ${INPUT_SIZE}, received ${raw?.input}`);
  }

  const layers = Array.isArray(raw?.layers) ? raw.layers : [];
  if (layers.length === 0) {
    throw new Error('Model definition must include at least one layer.');
  }

  const builtLayers = [];
  let previousSize = inputSize;
  layers.forEach((layer) => {
    const built = buildLayer(layer, previousSize);
    builtLayers.push(built);
    previousSize = built.outputSize;
  });

  const outputSize = builtLayers[builtLayers.length - 1].outputSize;
  if (outputSize !== OUTPUT_SIZE) {
    throw new Error(`Model output size mismatch. Expected ${OUTPUT_SIZE}, received ${outputSize}.`);
  }

  const normalizationMean = toFloat32Vector(raw?.normalization?.mean, inputSize, 0);
  const normalizationStd = toFloat32Vector(raw?.normalization?.std, inputSize, 1);
  const normalizationInvStd = new Float32Array(inputSize);
  for (let i = 0; i < inputSize; i += 1) {
    const std = normalizationStd[i];
    normalizationInvStd[i] = std > 0 && Number.isFinite(std) ? 1 / std : 1;
  }

  return {
    layers: builtLayers,
    normalization: {
      mean: normalizationMean,
      invStd: normalizationInvStd,
    },
  };
}

function generateFeatureSample(target, random) {
  for (let i = 0; i < INPUT_SIZE; i += 1) {
    const label = FEATURE_LABELS[i];
    const type = FEATURE_TYPES[label] || 'positive';
    target[i] = type === 'signed' ? randomSigned(random) : randomPositive(random);
  }
  return target;
}

function normalizeFeatures(rawFeatures, normalization, target) {
  const { mean, invStd } = normalization;
  for (let i = 0; i < INPUT_SIZE; i += 1) {
    const value = rawFeatures[i];
    target[i] = (value - mean[i]) * invStd[i];
  }
  return target;
}

function forwardModel(model, normalizedFeatures, out) {
  let current = normalizedFeatures;
  model.layers.forEach((layer) => {
    const { activation, biases, buffer, inputSize, outputSize, weights } = layer;
    for (let outIndex = 0; outIndex < outputSize; outIndex += 1) {
      let sum = biases[outIndex];
      const weightOffset = outIndex * inputSize;
      for (let inIndex = 0; inIndex < inputSize; inIndex += 1) {
        sum += weights[weightOffset + inIndex] * current[inIndex];
      }
      buffer[outIndex] = activation(sum);
    }
    current = buffer;
  });

  for (let i = 0; i < out.length; i += 1) {
    out[i] = current[i];
  }
  return out;
}

function computeCorrelation(n, sumX, sumX2, sumY, sumY2, sumXY) {
  const meanX = sumX / n;
  const meanY = sumY / n;
  const cov = sumXY / n - meanX * meanY;
  const varX = sumX2 / n - meanX * meanX;
  const varY = sumY2 / n - meanY * meanY;
  const denom = Math.sqrt(varX * varY);
  if (!Number.isFinite(denom) || denom <= 0) {
    return 0;
  }
  return cov / denom;
}

function analyzeCorrelations(model, options) {
  const { samples, seed } = options;
  const random = createRandom(seed);

  const featureSums = new Float64Array(INPUT_SIZE);
  const featureSqSums = new Float64Array(INPUT_SIZE);
  const outputSums = new Float64Array(OUTPUT_SIZE);
  const outputSqSums = new Float64Array(OUTPUT_SIZE);
  const crossSums = Array.from({ length: INPUT_SIZE }, () => new Float64Array(OUTPUT_SIZE));

  const featureVector = new Float32Array(INPUT_SIZE);
  const normalizedVector = new Float32Array(INPUT_SIZE);
  const outputVector = new Float32Array(OUTPUT_SIZE);

  for (let i = 0; i < samples; i += 1) {
    generateFeatureSample(featureVector, random);
    normalizeFeatures(featureVector, model.normalization, normalizedVector);
    forwardModel(model, normalizedVector, outputVector);

    for (let featureIndex = 0; featureIndex < INPUT_SIZE; featureIndex += 1) {
      const featureValue = featureVector[featureIndex];
      featureSums[featureIndex] += featureValue;
      featureSqSums[featureIndex] += featureValue * featureValue;

      const row = crossSums[featureIndex];
      for (let outputIndex = 0; outputIndex < OUTPUT_SIZE; outputIndex += 1) {
        row[outputIndex] += featureValue * outputVector[outputIndex];
      }
    }

    for (let outputIndex = 0; outputIndex < OUTPUT_SIZE; outputIndex += 1) {
      const outputValue = outputVector[outputIndex];
      outputSums[outputIndex] += outputValue;
      outputSqSums[outputIndex] += outputValue * outputValue;
    }
  }

  const correlations = [];
  for (let featureIndex = 0; featureIndex < INPUT_SIZE; featureIndex += 1) {
    for (let outputIndex = 0; outputIndex < OUTPUT_SIZE; outputIndex += 1) {
      const correlation = computeCorrelation(
        samples,
        featureSums[featureIndex],
        featureSqSums[featureIndex],
        outputSums[outputIndex],
        outputSqSums[outputIndex],
        crossSums[featureIndex][outputIndex],
      );
      correlations.push({
        featureIndex,
        outputIndex,
        correlation,
      });
    }
  }

  return correlations;
}

function formatCorrelationEntry(entry) {
  const featureName = FEATURE_LABELS[entry.featureIndex];
  const outputName = OUTPUT_LABELS[entry.outputIndex];
  const magnitude = Math.abs(entry.correlation);
  const orientation = entry.correlation >= 0 ? 'direct' : 'inverse';
  return {
    featureName,
    outputName,
    correlation: entry.correlation,
    magnitude,
    orientation,
  };
}

function main() {
  try {
    const args = parseArgs(process.argv.slice(2));
    const model = loadModel(args.track);
    const correlations = analyzeCorrelations(model, args);
    correlations.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));

    console.log(
      `Analyzing correlations for track [${args.track.index}] ${args.track.name} (${args.samples} samples${
        args.seed !== null ? `, seed ${args.seed}` : ''
      })`,
    );

    const topCount = Math.min(args.top, correlations.length);
    console.log(`Top ${topCount} correlations (by absolute value):`);

    for (let i = 0; i < topCount; i += 1) {
      const formatted = formatCorrelationEntry(correlations[i]);
      console.log(
        `${String(i + 1).padStart(2, ' ')}. ${formatted.outputName} ↔ ${formatted.featureName} → ${
          formatted.orientation
        } (${formatted.correlation.toFixed(4)})`,
      );
    }
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
  }
}

main();
