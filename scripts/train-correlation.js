#!/usr/bin/env node
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { findTrack, formatTrackList } from './tracks.js';

const INPUT_SIZE = 24;
const HIDDEN_SIZE = 16;
const OUTPUT_SIZE = 11;

const DEFAULT_EPOCHS = 400;
const DEFAULT_SAMPLES = 4096;
const DEFAULT_LEARNING_RATE = 0.01;

const FEATURE_LABELS = Object.freeze([
  'sub',
  'bass',
  'lowMid',
  'mid',
  'high',
  'rms',
  'centroid',
  'rollOff',
  'flatness',
  'deltaSub',
  'deltaBass',
  'deltaLowMid',
  'deltaMid',
  'deltaHigh',
  'deltaRms',
  'emaSub',
  'emaBass',
  'emaLowMid',
  'emaMid',
  'emaHigh',
  'emaRms',
  'flux',
  'fluxEma',
  'trackPosition',
]);

const OUTPUT_LABELS = Object.freeze([
  'spawnRate',
  'fieldStrength',
  'cohesion',
  'repelImpulse',
  'trailFade',
  'glow',
  'sizeJitter',
  'hueShift',
  'sparkleDensity',
  'vortexAmount',
  'zoom',
]);

const FEATURE_TYPES = /** @type {const} */ ({
  sub: 'signed',
  bass: 'signed',
  lowMid: 'signed',
  mid: 'signed',
  high: 'signed',
  rms: 'positive',
  centroid: 'positive',
  rollOff: 'positive',
  flatness: 'positive',
  deltaSub: 'signed',
  deltaBass: 'signed',
  deltaLowMid: 'signed',
  deltaMid: 'signed',
  deltaHigh: 'signed',
  deltaRms: 'signed',
  emaSub: 'signed',
  emaBass: 'signed',
  emaLowMid: 'signed',
  emaMid: 'signed',
  emaHigh: 'signed',
  emaRms: 'positive',
  flux: 'positive',
  fluxEma: 'positive',
  trackPosition: 'positive',
});

const FEATURE_INDEX_BY_NAME = new Map(
  FEATURE_LABELS.map((label, index) => [label.toLowerCase(), index]),
);

const OUTPUT_INDEX_BY_NAME = new Map(
  OUTPUT_LABELS.map((label, index) => [label.toLowerCase(), index]),
);

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PROJECT_ROOT = dirname(__dirname);
const MODELS_DIR = join(PROJECT_ROOT, 'models');

/**
 * Create a deterministic random number generator when a seed is provided.
 * @param {number|null} seed
 * @returns {() => number}
 */
function createRandom(seed) {
  if (seed === null || Number.isNaN(seed)) {
    return () => Math.random();
  }

  let state = (Math.abs(Math.trunc(seed)) || 0x9e3779b9) >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function clampSigned(value) {
  if (value < -1) {
    return -1;
  }
  if (value > 1) {
    return 1;
  }
  return value;
}

function randomSigned(random) {
  return random() * 2 - 1;
}

function randomPositive(random) {
  return random();
}

/**
 * Convert a feature value into the training target range of [-1, 1].
 * @param {number} rawValue
 * @param {'signed'|'positive'} featureType
 * @param {number} orientationSign
 */
function projectTarget(rawValue, featureType, orientationSign) {
  if (featureType === 'signed') {
    return clampSigned(rawValue * orientationSign);
  }
  const centered = rawValue * 2 - 1;
  return clampSigned(centered * orientationSign);
}

/**
 * Generate a synthetic dataset that approximates feature distributions.
 * @param {number} featureIndex
 * @param {'signed'|'positive'} featureType
 * @param {number} sampleCount
 * @param {() => number} random
 * @param {number} orientationSign
 */
function buildDataset(featureIndex, featureType, sampleCount, random, orientationSign) {
  const samples = new Array(sampleCount);

  for (let i = 0; i < sampleCount; i += 1) {
    const features = new Float32Array(INPUT_SIZE);
    for (let j = 0; j < INPUT_SIZE; j += 1) {
      const label = FEATURE_LABELS[j];
      const type = FEATURE_TYPES[label] || 'positive';
      features[j] = type === 'signed' ? randomSigned(random) : randomPositive(random);
    }
    const featureValue = features[featureIndex];
    samples[i] = {
      features,
      featureValue,
      normalized: new Float32Array(INPUT_SIZE),
      target: projectTarget(featureValue, featureType, orientationSign),
    };
  }

  const normalization = computeNormalization(samples);
  samples.forEach((sample) => {
    normalizeInto(sample.features, normalization.mean, normalization.std, sample.normalized);
  });

  return { samples, normalization };
}

/**
 * Compute normalization statistics for the dataset.
 * @param {{features: Float32Array}[]} samples
 */
function computeNormalization(samples) {
  const mean = new Float64Array(INPUT_SIZE);
  const variance = new Float64Array(INPUT_SIZE);
  const count = samples.length || 1;

  samples.forEach((sample) => {
    const { features } = sample;
    for (let i = 0; i < INPUT_SIZE; i += 1) {
      mean[i] += features[i];
    }
  });

  for (let i = 0; i < INPUT_SIZE; i += 1) {
    mean[i] /= count;
  }

  samples.forEach((sample) => {
    const { features } = sample;
    for (let i = 0; i < INPUT_SIZE; i += 1) {
      const diff = features[i] - mean[i];
      variance[i] += diff * diff;
    }
  });

  const std = new Float64Array(INPUT_SIZE);
  for (let i = 0; i < INPUT_SIZE; i += 1) {
    const value = Math.sqrt(variance[i] / count);
    std[i] = Number.isFinite(value) && value > 1e-6 ? value : 1;
  }

  return {
    mean: Float32Array.from(mean, (v) => Number(v)),
    std: Float32Array.from(std, (v) => Number(v)),
  };
}

/**
 * Normalize a feature vector into the provided target array.
 * @param {Float32Array} source
 * @param {Float32Array} mean
 * @param {Float32Array} std
 * @param {Float32Array} target
 */
function normalizeInto(source, mean, std, target) {
  for (let i = 0; i < INPUT_SIZE; i += 1) {
    target[i] = (source[i] - mean[i]) / std[i];
  }
  return target;
}

function initializeModel(random) {
  const layer1 = {
    weights: new Float32Array(HIDDEN_SIZE * INPUT_SIZE),
    biases: new Float32Array(HIDDEN_SIZE),
  };
  const layer2 = {
    weights: new Float32Array(OUTPUT_SIZE * HIDDEN_SIZE),
    biases: new Float32Array(OUTPUT_SIZE),
  };

  const scale1 = Math.sqrt(2 / INPUT_SIZE);
  const scale2 = Math.sqrt(2 / HIDDEN_SIZE);

  for (let i = 0; i < layer1.weights.length; i += 1) {
    layer1.weights[i] = randomSigned(random) * scale1;
  }
  layer1.biases.fill(0);

  for (let i = 0; i < layer2.weights.length; i += 1) {
    layer2.weights[i] = randomSigned(random) * scale2;
  }
  layer2.biases.fill(0);

  return { layer1, layer2 };
}

function forwardPass(model, input, scratch) {
  const { layer1, layer2 } = model;
  const { hiddenLinear, hiddenActivation, outputLinear, outputs } = scratch;

  for (let h = 0; h < HIDDEN_SIZE; h += 1) {
    let sum = layer1.biases[h];
    const weightOffset = h * INPUT_SIZE;
    for (let i = 0; i < INPUT_SIZE; i += 1) {
      sum += layer1.weights[weightOffset + i] * input[i];
    }
    hiddenLinear[h] = sum;
    hiddenActivation[h] = sum > 0 ? sum : 0;
  }

  for (let o = 0; o < OUTPUT_SIZE; o += 1) {
    let sum = layer2.biases[o];
    const weightOffset = o * HIDDEN_SIZE;
    for (let h = 0; h < HIDDEN_SIZE; h += 1) {
      sum += layer2.weights[weightOffset + h] * hiddenActivation[h];
    }
    outputLinear[o] = sum;
    outputs[o] = Math.tanh(sum);
  }

  return outputs;
}

function shuffleIndices(length, random) {
  const indices = new Array(length);
  for (let i = 0; i < length; i += 1) {
    indices[i] = i;
  }
  for (let i = length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    const temp = indices[i];
    indices[i] = indices[j];
    indices[j] = temp;
  }
  return indices;
}

function trainModel(model, samples, outputIndex, options, random) {
  const { epochs, learningRate } = options;
  const scratch = {
    hiddenLinear: new Float32Array(HIDDEN_SIZE),
    hiddenActivation: new Float32Array(HIDDEN_SIZE),
    outputLinear: new Float32Array(OUTPUT_SIZE),
    outputs: new Float32Array(OUTPUT_SIZE),
    gradW2: new Float32Array(HIDDEN_SIZE),
    gradHidden: new Float32Array(HIDDEN_SIZE),
  };

  let finalLoss = 0;

  for (let epoch = 0; epoch < epochs; epoch += 1) {
    const order = shuffleIndices(samples.length, random);
    let epochLoss = 0;

    for (let idx = 0; idx < order.length; idx += 1) {
      const sample = samples[order[idx]];
      const outputs = forwardPass(model, sample.normalized, scratch);
      const prediction = outputs[outputIndex];
      const error = prediction - sample.target;
      epochLoss += 0.5 * error * error;

      const deltaOut = error * (1 - prediction * prediction);
      const w2Offset = outputIndex * HIDDEN_SIZE;

      for (let h = 0; h < HIDDEN_SIZE; h += 1) {
        scratch.gradW2[h] = deltaOut * scratch.hiddenActivation[h];
      }

      for (let h = 0; h < HIDDEN_SIZE; h += 1) {
        let grad = model.layer2.weights[w2Offset + h] * deltaOut;
        if (scratch.hiddenLinear[h] <= 0) {
          grad = 0;
        }
        scratch.gradHidden[h] = grad;
      }

      for (let h = 0; h < HIDDEN_SIZE; h += 1) {
        model.layer2.weights[w2Offset + h] -= learningRate * scratch.gradW2[h];
      }
      model.layer2.biases[outputIndex] -= learningRate * deltaOut;

      for (let h = 0; h < HIDDEN_SIZE; h += 1) {
        const grad = scratch.gradHidden[h];
        if (grad === 0) {
          continue;
        }
        const weightOffset = h * INPUT_SIZE;
        for (let i = 0; i < INPUT_SIZE; i += 1) {
          model.layer1.weights[weightOffset + i] -= learningRate * grad * sample.normalized[i];
        }
        model.layer1.biases[h] -= learningRate * grad;
      }
    }

    finalLoss = epochLoss / samples.length;

    if (epoch === 0 || (epoch + 1) % 50 === 0 || epoch === epochs - 1) {
      console.log(`Epoch ${epoch + 1}/${epochs} - loss: ${finalLoss.toFixed(6)}`);
    }
  }

  return { loss: finalLoss };
}

function evaluateModel(model, samples, outputIndex, featureIndex, orientationSign) {
  const scratch = {
    hiddenLinear: new Float32Array(HIDDEN_SIZE),
    hiddenActivation: new Float32Array(HIDDEN_SIZE),
    outputLinear: new Float32Array(OUTPUT_SIZE),
    outputs: new Float32Array(OUTPUT_SIZE),
  };

  let sumFeature = 0;
  let sumOutput = 0;
  let sumFeatureSq = 0;
  let sumOutputSq = 0;
  let sumFeatureOutput = 0;
  let mse = 0;

  samples.forEach((sample) => {
    const outputs = forwardPass(model, sample.normalized, scratch);
    const prediction = outputs[outputIndex];
    const featureValue = sample.featureValue;

    sumFeature += featureValue;
    sumOutput += prediction;
    sumFeatureSq += featureValue * featureValue;
    sumOutputSq += prediction * prediction;
    sumFeatureOutput += featureValue * prediction;

    const error = prediction - sample.target;
    mse += error * error;
  });

  const n = samples.length || 1;
  mse /= n;
  const numerator = n * sumFeatureOutput - sumFeature * sumOutput;
  const denomFeature = n * sumFeatureSq - sumFeature * sumFeature;
  const denomOutput = n * sumOutputSq - sumOutput * sumOutput;
  const denominator = Math.sqrt(Math.max(denomFeature, 0) * Math.max(denomOutput, 0));
  const correlation = denominator > 0 ? numerator / denominator : 0;
  const fitness = correlation * orientationSign;

  return {
    correlation,
    fitness,
    mse,
  };
}

function parseArguments(rawArgs) {
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

  if (positionals.length < 3) {
    throw new Error(
      'Usage: node scripts/train-correlation.js <track> <feature> <output> [inverse|direct] [--epochs=400] [--samples=4096] [--rate=0.01] [--seed=42]',
    );
  }

  const [trackRef, featureRef, outputRef] = positionals;
  const orientationToken = positionals[3] || 'direct';

  const track = findTrack(trackRef);
  if (!track) {
    throw new Error(`Unknown track reference "${trackRef}". Available: ${formatTrackList()}`);
  }

  const featureIndex = FEATURE_INDEX_BY_NAME.get(String(featureRef).toLowerCase());
  if (featureIndex === undefined) {
    throw new Error(
      `Unknown feature "${featureRef}". Choose from: ${FEATURE_LABELS.join(', ')}`,
    );
  }
  const featureName = FEATURE_LABELS[featureIndex];
  const featureType = FEATURE_TYPES[featureName] || 'positive';

  const outputIndex = OUTPUT_INDEX_BY_NAME.get(String(outputRef).toLowerCase());
  if (outputIndex === undefined) {
    throw new Error(
      `Unknown output "${outputRef}". Choose from: ${OUTPUT_LABELS.join(', ')}`,
    );
  }
  const outputName = OUTPUT_LABELS[outputIndex];

  const orientation = String(orientationToken).toLowerCase();
  const inverse = orientation === 'inverse' || orientation === 'invert' || orientation === 'negative' || orientation === '-';
  const orientationSign = inverse ? -1 : 1;

  const epochs = options.epochs !== undefined ? Number(options.epochs) : DEFAULT_EPOCHS;
  if (!Number.isFinite(epochs) || epochs <= 0) {
    throw new Error(`Invalid epoch count: ${options.epochs}`);
  }

  const samples = options.samples !== undefined ? Number(options.samples) : DEFAULT_SAMPLES;
  if (!Number.isFinite(samples) || samples <= 0) {
    throw new Error(`Invalid sample count: ${options.samples}`);
  }

  const learningRate = options.rate !== undefined ? Number(options.rate) : DEFAULT_LEARNING_RATE;
  if (!Number.isFinite(learningRate) || learningRate <= 0) {
    throw new Error(`Invalid learning rate: ${options.rate}`);
  }

  const seed = options.seed !== undefined ? Number(options.seed) : null;

  return {
    track,
    featureIndex,
    featureName,
    featureType,
    outputIndex,
    outputName,
    inverse,
    orientationSign,
    epochs: Math.trunc(epochs),
    samples: Math.trunc(samples),
    learningRate,
    seed,
  };
}

function serializeModel(model, normalization, track, details, stats) {
  const { featureName, outputName, inverse, featureType, epochs, samples, learningRate } = details;
  const { loss, evaluation } = stats;

  return {
    input: INPUT_SIZE,
    normalization: {
      mean: Array.from(normalization.mean, (value) => Number(Math.fround(value))),
      std: Array.from(normalization.std, (value) => Number(Math.fround(value))),
    },
    layers: [
      {
        activation: 'relu',
        weights: Array.from(model.layer1.weights, (value) => Number(Math.fround(value))),
        bias: Array.from(model.layer1.biases, (value) => Number(Math.fround(value))),
      },
      {
        activation: 'tanh',
        weights: Array.from(model.layer2.weights, (value) => Number(Math.fround(value))),
        bias: Array.from(model.layer2.biases, (value) => Number(Math.fround(value))),
      },
    ],
    meta: {
      name: track.name,
      slug: track.slug,
      inputs: INPUT_SIZE,
      outputs: OUTPUT_SIZE,
      trackIndex: track.index,
      file: track.file,
      trained: new Date().toISOString(),
      training: {
        feature: featureName,
        featureType,
        output: outputName,
        orientation: inverse ? 'inverse' : 'direct',
        epochs,
        samples,
        learningRate,
        loss: Number(Math.fround(loss)),
        correlation: Number(Math.fround(evaluation.correlation)),
        fitness: Number(Math.fround(evaluation.fitness)),
        mse: Number(Math.fround(evaluation.mse)),
      },
    },
  };
}

function main() {
  try {
    const args = parseArguments(process.argv.slice(2));
    const random = createRandom(args.seed);

    console.log(`Training track [${args.track.index}] ${args.track.name}`);
    console.log(`  Feature → ${args.featureName} (${args.inverse ? 'inverse' : 'direct'})`);
    console.log(`  Output  → ${args.outputName}`);
    console.log(
      `  Samples: ${args.samples}, Epochs: ${args.epochs}, Learning rate: ${args.learningRate}${
        args.seed !== null ? `, Seed: ${args.seed}` : ''
      }`,
    );

    const dataset = buildDataset(
      args.featureIndex,
      args.featureType,
      args.samples,
      random,
      args.orientationSign,
    );
    const model = initializeModel(random);
    const training = trainModel(
      model,
      dataset.samples,
      args.outputIndex,
      { epochs: args.epochs, learningRate: args.learningRate },
      random,
    );
    const evaluation = evaluateModel(
      model,
      dataset.samples,
      args.outputIndex,
      args.featureIndex,
      args.orientationSign,
    );

    console.log(`Final loss: ${training.loss.toFixed(6)}`);
    console.log(`Correlation: ${evaluation.correlation.toFixed(4)}`);
    console.log(`Fitness: ${evaluation.fitness.toFixed(4)}`);
    console.log(`MSE: ${evaluation.mse.toFixed(6)}`);

    const definition = serializeModel(model, dataset.normalization, args.track, args, {
      loss: training.loss,
      evaluation,
    });

    const targetPath = join(MODELS_DIR, args.track.file);
    writeFileSync(targetPath, `${JSON.stringify(definition, null, 2)}\n`, 'utf8');
    console.log(`Saved model → ${targetPath}`);
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
  }
}

main();
