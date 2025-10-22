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
const PRIMARY_WEIGHT = 1;
const SECONDARY_WEIGHT = 0.5;

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

const ORIENTATION_TOKENS = new Map([
  ['direct', { inverse: false, orientation: 'direct', orientationSign: 1 }],
  ['positive', { inverse: false, orientation: 'direct', orientationSign: 1 }],
  ['+', { inverse: false, orientation: 'direct', orientationSign: 1 }],
  ['inverse', { inverse: true, orientation: 'inverse', orientationSign: -1 }],
  ['invert', { inverse: true, orientation: 'inverse', orientationSign: -1 }],
  ['negative', { inverse: true, orientation: 'inverse', orientationSign: -1 }],
  ['-', { inverse: true, orientation: 'inverse', orientationSign: -1 }],
]);

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
 * @param {{featureIndex: number, featureType: 'signed'|'positive', orientationSign: number}[]} correlations
 * @param {number} sampleCount
 * @param {() => number} random
 */
function buildDataset(correlations, sampleCount, random) {
  const samples = new Array(sampleCount);

  for (let i = 0; i < sampleCount; i += 1) {
    const features = new Float32Array(INPUT_SIZE);
    for (let j = 0; j < INPUT_SIZE; j += 1) {
      const label = FEATURE_LABELS[j];
      const type = FEATURE_TYPES[label] || 'positive';
      features[j] = type === 'signed' ? randomSigned(random) : randomPositive(random);
    }
    const targets = new Float32Array(correlations.length);
    const featureValues = new Float32Array(correlations.length);
    correlations.forEach((correlation, index) => {
      const featureValue = features[correlation.featureIndex];
      featureValues[index] = featureValue;
      targets[index] = projectTarget(
        featureValue,
        correlation.featureType,
        correlation.orientationSign,
      );
    });
    samples[i] = {
      features,
      featureValues,
      normalized: new Float32Array(INPUT_SIZE),
      targets,
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

function trainModel(model, samples, correlations, options, random) {
  const { epochs, learningRate } = options;
  const scratch = {
    hiddenLinear: new Float32Array(HIDDEN_SIZE),
    hiddenActivation: new Float32Array(HIDDEN_SIZE),
    outputLinear: new Float32Array(OUTPUT_SIZE),
    outputs: new Float32Array(OUTPUT_SIZE),
    gradOutput: new Float32Array(OUTPUT_SIZE),
    gradHidden: new Float32Array(HIDDEN_SIZE),
  };
  const targetedOutputs = Array.from(
    new Set(correlations.map((correlation) => correlation.outputIndex)),
  );

  let finalLoss = 0;

  for (let epoch = 0; epoch < epochs; epoch += 1) {
    const order = shuffleIndices(samples.length, random);
    let epochLoss = 0;

    for (let idx = 0; idx < order.length; idx += 1) {
      const sample = samples[order[idx]];
      const outputs = forwardPass(model, sample.normalized, scratch);
      scratch.gradOutput.fill(0);
      scratch.gradHidden.fill(0);

      let sampleLoss = 0;

      correlations.forEach((correlation, correlationIndex) => {
        const prediction = outputs[correlation.outputIndex];
        const target = sample.targets[correlationIndex];
        const error = prediction - target;
        sampleLoss += 0.5 * correlation.weight * error * error;
        const deltaOut = correlation.weight * error * (1 - prediction * prediction);
        scratch.gradOutput[correlation.outputIndex] += deltaOut;
      });

      targetedOutputs.forEach((outputIndex) => {
        const deltaOut = scratch.gradOutput[outputIndex];
        if (deltaOut === 0) {
          return;
        }
        const w2Offset = outputIndex * HIDDEN_SIZE;
        for (let h = 0; h < HIDDEN_SIZE; h += 1) {
          scratch.gradHidden[h] += model.layer2.weights[w2Offset + h] * deltaOut;
        }
      });

      for (let h = 0; h < HIDDEN_SIZE; h += 1) {
        if (scratch.hiddenLinear[h] <= 0) {
          scratch.gradHidden[h] = 0;
        }
      }

      targetedOutputs.forEach((outputIndex) => {
        const deltaOut = scratch.gradOutput[outputIndex];
        if (deltaOut === 0) {
          return;
        }
        const w2Offset = outputIndex * HIDDEN_SIZE;
        for (let h = 0; h < HIDDEN_SIZE; h += 1) {
          model.layer2.weights[w2Offset + h] -= learningRate * deltaOut * scratch.hiddenActivation[h];
        }
        model.layer2.biases[outputIndex] -= learningRate * deltaOut;
      });

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

      epochLoss += sampleLoss;
    }

    finalLoss = epochLoss / samples.length;

    if (epoch === 0 || (epoch + 1) % 50 === 0 || epoch === epochs - 1) {
      console.log(`Epoch ${epoch + 1}/${epochs} - loss: ${finalLoss.toFixed(6)}`);
    }
  }

  return { loss: finalLoss };
}

function evaluateModel(model, samples, correlations) {
  const scratch = {
    hiddenLinear: new Float32Array(HIDDEN_SIZE),
    hiddenActivation: new Float32Array(HIDDEN_SIZE),
    outputLinear: new Float32Array(OUTPUT_SIZE),
    outputs: new Float32Array(OUTPUT_SIZE),
  };

  const aggregators = correlations.map(() => ({
    sumFeature: 0,
    sumOutput: 0,
    sumFeatureSq: 0,
    sumOutputSq: 0,
    sumFeatureOutput: 0,
    mse: 0,
  }));

  samples.forEach((sample) => {
    const outputs = forwardPass(model, sample.normalized, scratch);
    correlations.forEach((correlation, index) => {
      const prediction = outputs[correlation.outputIndex];
      const featureValue = sample.featureValues[index];
      aggregators[index].sumFeature += featureValue;
      aggregators[index].sumOutput += prediction;
      aggregators[index].sumFeatureSq += featureValue * featureValue;
      aggregators[index].sumOutputSq += prediction * prediction;
      aggregators[index].sumFeatureOutput += featureValue * prediction;
      const error = prediction - sample.targets[index];
      aggregators[index].mse += error * error;
    });
  });

  const n = samples.length || 1;
  const perCorrelation = aggregators.map((aggregator, index) => {
    const numerator =
      n * aggregator.sumFeatureOutput - aggregator.sumFeature * aggregator.sumOutput;
    const denomFeature =
      n * aggregator.sumFeatureSq - aggregator.sumFeature * aggregator.sumFeature;
    const denomOutput =
      n * aggregator.sumOutputSq - aggregator.sumOutput * aggregator.sumOutput;
    const denominator = Math.sqrt(
      Math.max(denomFeature, 0) * Math.max(denomOutput, 0),
    );
    const correlation = denominator > 0 ? numerator / denominator : 0;
    const fitness = correlation * correlations[index].orientationSign;
    return {
      correlation,
      fitness,
      mse: aggregator.mse / n,
    };
  });

  const totalWeight = correlations.reduce((sum, correlation) => sum + correlation.weight, 0);
  const combinedFitness =
    totalWeight > 0
      ? perCorrelation.reduce(
          (sum, metrics, index) => sum + correlations[index].weight * metrics.fitness,
          0,
        ) / totalWeight
      : 0;
  const averageMse =
    perCorrelation.reduce((sum, metrics) => sum + metrics.mse, 0) /
    (perCorrelation.length || 1);

  return {
    perCorrelation,
    combinedFitness,
    averageMse,
  };
}

function resolveOrientation(token) {
  if (token === undefined) {
    return null;
  }
  const normalized = String(token).toLowerCase();
  const info = ORIENTATION_TOKENS.get(normalized);
  if (!info) {
    return null;
  }
  return info;
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
      'Usage: node scripts/train-correlation.js <track> <feature> <output> [direct|inverse] [<feature> <output> [direct|inverse] ...] [--epochs=400] [--samples=4096] [--rate=0.01] [--seed=42]',
    );
  }

  const [trackRef, ...rest] = positionals;
  if (rest.length < 2) {
    throw new Error('At least one <feature> <output> pair is required.');
  }

  const track = findTrack(trackRef);
  if (!track) {
    throw new Error(`Unknown track reference "${trackRef}". Available: ${formatTrackList()}`);
  }

  const correlations = [];
  let index = 0;
  while (index < rest.length) {
    const featureRef = rest[index];
    const outputRef = rest[index + 1];
    if (outputRef === undefined) {
      throw new Error('Each correlation requires <feature> <output> [direct|inverse].');
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

    const orientationCandidate = resolveOrientation(rest[index + 2]);
    const orientationInfo = orientationCandidate || ORIENTATION_TOKENS.get('direct');
    const consumed = orientationCandidate ? 3 : 2;

    correlations.push({
      featureIndex,
      featureName,
      featureType,
      outputIndex,
      outputName,
      inverse: orientationInfo.inverse,
      orientation: orientationInfo.orientation,
      orientationSign: orientationInfo.orientationSign,
      weight: correlations.length === 0 ? PRIMARY_WEIGHT : SECONDARY_WEIGHT,
    });

    index += consumed;
  }

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
    correlations,
    epochs: Math.trunc(epochs),
    samples: Math.trunc(samples),
    learningRate,
    seed,
  };
}

function serializeModel(model, normalization, track, details, stats) {
  const { correlations, epochs, samples, learningRate } = details;
  const { loss, evaluation } = stats;
  const primaryCorrelation = correlations[0];
  const primaryEvaluation = evaluation.perCorrelation[0];

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
        feature: primaryCorrelation.featureName,
        featureType: primaryCorrelation.featureType,
        output: primaryCorrelation.outputName,
        orientation: primaryCorrelation.inverse ? 'inverse' : 'direct',
        weight: primaryCorrelation.weight,
        epochs,
        samples,
        learningRate,
        loss: Number(Math.fround(loss)),
        correlation: Number(Math.fround(primaryEvaluation?.correlation ?? 0)),
        fitness: Number(Math.fround(primaryEvaluation?.fitness ?? 0)),
        mse: Number(Math.fround(primaryEvaluation?.mse ?? 0)),
        combinedFitness: Number(Math.fround(evaluation.combinedFitness ?? 0)),
        averageMse: Number(Math.fround(evaluation.averageMse ?? 0)),
        correlations: correlations.map((correlation, index) => {
          const metrics = evaluation.perCorrelation[index];
          return {
            feature: correlation.featureName,
            featureType: correlation.featureType,
            output: correlation.outputName,
            orientation: correlation.inverse ? 'inverse' : 'direct',
            weight: correlation.weight,
            correlation: Number(Math.fround(metrics?.correlation ?? 0)),
            fitness: Number(Math.fround(metrics?.fitness ?? 0)),
            mse: Number(Math.fround(metrics?.mse ?? 0)),
          };
        }),
      },
    },
  };
}

function main() {
  try {
    const args = parseArguments(process.argv.slice(2));
    const random = createRandom(args.seed);

    console.log(`Training track [${args.track.index}] ${args.track.name}`);
    args.correlations.forEach((correlation, index) => {
      const label = index === 0 ? 'Primary' : `Secondary #${index}`;
      console.log(
        `  ${label} → ${correlation.featureName} → ${correlation.outputName} (${correlation.inverse ? 'inverse' : 'direct'}, weight ${correlation.weight.toFixed(2)})`,
      );
    });
    console.log(
      `  Samples: ${args.samples}, Epochs: ${args.epochs}, Learning rate: ${args.learningRate}${
        args.seed !== null ? `, Seed: ${args.seed}` : ''
      }`,
    );

    const dataset = buildDataset(args.correlations, args.samples, random);
    const model = initializeModel(random);
    const training = trainModel(
      model,
      dataset.samples,
      args.correlations,
      { epochs: args.epochs, learningRate: args.learningRate },
      random,
    );
    const evaluation = evaluateModel(model, dataset.samples, args.correlations);

    console.log(`Final loss: ${training.loss.toFixed(6)}`);
    evaluation.perCorrelation.forEach((metrics, index) => {
      const correlation = args.correlations[index];
      const label = index === 0 ? 'Primary' : `Secondary #${index}`;
      console.log(
        `${label} correlation (${correlation.featureName} → ${correlation.outputName}): ${metrics.correlation.toFixed(4)}`,
      );
      console.log(
        `  Fitness: ${metrics.fitness.toFixed(4)}, MSE: ${metrics.mse.toFixed(6)}`,
      );
    });
    console.log(`Combined fitness: ${evaluation.combinedFitness.toFixed(4)}`);
    console.log(`Average MSE: ${evaluation.averageMse.toFixed(6)}`);

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
