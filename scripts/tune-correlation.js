#!/usr/bin/env node
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  HIDDEN_SIZE,
  INPUT_SIZE,
  OUTPUT_SIZE,
  buildDataset,
  createRandom,
  evaluateModel,
  parseCorrelationArguments,
  trainModel,
} from './correlation-common.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PROJECT_ROOT = dirname(__dirname);
const MODELS_DIR = join(PROJECT_ROOT, 'models');

function isArrayLike(value) {
  return Array.isArray(value) || ArrayBuffer.isView(value);
}

function toFloat32ArrayStrict(source, expectedLength, label) {
  if (!isArrayLike(source)) {
    throw new Error(`Model definition is missing ${label}.`);
  }
  const values = Array.from(source, (value) => Number(value));
  if (values.length !== expectedLength) {
    throw new Error(
      `Expected ${label} length ${expectedLength}, received ${values.length}.`,
    );
  }
  const result = new Float32Array(expectedLength);
  for (let i = 0; i < expectedLength; i += 1) {
    const value = values[i];
    result[i] = Number.isFinite(value) ? value : 0;
  }
  return result;
}

function toFloat32Vector(source, expectedLength, fallback) {
  const values = isArrayLike(source)
    ? Array.from(source, (value) => Number(value))
    : [];
  const result = new Float32Array(expectedLength);
  for (let i = 0; i < expectedLength; i += 1) {
    const value = values[i];
    result[i] = Number.isFinite(value) ? value : fallback;
  }
  return result;
}

function cloneMeta(meta) {
  if (!meta || typeof meta !== 'object') {
    return {};
  }
  return JSON.parse(JSON.stringify(meta));
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

  const layers = Array.isArray(raw?.layers) ? raw.layers : null;
  if (!layers || layers.length < 2) {
    throw new Error('Model definition must include at least two layers.');
  }

  const [layer1, layer2] = layers;
  const layer1Weights = toFloat32ArrayStrict(
    layer1?.weights,
    HIDDEN_SIZE * INPUT_SIZE,
    'layer1.weights',
  );
  const layer1Biases = toFloat32ArrayStrict(layer1?.bias ?? layer1?.biases, HIDDEN_SIZE, 'layer1.bias');
  const layer2Weights = toFloat32ArrayStrict(
    layer2?.weights,
    OUTPUT_SIZE * HIDDEN_SIZE,
    'layer2.weights',
  );
  const layer2Biases = toFloat32ArrayStrict(layer2?.bias ?? layer2?.biases, OUTPUT_SIZE, 'layer2.bias');

  const normalizationMean = toFloat32Vector(raw?.normalization?.mean, INPUT_SIZE, 0);
  const normalizationStd = toFloat32Vector(raw?.normalization?.std, INPUT_SIZE, 1);

  const meta = cloneMeta(raw?.meta);
  const previousTraining = meta?.training ? JSON.parse(JSON.stringify(meta.training)) : null;

  return {
    path: targetPath,
    model: {
      layer1: { weights: layer1Weights, biases: layer1Biases },
      layer2: { weights: layer2Weights, biases: layer2Biases },
    },
    normalization: {
      mean: normalizationMean,
      std: normalizationStd,
    },
    meta,
    previousTraining,
  };
}

function serializeTunedModel(
  model,
  normalization,
  track,
  details,
  stats,
  baseMeta,
  previousTraining,
) {
  const { correlations, epochs, samples, learningRate } = details;
  const { loss, evaluation } = stats;
  const primaryCorrelation = correlations[0];
  const primaryEvaluation = evaluation.perCorrelation[0];

  const meta = {
    ...baseMeta,
    name: track.name,
    slug: track.slug,
    inputs: INPUT_SIZE,
    outputs: OUTPUT_SIZE,
    trackIndex: track.index,
    file: track.file,
    trained: new Date().toISOString(),
  };

  const training = {
    mode: 'tune',
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
  };

  if (previousTraining) {
    training.previous = previousTraining;
  }

  meta.training = training;

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
    meta,
  };
}

function main() {
  try {
    const args = parseCorrelationArguments(process.argv.slice(2), {
      usage:
        'Usage: node scripts/tune-correlation.js <track> <feature> <output> [direct|inverse] [<feature> <output> [direct|inverse] ...] [--epochs=400] [--samples=4096] [--rate=0.01] [--seed=42]',
    });
    const random = createRandom(args.seed);

    const context = loadModel(args.track);

    console.log(`Fine-tuning track [${args.track.index}] ${args.track.name}`);
    console.log(`  Starting from model: ${context.path}`);
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

    const dataset = buildDataset(
      args.correlations,
      args.samples,
      random,
      context.normalization,
    );
    const training = trainModel(
      context.model,
      dataset.samples,
      args.correlations,
      { epochs: args.epochs, learningRate: args.learningRate },
      random,
    );
    const evaluation = evaluateModel(context.model, dataset.samples, args.correlations);

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

    const definition = serializeTunedModel(
      context.model,
      context.normalization,
      args.track,
      args,
      { loss: training.loss, evaluation },
      context.meta,
      context.previousTraining,
    );

    writeFileSync(context.path, `${JSON.stringify(definition, null, 2)}\n`, 'utf8');
    console.log(`Updated model → ${context.path}`);
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
  }
}

main();
