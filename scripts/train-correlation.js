#!/usr/bin/env node
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  INPUT_SIZE,
  OUTPUT_SIZE,
  buildDataset,
  createRandom,
  enforceCorrelationCaps,
  evaluateModel,
  initializeModel,
  parseCorrelationArguments,
  trainModel,
} from './correlation-common.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PROJECT_ROOT = dirname(__dirname);
const MODELS_DIR = join(PROJECT_ROOT, 'models');

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
        maxCorrelation: primaryCorrelation.maxCorrelation,
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
            maxCorrelation: correlation.maxCorrelation,
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
    const args = parseCorrelationArguments(process.argv.slice(2), {
      usage:
        'Usage: node scripts/train-correlation.js <track> <feature> <output> [direct|inverse] [max=<0-1>] [<feature> <output> [direct|inverse] [max=<0-1>] ...] [--epochs=400] [--samples=4096] [--rate=0.01] [--seed=42]',
    });
    const random = createRandom(args.seed);

    console.log(`Training track [${args.track.index}] ${args.track.name}`);
    args.correlations.forEach((correlation, index) => {
      const label = index === 0 ? 'Primary' : `Secondary #${index}`;
      console.log(
        `  ${label} → ${correlation.featureName} → ${correlation.outputName} (${correlation.inverse ? 'inverse' : 'direct'}, weight ${correlation.weight.toFixed(2)}, max ${correlation.maxCorrelation.toFixed(2)})`,
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

    enforceCorrelationCaps(args.correlations, evaluation);

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
