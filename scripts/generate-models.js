#!/usr/bin/env node
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { TRACK_MODELS } from './tracks.js';

const INPUT_SIZE = 24;
const HIDDEN_SIZE = 16;
const OUTPUT_SIZE = 11;
const WEIGHT_SCALE = 0.35;
const BIAS_SCALE = 0.25;

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PROJECT_ROOT = dirname(__dirname);
const MODELS_DIR = join(PROJECT_ROOT, 'models');

function randomFloat(scale) {
  const value = (Math.random() * 2 - 1) * scale;
  return Math.fround(value);
}

function createLayer(inputSize, outputSize, activation) {
  const weights = new Array(inputSize * outputSize);
  for (let i = 0; i < weights.length; i += 1) {
    weights[i] = randomFloat(WEIGHT_SCALE);
  }

  const bias = new Array(outputSize);
  for (let i = 0; i < outputSize; i += 1) {
    bias[i] = randomFloat(BIAS_SCALE);
  }

  return { activation, weights, bias };
}

function createModelDefinition(track) {
  const normalization = {
    mean: Array(INPUT_SIZE).fill(0),
    std: Array(INPUT_SIZE).fill(1),
  };

  const layers = [
    createLayer(INPUT_SIZE, HIDDEN_SIZE, 'relu'),
    createLayer(HIDDEN_SIZE, OUTPUT_SIZE, 'tanh'),
  ];

  return {
    input: INPUT_SIZE,
    normalization,
    layers,
    meta: {
      name: track.name,
      slug: track.slug,
      inputs: INPUT_SIZE,
      outputs: OUTPUT_SIZE,
      placeholder: true,
      generated: new Date().toISOString(),
    },
  };
}

function formatModel(model) {
  return `${JSON.stringify(model, null, 2)}\n`;
}

function resolveTracks(args) {
  if (args.length === 0) {
    return TRACK_MODELS;
  }

  const selected = new Map();

  args.forEach((raw) => {
    if (raw === '--') {
      return;
    }
    const token = raw.toLowerCase();
    let track = null;

    if (/^\d+$/.test(token)) {
      const index = Number.parseInt(token, 10);
      track = TRACK_MODELS.find((entry) => entry.index === index);
    }

    if (!track) {
      track = TRACK_MODELS.find((entry) => entry.slug === token || entry.file === token);
    }

    if (!track) {
      throw new Error(`Unknown track reference: ${raw}`);
    }

    selected.set(track.index, track);
  });

  return Array.from(selected.values()).sort((a, b) => a.index - b.index);
}

function main() {
  const args = process.argv.slice(2);
  const tracks = resolveTracks(args);

  if (tracks.length === 0) {
    console.error('No tracks selected.');
    process.exit(1);
  }

  tracks.forEach((track) => {
    const definition = createModelDefinition(track);
    const targetPath = join(MODELS_DIR, track.file);
    writeFileSync(targetPath, formatModel(definition), 'utf8');
    console.log(`Generated model for [${track.index}] ${track.name} → ${track.file}`);
  });
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
}
