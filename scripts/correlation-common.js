import { findTrack, formatTrackList } from './tracks.js';

export const INPUT_SIZE = 24;
export const HIDDEN_SIZE = 16;
export const OUTPUT_SIZE = 11;

export const DEFAULT_EPOCHS = 400;
export const DEFAULT_SAMPLES = 4096;
export const DEFAULT_LEARNING_RATE = 0.01;
export const PRIMARY_WEIGHT = 1;
export const SECONDARY_WEIGHT = 0.5;
export const DEFAULT_MAX_CORRELATION = 0.5;

export const FEATURE_LABELS = Object.freeze([
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

export const OUTPUT_LABELS = Object.freeze([
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

export const FEATURE_TYPES = /** @type {const} */ ({
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

export const FEATURE_INDEX_BY_NAME = new Map(
  FEATURE_LABELS.map((label, index) => [label.toLowerCase(), index]),
);

export const OUTPUT_INDEX_BY_NAME = new Map(
  OUTPUT_LABELS.map((label, index) => [label.toLowerCase(), index]),
);

export const ORIENTATION_TOKENS = new Map([
  ['direct', { inverse: false, orientation: 'direct', orientationSign: 1 }],
  ['positive', { inverse: false, orientation: 'direct', orientationSign: 1 }],
  ['+', { inverse: false, orientation: 'direct', orientationSign: 1 }],
  ['inverse', { inverse: true, orientation: 'inverse', orientationSign: -1 }],
  ['invert', { inverse: true, orientation: 'inverse', orientationSign: -1 }],
  ['negative', { inverse: true, orientation: 'inverse', orientationSign: -1 }],
  ['-', { inverse: true, orientation: 'inverse', orientationSign: -1 }],
]);

export function createRandom(seed) {
  if (seed === null || Number.isNaN(seed)) {
    return () => Math.random();
  }

  let state = (Math.abs(Math.trunc(seed)) || 0x9e3779b9) >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

export function clampSigned(value) {
  if (value < -1) {
    return -1;
  }
  if (value > 1) {
    return 1;
  }
  return value;
}

export function randomSigned(random) {
  return random() * 2 - 1;
}

export function randomPositive(random) {
  return random();
}

export function projectTarget(rawValue, featureType, orientationSign) {
  if (featureType === 'signed') {
    return clampSigned(rawValue * orientationSign);
  }
  const centered = rawValue * 2 - 1;
  return clampSigned(centered * orientationSign);
}

export function buildDataset(
  correlations,
  sampleCount,
  random,
  normalizationOverride = null,
) {
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

  const normalization =
    normalizationOverride !== null
      ? {
          mean: Float32Array.from(normalizationOverride.mean, (value) => Number(value)),
          std: Float32Array.from(normalizationOverride.std, (value) => Number(value)),
        }
      : computeNormalization(samples);

  samples.forEach((sample) => {
    normalizeInto(sample.features, normalization.mean, normalization.std, sample.normalized);
  });

  return { samples, normalization };
}

export function computeNormalization(samples) {
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

export function normalizeInto(source, mean, std, target) {
  for (let i = 0; i < INPUT_SIZE; i += 1) {
    target[i] = (source[i] - mean[i]) / std[i];
  }
  return target;
}

export function initializeModel(random) {
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

export function forwardPass(model, input, scratch) {
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

export function shuffleIndices(length, random) {
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

export function trainModel(model, samples, correlations, options, random) {
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

export function evaluateModel(model, samples, correlations) {
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

export function enforceCorrelationCaps(correlations, evaluation, tolerance = 1e-4) {
  if (!evaluation?.perCorrelation) {
    return;
  }

  const violations = [];

  evaluation.perCorrelation.forEach((metrics, index) => {
    const correlation = correlations[index];
    if (!correlation) {
      return;
    }

    const limit = correlation.maxCorrelation ?? DEFAULT_MAX_CORRELATION;
    const magnitude = Math.abs(metrics?.correlation ?? 0);

    if (Number.isFinite(limit) && limit > 0 && magnitude - limit > tolerance) {
      violations.push({
        featureName: correlation.featureName,
        outputName: correlation.outputName,
        limit,
        magnitude,
      });
    }
  });

  if (violations.length === 0) {
    return;
  }

  const details = violations
    .map(
      (violation) =>
        `${violation.featureName} → ${violation.outputName}: |correlation| ${violation.magnitude.toFixed(4)} > max ${violation.limit.toFixed(4)}`,
    )
    .join('\n');

  const error = new Error(`Model correlations exceed configured maxima.\n${details}`);
  error.violations = violations;
  throw error;
}

export function resolveOrientation(token) {
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

export function parseCorrelationArguments(
  rawArgs,
  {
    usage,
    defaultEpochs = DEFAULT_EPOCHS,
    defaultSamples = DEFAULT_SAMPLES,
    defaultRate = DEFAULT_LEARNING_RATE,
    defaultMaxCorrelation = DEFAULT_MAX_CORRELATION,
  } = {},
) {
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
      usage ||
        'Usage: <track> <feature> <output> [direct|inverse] [<feature> <output> [direct|inverse] ...] [--epochs=400] [--samples=4096] [--rate=0.01] [--seed=42]',
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
      throw new Error('Each correlation requires <feature> <output> [direct|inverse] [max=<0-1>].');
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

    let orientationInfo = ORIENTATION_TOKENS.get('direct');
    let consumed = 2;
    const orientationCandidate = resolveOrientation(rest[index + consumed]);
    if (orientationCandidate) {
      orientationInfo = orientationCandidate;
      consumed += 1;
    }

    let maxCorrelation = defaultMaxCorrelation;
    const maxCandidate = rest[index + consumed];
    if (maxCandidate !== undefined) {
      const parsed = parseMaxCorrelation(maxCandidate);
      if (parsed !== null) {
        maxCorrelation = parsed;
        consumed += 1;
      }
    }

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
      maxCorrelation,
    });

    index += consumed;
  }

  const epochs = options.epochs !== undefined ? Number(options.epochs) : defaultEpochs;
  if (!Number.isFinite(epochs) || epochs <= 0) {
    throw new Error(`Invalid epoch count: ${options.epochs}`);
  }

  const samples = options.samples !== undefined ? Number(options.samples) : defaultSamples;
  if (!Number.isFinite(samples) || samples <= 0) {
    throw new Error(`Invalid sample count: ${options.samples}`);
  }

  const learningRate = options.rate !== undefined ? Number(options.rate) : defaultRate;
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

export function parseMaxCorrelation(token) {
  if (token === undefined || token === null) {
    return null;
  }

  const normalized = String(token).trim();
  if (normalized.length === 0) {
    return null;
  }

  let valueString = normalized;
  if (normalized.includes('=')) {
    const [key, value] = normalized.split('=');
    if (key.toLowerCase() !== 'max' && key.toLowerCase() !== 'maxcorr') {
      return null;
    }
    valueString = value;
  } else if (normalized.startsWith('max:')) {
    valueString = normalized.slice(4);
  } else if (normalized.toLowerCase().startsWith('max')) {
    valueString = normalized.slice(3);
  }

  const value = Number(valueString);
  if (!Number.isFinite(value)) {
    return null;
  }

  if (value <= 0 || value > 1) {
    throw new Error(`Max correlation must be between 0 and 1, received ${token}.`);
  }

  return value;
}
