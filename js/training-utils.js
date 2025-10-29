const DEFAULT_REWARD_CLAMP_MIN = -1.5;
const DEFAULT_REWARD_CLAMP_MAX = 1.5;
const DEFAULT_SCORE_CLAMP = 2;
const MIN_SAMPLE_WEIGHT = 1e-3;

export function sanitizeFiniteNumber(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

export function clampNumber(value, min, max) {
  if (value < min) {
    return min;
  }
  if (value > max) {
    return max;
  }
  return value;
}

export function ensureFloat32Array(source, size, fallback = 0) {
  let array;
  if (source instanceof Float32Array) {
    array = new Float32Array(source);
  } else if (Array.isArray(source)) {
    array = Float32Array.from(source);
  } else if (typeof source === 'number' && typeof size === 'number' && size > 0) {
    array = new Float32Array(size);
    const value = sanitizeFiniteNumber(source, fallback);
    array.fill(value);
    return array;
  } else if (typeof size === 'number' && size > 0) {
    array = new Float32Array(size);
  } else {
    return new Float32Array();
  }

  if (typeof size === 'number' && size > 0) {
    if (array.length !== size) {
      const resized = new Float32Array(size);
      const count = Math.min(size, array.length);
      for (let i = 0; i < count; i += 1) {
        resized[i] = sanitizeFiniteNumber(array[i], fallback);
      }
      return resized;
    }
  }

  for (let i = 0; i < array.length; i += 1) {
    const value = array[i];
    array[i] = Number.isFinite(value) ? value : fallback;
  }
  return array;
}

export function translateRewardToTargets(sample, outputSize, { clampMin = DEFAULT_REWARD_CLAMP_MIN, clampMax = DEFAULT_REWARD_CLAMP_MAX } = {}) {
  const score = clampNumber(Math.abs(sanitizeFiniteNumber(sample?.score, 0)), 0, DEFAULT_SCORE_CLAMP);
  const magnitude = clampNumber(score, 0, 1);
  const weight = Math.max(magnitude, MIN_SAMPLE_WEIGHT);

  const outputs = ensureFloat32Array(sample?.outputs, outputSize, 0);
  const desired = ensureFloat32Array(sample?.targetOutputs ?? outputs, outputSize, 0);
  const targets = new Float32Array(outputSize);

  for (let i = 0; i < outputSize; i += 1) {
    const from = sanitizeFiniteNumber(outputs[i], 0);
    const to = sanitizeFiniteNumber(desired[i], from);
    const value = from + (to - from) * magnitude;
    targets[i] = clampNumber(value, clampMin, clampMax);
  }

  return { targets, weight };
}

export function normalizeInto(features, mean, invStd, out) {
  const size = Math.min(mean.length, invStd.length);
  const target = out instanceof Float32Array && out.length >= size ? out : new Float32Array(size);
  for (let i = 0; i < size; i += 1) {
    const raw = sanitizeFiniteNumber(i < features.length ? features[i] : 0, 0);
    const centered = raw - sanitizeFiniteNumber(mean[i], 0);
    const normalized = centered * sanitizeFiniteNumber(invStd[i], 1);
    target[i] = sanitizeFiniteNumber(normalized, 0);
  }
  return target;
}

export function prepareReinforcementBatch(samples, shape, options = {}) {
  const prepared = [];
  const transfers = [];
  if (!Array.isArray(samples) || !shape || !Number.isFinite(shape.inputSize) || !Number.isFinite(shape.outputSize)) {
    return { samples: prepared, transfers, totalWeight: 0 };
  }
  let totalWeight = 0;
  for (const sample of samples) {
    if (!sample) {
      continue;
    }
    const features = ensureFloat32Array(sample.features, shape.inputSize, 0);
    const { targets, weight } = translateRewardToTargets(sample, shape.outputSize, options.targets);
    if (features.length !== shape.inputSize || targets.length !== shape.outputSize) {
      continue;
    }
    const payload = {
      features,
      targets,
      weight,
      score: sanitizeFiniteNumber(sample.score, 0),
      timestamp: sanitizeFiniteNumber(sample.timestamp, 0),
      frameTimestamp: sanitizeFiniteNumber(sample.frameTimestamp, 0),
    };
    prepared.push(payload);
    transfers.push(features.buffer, targets.buffer);
    totalWeight += weight;
  }
  return { samples: prepared, transfers, totalWeight };
}
