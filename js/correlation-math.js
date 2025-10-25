import { FEATURE_TYPES } from './audio-features.js';

export const PRIMARY_WEIGHT = 1;
export const SECONDARY_WEIGHT = 0.5;

export function clampSigned(value) {
  if (!Number.isFinite(value)) {
    return 0;
  }
  if (value < -1) {
    return -1;
  }
  if (value > 1) {
    return 1;
  }
  return value;
}

export function clamp01(value) {
  if (!Number.isFinite(value)) {
    return 0;
  }
  if (value <= 0) {
    return 0;
  }
  if (value >= 1) {
    return 1;
  }
  return value;
}

export function resolveFeatureType(featureName) {
  const type = FEATURE_TYPES[featureName];
  return type === 'signed' ? 'signed' : 'positive';
}

export function projectFeatureValue(rawValue, featureType, orientationSign = 1) {
  const type = featureType === 'signed' ? 'signed' : 'positive';
  const sign = orientationSign === -1 ? -1 : 1;
  if (type === 'signed') {
    return clampSigned(rawValue * sign);
  }
  const centered = clamp01(rawValue) * 2 - 1;
  return clampSigned(centered * sign);
}

export function formatCorrelation(value) {
  if (!Number.isFinite(value)) {
    return '—';
  }
  const magnitude = Math.abs(value);
  if (magnitude >= 0.9995) {
    return value.toFixed(4);
  }
  if (magnitude >= 0.1) {
    return value.toFixed(3);
  }
  return value.toFixed(4);
}
