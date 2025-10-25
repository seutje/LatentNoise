import { FEATURE_LABELS } from './audio-features.js';
import { PARAM_NAMES } from './map.js';

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
  trackPosition: 'signed',
});

function formatLabel(name) {
  if (typeof name !== 'string' || name.length === 0) {
    return '';
  }
  return name
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/^[a-z]/, (match) => match.toUpperCase());
}

export const MUSIC_FEATURES = FEATURE_LABELS.map((name, index) => ({
  index,
  name,
  label: formatLabel(name),
  type: FEATURE_TYPES[name] ?? 'positive',
}));

export const SIMULATION_FEATURES = PARAM_NAMES.map((name, index) => ({
  index,
  name,
  label: formatLabel(name),
}));

export function getFeatureByIndex(index) {
  if (!Number.isInteger(index) || index < 0 || index >= MUSIC_FEATURES.length) {
    return null;
  }
  return MUSIC_FEATURES[index];
}

export function getOutputByIndex(index) {
  if (!Number.isInteger(index) || index < 0 || index >= SIMULATION_FEATURES.length) {
    return null;
  }
  return SIMULATION_FEATURES[index];
}

function clampSigned(value) {
  if (!Number.isFinite(value)) {
    return 0;
  }
  if (value > 1) {
    return 1;
  }
  if (value < -1) {
    return -1;
  }
  return value;
}

export function projectFeatureValue(rawValue, featureType, orientationSign) {
  const sign = orientationSign === -1 ? -1 : 1;
  if (featureType === 'signed') {
    return clampSigned(rawValue) * sign;
  }
  const centered = Number.isFinite(rawValue) ? rawValue * 2 - 1 : 0;
  return clampSigned(centered) * sign;
}

export function sanitizeCorrelations(correlations, dataset) {
  if (!Array.isArray(correlations) || correlations.length === 0) {
    return [];
  }
  const featureSize = Number.isFinite(dataset?.featureSize) ? dataset.featureSize : MUSIC_FEATURES.length;
  const targetSize = Number.isFinite(dataset?.targetSize) ? dataset.targetSize : SIMULATION_FEATURES.length;
  return correlations
    .map((correlation) => {
      if (!correlation || typeof correlation !== 'object') {
        return null;
      }
      const featureIndex = Number(correlation.featureIndex);
      const outputIndex = Number(correlation.outputIndex);
      if (!Number.isInteger(featureIndex) || featureIndex < 0 || featureIndex >= featureSize) {
        return null;
      }
      if (!Number.isInteger(outputIndex) || outputIndex < 0 || outputIndex >= targetSize) {
        return null;
      }
      const feature = getFeatureByIndex(featureIndex);
      const output = getOutputByIndex(outputIndex);
      if (!feature || !output) {
        return null;
      }
      const inverse = Boolean(correlation.inverse);
      const orientationSign = inverse ? -1 : 1;
      return {
        id: correlation.id ?? `${featureIndex}:${outputIndex}:${inverse ? 'inv' : 'dir'}`,
        featureIndex,
        featureName: feature.name,
        featureLabel: feature.label,
        featureType: feature.type,
        outputIndex,
        outputName: output.name,
        outputLabel: output.label,
        inverse,
        orientationSign,
      };
    })
    .filter(Boolean);
}

export function applyCorrelationsToTargetBuffer(dataset, correlations, targetBuffer) {
  if (!dataset || !targetBuffer || !Array.isArray(correlations) || correlations.length === 0) {
    return;
  }
  const { features, frameCount, featureSize, targetSize } = dataset;
  if (!(features instanceof Float32Array) || frameCount <= 0) {
    return;
  }
  for (let frame = 0; frame < frameCount; frame += 1) {
    const featureOffset = frame * featureSize;
    const targetOffset = frame * targetSize;
    correlations.forEach((correlation) => {
      const featureValue = features[featureOffset + correlation.featureIndex];
      const projected = projectFeatureValue(featureValue, correlation.featureType, correlation.orientationSign);
      targetBuffer[targetOffset + correlation.outputIndex] = projected;
    });
  }
}

export function formatOrientation(inverse) {
  return inverse ? 'Inverse' : 'Direct';
}

export function formatFeatureLabel(name) {
  return formatLabel(name);
}

export function formatOutputLabel(name) {
  return formatLabel(name);
}
