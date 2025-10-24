// Audio feature extraction utilities shared between realtime playback and offline BYOM analysis.

const DEFAULT_BAND_DEFS = Object.freeze([
  { name: 'sub', min: 0, max: 60 },
  { name: 'bass', min: 60, max: 250 },
  { name: 'lowMid', min: 250, max: 500 },
  { name: 'mid', min: 500, max: 2000 },
  { name: 'high', min: 2000, max: Number.POSITIVE_INFINITY },
]);

const DEFAULT_BAND_GAINS = Object.freeze([
  20,
  50,
  100,
  150,
  200,
]);

export const FEATURE_INDEX = Object.freeze({
  SUB: 0,
  BASS: 1,
  LOW_MID: 2,
  MID: 3,
  HIGH: 4,
  RMS: 5,
  CENTROID: 6,
  ROLL_OFF: 7,
  FLATNESS: 8,
  DELTA_SUB: 9,
  DELTA_BASS: 10,
  DELTA_LOW_MID: 11,
  DELTA_MID: 12,
  DELTA_HIGH: 13,
  DELTA_RMS: 14,
  EMA_SUB: 15,
  EMA_BASS: 16,
  EMA_LOW_MID: 17,
  EMA_MID: 18,
  EMA_HIGH: 19,
  EMA_RMS: 20,
  FLUX: 21,
  FLUX_EMA: 22,
  TRACK_POSITION: 23,
});

export const FEATURE_COUNT = 24;

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

const BAND_EMA_MS = 300;
const RMS_EMA_MS = 250;
const FLUX_EMA_MS = 200;
const EPSILON = 1e-12;

function clamp01(value) {
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

function clampSigned(value, limit = 1) {
  if (!Number.isFinite(value)) {
    return 0;
  }
  const bound = limit > 0 ? limit : 1;
  if (value > bound) {
    return bound;
  }
  if (value < -bound) {
    return -bound;
  }
  return value;
}

function computeRms(buffer) {
  if (!buffer || buffer.length === 0) {
    return 0;
  }
  let sum = 0;
  for (let i = 0; i < buffer.length; i += 1) {
    const value = buffer[i];
    sum += value * value;
  }
  const mean = sum / buffer.length;
  return mean > 0 ? Math.sqrt(mean) : 0;
}

function computeAlpha(deltaMs, targetMs) {
  if (!Number.isFinite(deltaMs) || deltaMs <= 0) {
    return 0;
  }
  if (!Number.isFinite(targetMs) || targetMs <= 0) {
    return 1;
  }
  const alpha = 1 - Math.exp(-deltaMs / targetMs);
  if (!Number.isFinite(alpha)) {
    return 1;
  }
  if (alpha < 0) {
    return 0;
  }
  if (alpha > 1) {
    return 1;
  }
  return alpha;
}

function buildBandLut(sampleRate, fftSize, bandDefs) {
  const binCount = Math.max(1, Math.floor(fftSize / 2));
  const binToBand = new Int8Array(binCount);
  binToBand.fill(-1);
  const bandBinCounts = new Uint16Array(bandDefs.length);
  const binHz = sampleRate / fftSize;

  for (let band = 0; band < bandDefs.length; band += 1) {
    const def = bandDefs[band];
    const start = Math.max(0, Math.floor(def.min / binHz));
    const rawEnd = Number.isFinite(def.max) ? Math.ceil(def.max / binHz) : binCount;
    const end = Math.max(start + 1, Math.min(binCount, rawEnd));
    for (let i = start; i < end; i += 1) {
      binToBand[i] = band;
      bandBinCounts[band] += 1;
    }
  }

  for (let band = 0; band < bandBinCounts.length; band += 1) {
    if (bandBinCounts[band] === 0) {
      bandBinCounts[band] = 1;
    }
  }

  return { binToBand, bandBinCounts, binHz };
}

export function createFeatureExtractor({
  sampleRate = 44100,
  fftSize = 2048,
  bandDefs = DEFAULT_BAND_DEFS,
  bandGains = DEFAULT_BAND_GAINS,
} = {}) {
  const bandCount = bandDefs.length;
  const vector = new Float32Array(FEATURE_COUNT);
  const bandValues = new Float32Array(bandCount);
  const previousBandValues = new Float32Array(bandCount);
  const bandEma = new Float32Array(bandCount);
  let { binToBand, bandBinCounts, binHz } = buildBandLut(sampleRate, fftSize, bandDefs);
  let previousSpectrum = new Float32Array(binToBand.length);
  let previousRms = 0;
  let emaRms = 0;
  let fluxEma = 0;
  let initialized = false;

  function reset() {
    bandValues.fill(0);
    previousBandValues.fill(0);
    bandEma.fill(0);
    previousSpectrum.fill(0);
    vector.fill(0);
    previousRms = 0;
    emaRms = 0;
    fluxEma = 0;
    initialized = false;
  }

  function reconfigure(nextSampleRate, nextFftSize) {
    sampleRate = nextSampleRate;
    fftSize = nextFftSize;
    const mapping = buildBandLut(sampleRate, fftSize, bandDefs);
    binToBand = mapping.binToBand;
    bandBinCounts = mapping.bandBinCounts;
    binHz = mapping.binHz;
    previousSpectrum = new Float32Array(binToBand.length);
    reset();
  }

  function ensureMagnitudesLength(length) {
    if (length !== previousSpectrum.length) {
      reconfigure(sampleRate, fftSize);
      if (previousSpectrum.length !== length) {
        previousSpectrum = new Float32Array(length);
        bandValues.fill(0);
        previousBandValues.fill(0);
        bandEma.fill(0);
        fluxEma = 0;
        previousRms = 0;
        emaRms = 0;
        initialized = false;
      }
    }
  }

  function process({
    magnitudes,
    frequencyDb,
    waveform,
    deltaMs,
    trackPosition = -1,
    sampleRateOverride,
    fftSizeOverride,
  } = {}) {
    if (Number.isFinite(sampleRateOverride) && sampleRateOverride > 0) {
      sampleRate = sampleRateOverride;
    }
    if (Number.isFinite(fftSizeOverride) && fftSizeOverride > 0) {
      fftSize = fftSizeOverride;
    }

    const sourceMagnitudes = (() => {
      if (magnitudes && magnitudes.length > 0) {
        return magnitudes;
      }
      if (frequencyDb && frequencyDb.length > 0) {
        const converted = new Float32Array(frequencyDb.length);
        const scale = Math.LN10 / 20;
        for (let i = 0; i < frequencyDb.length; i += 1) {
          const db = frequencyDb[i];
          converted[i] = Number.isFinite(db) ? Math.exp(db * scale) : 0;
        }
        return converted;
      }
      return null;
    })();

    if (!sourceMagnitudes) {
      if (Number.isFinite(trackPosition)) {
        vector[FEATURE_INDEX.TRACK_POSITION] = trackPosition;
      }
      return vector;
    }

    ensureMagnitudesLength(sourceMagnitudes.length);

    const binCount = sourceMagnitudes.length;
    const nyquist = Math.max(sampleRate / 2, 1);
    const clampedDeltaMs = Number.isFinite(deltaMs) && deltaMs > 0 ? deltaMs : 16.67;
    const alphaBand = computeAlpha(clampedDeltaMs, BAND_EMA_MS);
    const alphaRms = computeAlpha(clampedDeltaMs, RMS_EMA_MS);
    const alphaFlux = computeAlpha(clampedDeltaMs, FLUX_EMA_MS);

    bandValues.fill(0);

    let totalEnergy = 0;
    let centroidNumerator = 0;
    let logSum = 0;
    let fluxSum = 0;

    for (let i = 0; i < binCount; i += 1) {
      const magnitude = Number.isFinite(sourceMagnitudes[i]) && sourceMagnitudes[i] > 0 ? sourceMagnitudes[i] : 0;
      const previous = previousSpectrum[i];
      const diff = magnitude - previous;
      if (diff > 0) {
        fluxSum += diff;
      }
      previousSpectrum[i] = magnitude;
      totalEnergy += magnitude;
      centroidNumerator += magnitude * (i * binHz);
      logSum += Math.log(magnitude + EPSILON);

      const bandIndex = i < binToBand.length ? binToBand[i] : -1;
      if (bandIndex >= 0) {
        bandValues[bandIndex] += magnitude;
      }
    }

    for (let band = 0; band < bandValues.length; band += 1) {
      const divisor = bandBinCounts[band] || 1;
      const average = bandValues[band] / divisor;
      const amplified = average * bandGains[band];
      const bounded = clamp01(amplified);
      const normalized = clampSigned(bounded * 2 - 1);
      const delta = clampSigned(normalized - previousBandValues[band]);
      previousBandValues[band] = normalized;

      if (!initialized) {
        bandEma[band] = normalized;
      } else {
        bandEma[band] += alphaBand * (normalized - bandEma[band]);
      }

      vector[band] = normalized;
      vector[FEATURE_INDEX.DELTA_SUB + band] = delta;
      vector[FEATURE_INDEX.EMA_SUB + band] = clampSigned(bandEma[band]);
    }

    const rms = waveform ? clamp01(computeRms(waveform)) : 0;
    vector[FEATURE_INDEX.RMS] = rms;
    const deltaRms = clampSigned(rms - previousRms);
    previousRms = rms;

    if (!initialized) {
      emaRms = rms;
    } else {
      emaRms += alphaRms * (rms - emaRms);
    }
    emaRms = clamp01(emaRms);

    vector[FEATURE_INDEX.DELTA_RMS] = deltaRms;
    vector[FEATURE_INDEX.EMA_RMS] = emaRms;

    const centroidFreq = totalEnergy > EPSILON ? centroidNumerator / totalEnergy : 0;
    vector[FEATURE_INDEX.CENTROID] = clamp01(centroidFreq / nyquist);

    let rollOffFrequency = 0;
    if (totalEnergy > EPSILON) {
      const target = totalEnergy * 0.85;
      let cumulative = 0;
      for (let i = 0; i < binCount; i += 1) {
        cumulative += previousSpectrum[i];
        if (cumulative >= target) {
          rollOffFrequency = i * binHz;
          break;
        }
      }
    }
    vector[FEATURE_INDEX.ROLL_OFF] = clamp01(rollOffFrequency / nyquist);

    let flatness = 0;
    if (binCount > 0) {
      const geometric = Math.exp(logSum / binCount);
      const arithmetic = totalEnergy / binCount;
      flatness = arithmetic > EPSILON ? clamp01(geometric / (arithmetic + EPSILON)) : 0;
    }
    vector[FEATURE_INDEX.FLATNESS] = flatness;

    const flux = clamp01(fluxSum / (binCount || 1));
    vector[FEATURE_INDEX.FLUX] = flux;
    if (!initialized) {
      fluxEma = flux;
    } else {
      fluxEma += alphaFlux * (flux - fluxEma);
    }
    vector[FEATURE_INDEX.FLUX_EMA] = clamp01(fluxEma);

    if (Number.isFinite(trackPosition)) {
      vector[FEATURE_INDEX.TRACK_POSITION] = trackPosition;
    }

    initialized = true;
    return vector;
  }

  return {
    process,
    reset,
    setTrackPosition(value) {
      vector[FEATURE_INDEX.TRACK_POSITION] = Number.isFinite(value) ? value : -1;
    },
    getVector() {
      return vector;
    },
    getSampleRate() {
      return sampleRate;
    },
    getFftSize() {
      return fftSize;
    },
  };
}

export function computeTrackPosition(currentTime, duration) {
  if (!Number.isFinite(currentTime) || !Number.isFinite(duration) || duration <= 0) {
    return -1;
  }
  const ratio = currentTime / duration;
  if (!Number.isFinite(ratio)) {
    return -1;
  }
  if (ratio < 0) {
    return 0;
  }
  if (ratio > 1) {
    return 1;
  }
  return ratio;
}

export function mixToMono(buffer) {
  if (!buffer || buffer.length === 0) {
    return new Float32Array(0);
  }
  const length = buffer[0].length;
  const channels = buffer.length;
  const mixed = new Float32Array(length);
  for (let channel = 0; channel < channels; channel += 1) {
    const data = buffer[channel];
    for (let i = 0; i < length; i += 1) {
      mixed[i] += data[i];
    }
  }
  if (channels > 0) {
    const invChannels = 1 / channels;
    for (let i = 0; i < length; i += 1) {
      mixed[i] *= invChannels;
    }
  }
  return mixed;
}

export { computeRms };
