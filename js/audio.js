const STORAGE_KEY = 'ln.volume';
const DEFAULT_VOLUME = 0.7;
const RAMP_TIME = 0.05; // seconds for smooth gain transitions.

const EMPTY_FLOAT = new Float32Array(0);
const EMPTY_BYTE = new Uint8Array(0);

const DEFAULT_SAMPLE_RATE = 44100;
const EPSILON = 1e-12;
const DB_TO_LIN = Math.LN10 / 20;
const ROLLOFF_TARGET = 0.85;

const BAND_DEFS = [
  { name: 'sub', min: 0, max: 60 },
  { name: 'bass', min: 60, max: 250 },
  { name: 'lowMid', min: 250, max: 500 },
  { name: 'mid', min: 500, max: 2000 },
  { name: 'high', min: 2000, max: Number.POSITIVE_INFINITY },
];

const BAND_COUNT = BAND_DEFS.length;

const FEATURE_INDEX = Object.freeze({
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

const FEATURE_COUNT = 24;

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

const BAND_VALUE_FEATURES = [
  FEATURE_INDEX.SUB,
  FEATURE_INDEX.BASS,
  FEATURE_INDEX.LOW_MID,
  FEATURE_INDEX.MID,
  FEATURE_INDEX.HIGH,
];

const BAND_DELTA_FEATURES = [
  FEATURE_INDEX.DELTA_SUB,
  FEATURE_INDEX.DELTA_BASS,
  FEATURE_INDEX.DELTA_LOW_MID,
  FEATURE_INDEX.DELTA_MID,
  FEATURE_INDEX.DELTA_HIGH,
];

const BAND_EMA_FEATURES = [
  FEATURE_INDEX.EMA_SUB,
  FEATURE_INDEX.EMA_BASS,
  FEATURE_INDEX.EMA_LOW_MID,
  FEATURE_INDEX.EMA_MID,
  FEATURE_INDEX.EMA_HIGH,
];

const BAND_EMA_MS = 300;
const RMS_EMA_MS = 250;
const FLUX_EMA_MS = 200;

let audioElement = null;
let audioContext = null;
let sourceNode = null;
let gainNode = null;
let analyserNode = null;

let desiredVolume = DEFAULT_VOLUME;
let unlockHandlersBound = false;

let floatFrequencyData = EMPTY_FLOAT;
let byteFrequencyData = EMPTY_BYTE;
let timeDomainData = EMPTY_FLOAT;

let binToBand = new Int8Array(0);
let bandBinCounts = new Uint16Array(0);
let previousSpectrum = new Float32Array(0);

const featureVector = new Float32Array(FEATURE_COUNT);
const bandValues = new Float32Array(BAND_COUNT);
const previousBandValues = new Float32Array(BAND_COUNT);
const bandEma = new Float32Array(BAND_COUNT);

let previousRms = 0;
let emaRms = 0;
let fluxEma = 0;
let lastFeatureTimestamp = 0;
let featuresInitialized = false;

const frameState = {
  frequency: EMPTY_FLOAT,
  frequencyByte: EMPTY_BYTE,
  waveform: EMPTY_FLOAT,
  rms: 0,
  timestamp: 0,
  features: featureVector,
};

/**
 * Clamp value into [0, 1].
 * @param {number} value
 * @returns {number}
 */
function clamp01(value) {
  if (!Number.isFinite(value)) {
    return 0;
  }
  if (value < 0) {
    return 0;
  }
  if (value > 1) {
    return 1;
  }
  return value;
}

/**
 * Clamp value symmetrically into [-limit, limit].
 * @param {number} value
 * @param {number} limit
 * @returns {number}
 */
function clampSigned(value, limit = 1) {
  if (!Number.isFinite(value)) {
    return 0;
  }
  const max = limit > 0 ? limit : 1;
  if (value > max) {
    return max;
  }
  if (value < -max) {
    return -max;
  }
  return value;
}

function getTrackPositionValue() {
  if (!audioElement) {
    return -1;
  }
  const duration = Number(audioElement.duration);
  if (!Number.isFinite(duration) || duration <= 0) {
    return -1;
  }
  const current = Number(audioElement.currentTime);
  if (!Number.isFinite(current) || current <= 0) {
    return -1;
  }
  const ratio = Math.min(Math.max(current / duration, 0), 1);
  const signed = ratio * 2 - 1;
  return clampSigned(signed, 1);
}

function updateTrackPositionFeature() {
  featureVector[FEATURE_INDEX.TRACK_POSITION] = getTrackPositionValue();
}

function readStoredVolume() {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored === null) {
      return DEFAULT_VOLUME;
    }
    const parsed = parseFloat(stored);
    if (Number.isFinite(parsed)) {
      return clamp01(parsed);
    }
  } catch {
    // Ignore storage access issues (private mode, disabled storage, etc.).
  }
  return DEFAULT_VOLUME;
}

function persistVolume(value) {
  try {
    window.localStorage.setItem(STORAGE_KEY, value.toString());
  } catch {
    // Ignore persistence failures and continue; volume will reset next boot.
  }
}

function bindUnlockHandlers() {
  if (!audioElement || unlockHandlersBound) {
    return;
  }
  unlockHandlersBound = true;

  const cleanup = () => {
    ['pointerdown', 'touchstart', 'keydown'].forEach((evt) => {
      window.removeEventListener(evt, unlock, true);
      window.removeEventListener(evt, unlock, false);
    });
    if (audioElement) {
      audioElement.removeEventListener('play', unlock);
    }
  };

  const unlock = () => {
    ensureContext()
      .then(() => {
        cleanup();
      })
      .catch(() => {
        // If resume fails we keep listeners so future gestures can retry.
      });
  };

  // Register listeners both capturing and bubbling to maximize unlock reliability.
  ['pointerdown', 'touchstart', 'keydown'].forEach((evt) => {
    window.addEventListener(evt, unlock, { passive: true });
    window.addEventListener(evt, unlock, { passive: true, capture: true });
  });
  audioElement.addEventListener('play', unlock);
}

function createGraph() {
  if (!audioElement) {
    throw new Error('Audio element not set; call init() first.');
  }
  if (audioContext) {
    return;
  }

  audioContext = new AudioContext();
  sourceNode = audioContext.createMediaElementSource(audioElement);
  gainNode = audioContext.createGain();
  analyserNode = audioContext.createAnalyser();

  analyserNode.fftSize = 2048;
  analyserNode.smoothingTimeConstant = 0.8;

  floatFrequencyData = new Float32Array(analyserNode.frequencyBinCount);
  byteFrequencyData = new Uint8Array(analyserNode.frequencyBinCount);
  timeDomainData = new Float32Array(analyserNode.fftSize);

  frameState.frequency = floatFrequencyData;
  frameState.frequencyByte = byteFrequencyData;
  frameState.waveform = timeDomainData;

  // Route analysis before volume adjustments so diagnostics/activity ignore the UI gain setting.
  sourceNode.connect(analyserNode);
  analyserNode.connect(gainNode);
  gainNode.connect(audioContext.destination);

  initializeFeatureBuffers();
  applyVolume(desiredVolume, true);
}

async function ensureContext() {
  if (!audioElement) {
    throw new Error('Audio element not set; call init() first.');
  }

  if (!audioContext) {
    createGraph();
  }

  if (audioContext.state === 'suspended') {
    await audioContext.resume();
  }

  return audioContext;
}

function applyVolume(value, immediate = false) {
  if (!gainNode || !audioContext) {
    return;
  }
  const time = audioContext.currentTime;
  gainNode.gain.cancelScheduledValues(time);
  if (immediate) {
    gainNode.gain.setValueAtTime(value, time);
  } else {
    gainNode.gain.setTargetAtTime(value, time, RAMP_TIME);
  }
}

function computeRms(buffer) {
  if (!buffer || buffer.length === 0) {
    return 0;
  }
  let sum = 0;
  for (let i = 0; i < buffer.length; i += 1) {
    const sample = buffer[i];
    sum += sample * sample;
  }
  return Math.sqrt(sum / buffer.length);
}

function getSampleRate() {
  if (audioContext && Number.isFinite(audioContext.sampleRate)) {
    return audioContext.sampleRate;
  }
  return DEFAULT_SAMPLE_RATE;
}

function computeAlpha(deltaMs, windowMs) {
  if (!Number.isFinite(deltaMs) || deltaMs <= 0) {
    return 1;
  }
  if (!Number.isFinite(windowMs) || windowMs <= 0) {
    return 1;
  }
  const ratio = -deltaMs / windowMs;
  if (ratio < -50) {
    return 1;
  }
  return 1 - Math.exp(ratio);
}

function resetFeatureHistory() {
  bandValues.fill(0);
  previousBandValues.fill(0);
  bandEma.fill(0);
  previousRms = 0;
  emaRms = 0;
  fluxEma = 0;
  lastFeatureTimestamp = 0;
  featuresInitialized = false;
  featureVector.fill(0);
  updateTrackPositionFeature();
}

function initializeFeatureBuffers() {
  if (!analyserNode) {
    binToBand = new Int8Array(0);
    bandBinCounts = new Uint16Array(0);
    previousSpectrum = new Float32Array(0);
    resetFeatureHistory();
    return;
  }

  const binCount = analyserNode.frequencyBinCount;
  const nextBinToBand = new Int8Array(binCount);
  nextBinToBand.fill(-1);
  const nextCounts = new Uint16Array(BAND_COUNT);
  const sampleRate = getSampleRate();
  const binHz = sampleRate / analyserNode.fftSize;

  for (let band = 0; band < BAND_COUNT; band += 1) {
    const def = BAND_DEFS[band];
    const start = Math.max(0, Math.floor(def.min / binHz));
    const rawEnd = Number.isFinite(def.max) ? Math.ceil(def.max / binHz) : binCount;
    const end = Math.max(start + 1, Math.min(binCount, rawEnd));
    for (let i = start; i < end; i += 1) {
      nextBinToBand[i] = band;
      nextCounts[band] += 1;
    }
  }

  for (let band = 0; band < BAND_COUNT; band += 1) {
    if (nextCounts[band] === 0 && binCount > 0) {
      nextCounts[band] = 1;
    }
  }

  binToBand = nextBinToBand;
  bandBinCounts = nextCounts;
  previousSpectrum = new Float32Array(binCount);
  resetFeatureHistory();
}

function ensureFeatureBuffers() {
  if (!analyserNode) {
    return false;
  }
  const binCount = analyserNode.frequencyBinCount;
  if (binCount === 0) {
    return false;
  }
  if (binToBand.length !== binCount || previousSpectrum.length !== binCount) {
    initializeFeatureBuffers();
  }
  return binToBand.length === binCount && previousSpectrum.length === binCount;
}

function updateFeatures(rms, now) {
  if (!analyserNode || floatFrequencyData.length === 0) {
    return;
  }
  if (!ensureFeatureBuffers()) {
    return;
  }

  const binCount = floatFrequencyData.length;
  const sampleRate = getSampleRate();
  const binHz = sampleRate / analyserNode.fftSize;
  const nyquist = Math.max(sampleRate / 2, 1);

  const deltaMs = lastFeatureTimestamp > 0 ? now - lastFeatureTimestamp : 0;
  const alphaBand = computeAlpha(deltaMs, BAND_EMA_MS);
  const alphaRms = computeAlpha(deltaMs, RMS_EMA_MS);
  const alphaFlux = computeAlpha(deltaMs, FLUX_EMA_MS);

  bandValues.fill(0);

  let totalEnergy = 0;
  let centroidNumerator = 0;
  let logSum = 0;
  let fluxSum = 0;

  for (let i = 0; i < binCount; i += 1) {
    const db = floatFrequencyData[i];
    const magnitude = Number.isFinite(db) ? Math.exp(db * DB_TO_LIN) : 0;

    const prevMag = previousSpectrum[i];
    const diff = magnitude - prevMag;
    if (diff > 0) {
      fluxSum += diff;
    }
    previousSpectrum[i] = magnitude;

    totalEnergy += magnitude;
    centroidNumerator += magnitude * (i * binHz);
    logSum += Math.log(magnitude + EPSILON);

    const bandIndex = binToBand[i];
    if (bandIndex >= 0) {
      bandValues[bandIndex] += magnitude;
    }
  }

  for (let band = 0; band < BAND_COUNT; band += 1) {
    const divisor = bandBinCounts[band] || 1;
    const normalized = clamp01(bandValues[band] / divisor);
    const delta = clampSigned(normalized - previousBandValues[band]);
    previousBandValues[band] = normalized;

    if (!featuresInitialized) {
      bandEma[band] = normalized;
    } else {
      bandEma[band] += alphaBand * (normalized - bandEma[band]);
    }

    featureVector[BAND_VALUE_FEATURES[band]] = normalized;
    featureVector[BAND_DELTA_FEATURES[band]] = delta;
    featureVector[BAND_EMA_FEATURES[band]] = clamp01(bandEma[band]);
  }

  const rmsClamped = clamp01(rms);
  featureVector[FEATURE_INDEX.RMS] = rmsClamped;
  const deltaRms = clampSigned(rmsClamped - previousRms);
  previousRms = rmsClamped;

  if (!featuresInitialized) {
    emaRms = rmsClamped;
  } else {
    emaRms += alphaRms * (rmsClamped - emaRms);
  }
  emaRms = clamp01(emaRms);

  featureVector[FEATURE_INDEX.DELTA_RMS] = deltaRms;
  featureVector[FEATURE_INDEX.EMA_RMS] = emaRms;

  const centroidFreq = totalEnergy > EPSILON ? centroidNumerator / totalEnergy : 0;
  const centroidNorm = clamp01(centroidFreq / nyquist);
  featureVector[FEATURE_INDEX.CENTROID] = centroidNorm;

  let rolloffFreq = 0;
  if (totalEnergy > EPSILON) {
    const target = totalEnergy * ROLLOFF_TARGET;
    let cumulative = 0;
    for (let i = 0; i < binCount; i += 1) {
      cumulative += previousSpectrum[i];
      if (cumulative >= target) {
        rolloffFreq = i * binHz;
        break;
      }
    }
  }
  featureVector[FEATURE_INDEX.ROLL_OFF] = clamp01(rolloffFreq / nyquist);

  let flatness = 0;
  if (binCount > 0) {
    const geometric = Math.exp(logSum / binCount);
    const arithmetic = totalEnergy / binCount;
    flatness = arithmetic > EPSILON ? clamp01(geometric / (arithmetic + EPSILON)) : 0;
  }
  featureVector[FEATURE_INDEX.FLATNESS] = flatness;

  const fluxNormalized = clamp01(fluxSum / (binCount || 1));
  featureVector[FEATURE_INDEX.FLUX] = fluxNormalized;

  if (!featuresInitialized) {
    fluxEma = fluxNormalized;
  } else {
    fluxEma += alphaFlux * (fluxNormalized - fluxEma);
  }
  fluxEma = clamp01(fluxEma);
  featureVector[FEATURE_INDEX.FLUX_EMA] = fluxEma;

  updateTrackPositionFeature();

  featuresInitialized = true;
  lastFeatureTimestamp = now;
}

/**
 * Initialize the audio graph. Must be called once with the shared HTMLAudioElement.
 * An AudioContext is created lazily on the first user interaction.
 * @param {HTMLAudioElement} element
 * @returns {number} The restored or default volume in [0, 1].
 */
export function init(element) {
  if (!(element instanceof HTMLAudioElement)) {
    throw new Error('init() requires an HTMLAudioElement.');
  }
  if (audioElement && audioElement !== element) {
    throw new Error('Audio module already initialized with a different element.');
  }

  audioElement = element;
  audioElement.volume = 1;
  desiredVolume = readStoredVolume();

  bindUnlockHandlers();

  return desiredVolume;
}

/**
 * Set gain volume and persist under ln.volume.
 * @param {number} value Linear gain in [0, 1].
 */
export function setVolume(value) {
  desiredVolume = clamp01(value);
  persistVolume(desiredVolume);
  applyVolume(desiredVolume);
}

/**
 * Current target volume.
 * @returns {number}
 */
export function getVolume() {
  return desiredVolume;
}

/**
 * Get the analyser node once the graph has been created.
 * @returns {AnalyserNode|null}
 */
export function getAnalyser() {
  return analyserNode;
}

/**
 * Access the current feature vector (length 24).
 * @returns {Float32Array}
 */
export function getFeatureVector() {
  return featureVector;
}

/**
 * Retrieve human-readable labels for the feature vector indices.
 * @returns {string[]}
 */
export function getFeatureLabels() {
  return FEATURE_LABELS.slice();
}

/**
 * Update cached analyser data. Safe to call before initialization.
 * @returns {{frequency: Float32Array, frequencyByte: Uint8Array, waveform: Float32Array, rms: number, timestamp: number, features: Float32Array}}
 */
export function frame() {
  const now = performance.now();

  if (!analyserNode) {
    frameState.timestamp = now;
    updateTrackPositionFeature();
    return frameState;
  }

  analyserNode.getFloatFrequencyData(floatFrequencyData);
  analyserNode.getByteFrequencyData(byteFrequencyData);
  analyserNode.getFloatTimeDomainData(timeDomainData);

  const rms = computeRms(timeDomainData);
  frameState.rms = rms;

  updateFeatures(rms, now);
  updateTrackPositionFeature();

  frameState.timestamp = now;
  return frameState;
}

/**
 * Expose a manual unlock helper for other modules (optional future use).
 * @returns {Promise<AudioContext>}
 */
export function unlock() {
  return ensureContext();
}
