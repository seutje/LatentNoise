import {
  createFeatureExtractor,
  FEATURE_INDEX,
  FEATURE_LABELS,
  computeTrackPosition,
} from './audio-features.js';

const STORAGE_KEY = 'ln.volume';
const DEFAULT_VOLUME = 0.7;
const RAMP_TIME = 0.05; // seconds for smooth gain transitions.

const EMPTY_FLOAT = new Float32Array(0);
const EMPTY_BYTE = new Uint8Array(0);

const DEFAULT_SAMPLE_RATE = 44100;
const RMS_ACTIVITY_FLOOR_DB = -55;
const RMS_ACTIVITY_CEILING_DB = 0;

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

const featureExtractor = createFeatureExtractor({ sampleRate: DEFAULT_SAMPLE_RATE, fftSize: 2048 });
const featureVector = featureExtractor.getVector();
let lastFrameTimestamp = 0;

const frameState = {
  frequency: EMPTY_FLOAT,
  frequencyByte: EMPTY_BYTE,
  waveform: EMPTY_FLOAT,
  rms: 0,
  activity: 0,
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
function getTrackPositionValue() {
  if (!audioElement) {
    return -1;
  }
  const duration = Number(audioElement.duration);
  if (!Number.isFinite(duration) || duration <= 0) {
    return -1;
  }
  const current = Number(audioElement.currentTime);
  if (!Number.isFinite(current) || current < 0) {
    return -1;
  }
  const ratio = computeTrackPosition(current, duration);
  if (!Number.isFinite(ratio) || ratio < 0) {
    return -1;
  }
  return Math.min(Math.max(ratio * 2 - 1, -1), 1);
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
  frameState.features = featureVector;

  // Route analysis before volume adjustments so diagnostics/activity ignore the UI gain setting.
  sourceNode.connect(analyserNode);
  analyserNode.connect(gainNode);
  gainNode.connect(audioContext.destination);

  featureExtractor.reset();
  featureExtractor.setTrackPosition(getTrackPositionValue());
  lastFrameTimestamp = 0;
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

function getSampleRate() {
  if (audioContext && Number.isFinite(audioContext.sampleRate)) {
    return audioContext.sampleRate;
  }
  return DEFAULT_SAMPLE_RATE;
}

function rmsToActivity(rms) {
  if (!Number.isFinite(rms) || rms <= 0) {
    return 0;
  }
  const db = 20 * Math.log10(rms);
  if (db <= RMS_ACTIVITY_FLOOR_DB) {
    return 0;
  }
  if (db >= RMS_ACTIVITY_CEILING_DB) {
    return 1;
  }
  const normalized = (db - RMS_ACTIVITY_FLOOR_DB) / (RMS_ACTIVITY_CEILING_DB - RMS_ACTIVITY_FLOOR_DB);
  return clamp01(normalized);
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
 * Convert a linear RMS value into a perceptual activity level within [0, 1].
 * @param {number} [rmsValue]
 * @returns {number}
 */
export function getActivityLevel(rmsValue = frameState.rms) {
  return rmsToActivity(Number.isFinite(rmsValue) ? rmsValue : 0);
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

  const trackPosition = getTrackPositionValue();

  if (!analyserNode) {
    featureExtractor.setTrackPosition(trackPosition);
    frameState.timestamp = now;
    frameState.rms = 0;
    frameState.activity = 0;
    return frameState;
  }

  analyserNode.getFloatFrequencyData(floatFrequencyData);
  analyserNode.getByteFrequencyData(byteFrequencyData);
  analyserNode.getFloatTimeDomainData(timeDomainData);

  const deltaMs = lastFrameTimestamp > 0 ? now - lastFrameTimestamp : 0;
  const features = featureExtractor.process({
    frequencyDb: floatFrequencyData,
    waveform: timeDomainData,
    deltaMs,
    trackPosition,
    sampleRateOverride: getSampleRate(),
    fftSizeOverride: analyserNode.fftSize,
  });

  const rms = Number.isFinite(features[FEATURE_INDEX.RMS]) ? features[FEATURE_INDEX.RMS] : 0;
  frameState.rms = rms;
  frameState.activity = rmsToActivity(rms);
  frameState.timestamp = now;
  frameState.features = features;

  lastFrameTimestamp = now;
  return frameState;
}

/**
 * Expose a manual unlock helper for other modules (optional future use).
 * @returns {Promise<AudioContext>}
 */
export function unlock() {
  return ensureContext();
}
