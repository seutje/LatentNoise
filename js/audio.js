const STORAGE_KEY = 'ln.volume';
const DEFAULT_VOLUME = 0.7;
const RAMP_TIME = 0.05; // seconds for smooth gain transitions.

const EMPTY_FLOAT = new Float32Array(0);
const EMPTY_BYTE = new Uint8Array(0);

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

const frameState = {
  frequency: EMPTY_FLOAT,
  frequencyByte: EMPTY_BYTE,
  waveform: EMPTY_FLOAT,
  rms: 0,
  timestamp: 0,
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
  } catch (error) {
    // Ignore storage access issues (private mode, disabled storage, etc.).
  }
  return DEFAULT_VOLUME;
}

function persistVolume(value) {
  try {
    window.localStorage.setItem(STORAGE_KEY, value.toString());
  } catch (error) {
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

  sourceNode.connect(gainNode);
  gainNode.connect(analyserNode);
  analyserNode.connect(audioContext.destination);

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
 * Update cached analyser data. Safe to call before initialization.
 * @returns {{frequency: Float32Array, frequencyByte: Uint8Array, waveform: Float32Array, rms: number, timestamp: number}}
 */
export function frame() {
  if (!analyserNode) {
    return frameState;
  }

  analyserNode.getFloatFrequencyData(floatFrequencyData);
  analyserNode.getByteFrequencyData(byteFrequencyData);
  analyserNode.getFloatTimeDomainData(timeDomainData);

  frameState.rms = computeRms(timeDomainData);
  frameState.timestamp = performance.now();

  return frameState;
}

/**
 * Expose a manual unlock helper for other modules (optional future use).
 * @returns {Promise<AudioContext>}
 */
export function unlock() {
  return ensureContext();
}
