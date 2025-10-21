import * as audio from './audio.js';
import * as nn from './nn.js';
import * as physics from './physics.js';
import * as map from './map.js';
import * as render from './render.js';
import { applyPreset as applyPresetScaling, getDefaultPreset, getPreset } from './presets.js';
import { getList, resolveUrl } from './playlist.js';
import { initDebugOverlay, runStartupDiagnostics, updateDebugOverlay } from './diagnostics.js';

const MODEL_FILES = Object.freeze([
  'models/meditation.json',
  'models/built-on-the-steppers.json',
  'models/unsound.json',
  'models/system-js.json',
  'models/binary-mirage.json',
  'models/traffic-jam.json',
  'models/backpack.json',
  'models/last-pack.json',
  'models/cloud.json',
  'models/ease-up.json',
  'models/epoch-infinity.json',
]);

const STORAGE_KEYS = Object.freeze({
  TRACK_INDEX: 'ln.lastTrack',
  SAFE_MODE: 'ln.safeMode',
  HUD_VISIBLE: 'ln.hudVisible',
  NN_BYPASS: 'ln.nnBypass',
});

const MAP_PARAM_COUNT = map.PARAM_NAMES.length;
const FALLBACK_NN_OUTPUTS = new Float32Array(MAP_PARAM_COUNT);

const RENDER_PARAMS_DEFAULT = Object.freeze({
  trailFade: 0.68,
  glow: 0.55,
  sizeJitter: 0.32,
  hueShift: 0,
  sparkleDensity: 0.14,
  zoom: 1,
});

const SIM_PARAMS_DEFAULT = Object.freeze({
  spawnRate: 0.45,
  fieldStrength: 0.62,
  cohesion: 0.54,
  repelImpulse: 0,
  vortexAmount: 0.28,
});

const PERFORMANCE_SAMPLE_WINDOW = 90;
const PERFORMANCE_DROP_FPS = 56;
const PERFORMANCE_SEVERE_FPS = 48;
const PERFORMANCE_RECOVER_FPS = 58.5;
const PERFORMANCE_DROP_WINDOW = 90;
const PERFORMANCE_SEVERE_WINDOW = 45;
const PERFORMANCE_RECOVERY_FRAMES = 120;
const PERFORMANCE_SCALE_STEPS = Object.freeze([1, 0.85, 0.7]);

const BASE_PARTICLE_CAP = 5200;
const MIN_PARTICLE_CAP = 800;
const HIDDEN_VISIBILITY_SCALE = 0.5;
const VISIBILITY_DEBOUNCE_MS = 220;

const qualityState = {
  visibilityScale: 1,
  performanceIndex: 0,
};

const performanceState = {
  dropFrames: 0,
  severeFrames: 0,
  recoveryFrames: 0,
};

let lastAppliedCap = BASE_PARTICLE_CAP;

const fpsMonitor = (() => {
  const samples = new Float32Array(PERFORMANCE_SAMPLE_WINDOW);
  let index = 0;
  let count = 0;
  let sum = 0;
  let instantaneousFps = 60;
  let averageFps = 60;
  let lastFrameTime = 1000 / 60;

  return {
    sample(frameTimeMs) {
      if (!Number.isFinite(frameTimeMs) || frameTimeMs <= 0) {
        return;
      }
      lastFrameTime = frameTimeMs;
      instantaneousFps = 1000 / frameTimeMs;
      if (count === samples.length) {
        sum -= samples[index];
      }
      samples[index] = instantaneousFps;
      sum += instantaneousFps;
      if (count < samples.length) {
        count += 1;
      }
      index = (index + 1) % samples.length;
      averageFps = count > 0 ? sum / count : instantaneousFps;
    },
    getAverageFps() {
      return averageFps;
    },
    getAverageFrameTime() {
      return averageFps > 0 ? 1000 / averageFps : lastFrameTime;
    },
    getInstantaneousFps() {
      return instantaneousFps;
    },
  };
})();

const visibilityState = {
  timer: 0,
  hidden: typeof document !== 'undefined' && document.visibilityState === 'hidden',
};

console.debug('[app] visibility state bootstrap', visibilityState.hidden);

function ensureNumberArray(source, expectedLength, label, contextLabel, options = {}) {
  if (!Array.isArray(source)) {
    throw new Error(`[${contextLabel}] ${label} must be an array of numbers.`);
  }
  if (typeof expectedLength === 'number' && source.length !== expectedLength) {
    throw new Error(
      `[${contextLabel}] ${label} expected length ${expectedLength}, received ${source.length}.`,
    );
  }
  for (let i = 0; i < source.length; i += 1) {
    const value = Number(source[i]);
    if (!Number.isFinite(value)) {
      throw new Error(`[${contextLabel}] ${label}[${i}] must be a finite number.`);
    }
    if (options.positive && !(value > 0)) {
      throw new Error(`[${contextLabel}] ${label}[${i}] must be greater than 0.`);
    }
  }
  return source;
}

function validateModelDefinition(definition, contextLabel = 'model') {
  if (!definition || typeof definition !== 'object') {
    throw new Error(`[${contextLabel}] Model definition must be an object.`);
  }

  const layers = Array.isArray(definition.layers) ? definition.layers : [];
  if (layers.length === 0) {
    throw new Error(`[${contextLabel}] Model must define at least one layer.`);
  }

  const inputSize = Number(definition.input);
  if (!Number.isFinite(inputSize) || inputSize <= 0) {
    throw new Error(`[${contextLabel}] "input" must be a positive number.`);
  }

  const norm = definition.normalization ?? {};
  ensureNumberArray(norm.mean, inputSize, 'normalization.mean', contextLabel);
  ensureNumberArray(norm.std, inputSize, 'normalization.std', contextLabel, { positive: true });

  let expectedInputs = inputSize;
  layers.forEach((rawLayer, layerIndex) => {
    if (!rawLayer || typeof rawLayer !== 'object') {
      throw new Error(`[${contextLabel}] Layer ${layerIndex} must be an object.`);
    }
    if (typeof rawLayer.activation !== 'string' || rawLayer.activation.length === 0) {
      throw new Error(`[${contextLabel}] Layer ${layerIndex} is missing an activation name.`);
    }
    const biases = ensureNumberArray(
      rawLayer.bias ?? rawLayer.biases,
      undefined,
      `layers[${layerIndex}].bias`,
      contextLabel,
    );
    if (biases.length === 0) {
      throw new Error(`[${contextLabel}] Layer ${layerIndex} must include at least one bias value.`);
    }
    const expectedWeights = expectedInputs * biases.length;
    ensureNumberArray(
      rawLayer.weights,
      expectedWeights,
      `layers[${layerIndex}].weights`,
      contextLabel,
    );
    expectedInputs = biases.length;
  });

  return true;
}

function clamp(value, min, max) {
  if (!Number.isFinite(value)) {
    return min;
  }
  if (value < min) {
    return min;
  }
  if (value > max) {
    return max;
  }
  return value;
}

function readStorage(key) {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStorage(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Ignore storage write failures (private mode, quota exceeded, etc.).
  }
}

function readStoredBoolean(key, defaultValue) {
  const stored = readStorage(key);
  if (stored === null) {
    return defaultValue;
  }
  return stored === '1' || stored.toLowerCase() === 'true';
}

function writeStoredBoolean(key, value) {
  writeStorage(key, value ? '1' : '0');
}

function readStoredInt(key, defaultValue, min = Number.MIN_SAFE_INTEGER, max = Number.MAX_SAFE_INTEGER) {
  const stored = readStorage(key);
  if (stored === null) {
    return defaultValue;
  }
  const parsed = parseInt(stored, 10);
  if (!Number.isFinite(parsed)) {
    return defaultValue;
  }
  return clamp(parsed, min, max);
}

function wrapHue(value) {
  if (!Number.isFinite(value)) {
    return 0;
  }
  let result = value;
  while (result > 180) {
    result -= 360;
  }
  while (result < -180) {
    result += 360;
  }
  return result;
}

const playlistSelect = document.getElementById('playlist');
const audioElement = document.getElementById('player');
const volumeSlider = document.getElementById('volume');
const playButton = document.getElementById('play');
const prevButton = document.getElementById('prev');
const nextButton = document.getElementById('next');
const seekSlider = document.getElementById('seek');

if (!playlistSelect || !audioElement || !volumeSlider || !playButton || !prevButton || !nextButton || !seekSlider) {
  throw new Error('Required controls missing from DOM (playlist, audio, volume, play, prev, next, or seek).');
}

render.init();
render.setWorldSize(2, 2);
render.setStatus('Idle · Particles 0');

physics.configure({
  bounds: { width: 2, height: 2, mode: 'wrap' },
  baseCap: BASE_PARTICLE_CAP,
  minCap: MIN_PARTICLE_CAP,
  defaults: {
    spawnRate: SIM_PARAMS_DEFAULT.spawnRate,
    fieldStrength: SIM_PARAMS_DEFAULT.fieldStrength,
    cohesion: SIM_PARAMS_DEFAULT.cohesion,
    repelImpulse: SIM_PARAMS_DEFAULT.repelImpulse,
    vortexAmount: SIM_PARAMS_DEFAULT.vortexAmount,
  },
});

console.info('[app] physics dynamicCap init', physics.getMetrics().dynamicCap);

if (visibilityState.hidden) {
  qualityState.visibilityScale = Math.max(MIN_PARTICLE_CAP / BASE_PARTICLE_CAP, HIDDEN_VISIBILITY_SCALE);
  applyQualityCap();
}

const renderParams = { ...RENDER_PARAMS_DEFAULT };
const simParams = { ...SIM_PARAMS_DEFAULT };
let activePreset = getDefaultPreset();
const manualAdjustments = {
  spawnOffset: 0,
  glowOffset: 0,
  sparkleOffset: 0,
  hueOffset: 0,
};


function resetManualAdjustments() {
  manualAdjustments.spawnOffset = 0;
  manualAdjustments.glowOffset = 0;
  manualAdjustments.sparkleOffset = 0;
  manualAdjustments.hueOffset = 0;
}

function constrainManualAdjustmentsForSafeMode(enabled) {
  if (!enabled) {
    return;
  }
  manualAdjustments.spawnOffset = Math.min(manualAdjustments.spawnOffset, 0);
  manualAdjustments.glowOffset = Math.min(manualAdjustments.glowOffset, 0);
  manualAdjustments.sparkleOffset = Math.min(manualAdjustments.sparkleOffset, 0);
}

const tracks = getList();
if (tracks.length === 0) {
  throw new Error('Playlist is empty; Phase 2 requires 11 static tracks.');
}
if (MODEL_FILES.length !== tracks.length) {
  throw new Error('Model placeholder count mismatch with playlist length.');
}

const initialTrackIndex = readStoredInt(STORAGE_KEYS.TRACK_INDEX, 0, 0, tracks.length - 1);
const storedSafeMode = readStoredBoolean(STORAGE_KEYS.SAFE_MODE, false);
const storedHudVisible = readStoredBoolean(STORAGE_KEYS.HUD_VISIBLE, true);
const storedBypass = readStoredBoolean(STORAGE_KEYS.NN_BYPASS, false);

let safeModeEnabled = storedSafeMode;
let nnBypass = storedBypass;
let lastModelOutputs = FALLBACK_NN_OUTPUTS;

map.configure({ safeMode: safeModeEnabled });
if (safeModeEnabled) {
  constrainManualAdjustmentsForSafeMode(true);
}

initDebugOverlay({ search: typeof window !== 'undefined' ? window.location.search : '' });
await runStartupDiagnostics({ safeMode: safeModeEnabled });

// Remove any stray options before populating the locked playlist.
playlistSelect.innerHTML = '';
tracks.forEach((track, index) => {
  const option = document.createElement('option');
  option.value = String(index);
  option.textContent = track.title;
  option.dataset.src = resolveUrl(index);
  playlistSelect.append(option);
});

render.setTrackTitle(tracks[initialTrackIndex]?.title ?? 'Latent Noise');

const modelCache = new Map();
let currentTrackIndex = -1;
let activeModelIndex = -1;
let modelLoadToken = 0;

const playback = {
  status: 'Idle',
  lastStatusText: '',
};

let autoAdvanceTimer = 0;

function clearAutoAdvanceTimer() {
  if (autoAdvanceTimer) {
    window.clearTimeout(autoAdvanceTimer);
    autoAdvanceTimer = 0;
  }
}

function copyParams(target, source) {
  if (!target || !source) {
    return target;
  }
  for (const key of Object.keys(source)) {
    target[key] = source[key];
  }
  return target;
}

function getPerformanceScale() {
  const scale = PERFORMANCE_SCALE_STEPS[qualityState.performanceIndex];
  return Number.isFinite(scale) ? scale : 1;
}

function applyQualityCap() {
  const minScale = MIN_PARTICLE_CAP / BASE_PARTICLE_CAP;
  const visibilityScale = clamp(qualityState.visibilityScale, minScale, 1);
  const performanceScale = clamp(getPerformanceScale(), minScale, 1);
  const combinedScale = clamp(visibilityScale * performanceScale, minScale, 1);
  const targetCap = clamp(Math.round(BASE_PARTICLE_CAP * combinedScale), MIN_PARTICLE_CAP, BASE_PARTICLE_CAP);
  if (targetCap === lastAppliedCap) {
    return;
  }
  lastAppliedCap = targetCap;
  physics.configure({ baseCap: targetCap });
  console.info('[app] quality cap update', targetCap, `(scale ${combinedScale.toFixed(2)})`);
}

function queueVisibilityUpdate(hidden) {
  if (!Number.isFinite(VISIBILITY_DEBOUNCE_MS) || VISIBILITY_DEBOUNCE_MS <= 0) {
    const nextScale = hidden ? Math.max(MIN_PARTICLE_CAP / BASE_PARTICLE_CAP, HIDDEN_VISIBILITY_SCALE) : 1;
    if (qualityState.visibilityScale !== nextScale) {
      qualityState.visibilityScale = nextScale;
      applyQualityCap();
    }
    visibilityState.hidden = hidden;
    return;
  }

  if (visibilityState.timer) {
    window.clearTimeout(visibilityState.timer);
  }
  visibilityState.timer = window.setTimeout(() => {
    visibilityState.timer = 0;
    const nextScale = hidden ? Math.max(MIN_PARTICLE_CAP / BASE_PARTICLE_CAP, HIDDEN_VISIBILITY_SCALE) : 1;
    if (qualityState.visibilityScale !== nextScale) {
      qualityState.visibilityScale = nextScale;
      applyQualityCap();
    }
    visibilityState.hidden = hidden;
  }, VISIBILITY_DEBOUNCE_MS);
}

function updatePerformanceScaling(averageFps) {
  if (!Number.isFinite(averageFps) || averageFps <= 0) {
    return;
  }

  if (averageFps < PERFORMANCE_SEVERE_FPS) {
    performanceState.severeFrames += 1;
  } else {
    performanceState.severeFrames = Math.max(0, performanceState.severeFrames - 2);
  }

  if (averageFps < PERFORMANCE_DROP_FPS) {
    performanceState.dropFrames += 1;
  } else {
    performanceState.dropFrames = Math.max(0, performanceState.dropFrames - 1);
  }

  if (averageFps > PERFORMANCE_RECOVER_FPS) {
    performanceState.recoveryFrames += 1;
  } else {
    performanceState.recoveryFrames = 0;
  }

  let changed = false;

  if (
    performanceState.severeFrames > PERFORMANCE_SEVERE_WINDOW
    && qualityState.performanceIndex < PERFORMANCE_SCALE_STEPS.length - 1
  ) {
    qualityState.performanceIndex = PERFORMANCE_SCALE_STEPS.length - 1;
    changed = true;
    performanceState.dropFrames = 0;
    performanceState.severeFrames = 0;
    performanceState.recoveryFrames = 0;
  } else if (
    performanceState.dropFrames > PERFORMANCE_DROP_WINDOW
    && qualityState.performanceIndex < PERFORMANCE_SCALE_STEPS.length - 1
  ) {
    qualityState.performanceIndex = Math.min(
      qualityState.performanceIndex + 1,
      PERFORMANCE_SCALE_STEPS.length - 1,
    );
    changed = true;
    performanceState.dropFrames = 0;
    performanceState.severeFrames = 0;
    performanceState.recoveryFrames = 0;
  } else if (
    qualityState.performanceIndex > 0
    && performanceState.recoveryFrames > PERFORMANCE_RECOVERY_FRAMES
    && averageFps > PERFORMANCE_RECOVER_FPS
  ) {
    qualityState.performanceIndex = Math.max(0, qualityState.performanceIndex - 1);
    changed = true;
    performanceState.recoveryFrames = 0;
    performanceState.dropFrames = 0;
    performanceState.severeFrames = 0;
  }

  if (changed) {
    applyQualityCap();
  }
}

function applyPresetForTrack(index) {
  const preset = getPreset(index) ?? activePreset ?? getDefaultPreset();
  activePreset = preset;

  resetManualAdjustments();
  copyParams(simParams, SIM_PARAMS_DEFAULT);
  copyParams(renderParams, RENDER_PARAMS_DEFAULT);

  const adjusted = applyPresetScaling(preset, { sim: simParams, render: renderParams });
  if (adjusted && typeof adjusted === 'object') {
    if (adjusted.sim) {
      copyParams(simParams, adjusted.sim);
    }
    if (adjusted.render) {
      copyParams(renderParams, adjusted.render);
    }
  }

  physics.configure({
    defaults: {
      spawnRate: simParams.spawnRate,
      fieldStrength: simParams.fieldStrength,
      cohesion: simParams.cohesion,
      repelImpulse: simParams.repelImpulse,
      vortexAmount: simParams.vortexAmount,
    },
  });

  map.reset({
    spawnRate: simParams.spawnRate,
    fieldStrength: simParams.fieldStrength,
    cohesion: simParams.cohesion,
    repelImpulse: simParams.repelImpulse,
    vortexAmount: simParams.vortexAmount,
    trailFade: renderParams.trailFade,
    glow: renderParams.glow,
    sizeJitter: renderParams.sizeJitter,
    hueShift: renderParams.hueShift,
    sparkleDensity: renderParams.sparkleDensity,
  });

  return preset;
}

function updateStatus(metrics) {
  const count = metrics?.count ?? 0;
  const cap = metrics?.dynamicCap ?? 0;
  const statusText = `${playback.status} · Particles ${count}/${cap}`;
  if (statusText !== playback.lastStatusText) {
    render.setStatus(statusText);
    playback.lastStatusText = statusText;
  }
}

function updatePlayButtonUi() {
  if (!playButton) {
    return;
  }
  playButton.textContent = audioElement.paused ? 'Play' : 'Pause';
}

function updateSeekUi(currentSeconds, durationSeconds) {
  if (!seekSlider) {
    return;
  }
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
    seekSlider.value = '0';
    seekSlider.disabled = true;
    return;
  }
  const percent = clamp((currentSeconds / durationSeconds) * 100, 0, 100);
  seekSlider.value = percent.toFixed(2);
  seekSlider.disabled = false;
}

function handleSeekInput() {
  if (!seekSlider) {
    return;
  }
  if (!Number.isFinite(audioElement.duration) || audioElement.duration <= 0) {
    return;
  }
  const percent = clamp(Number(seekSlider.value), 0, 100) / 100;
  audioElement.currentTime = percent * audioElement.duration;
  render.updateTrackTime(audioElement.currentTime, audioElement.duration);
  updateSeekUi(audioElement.currentTime, audioElement.duration);
}

function applyMappedParams(mapped) {
  if (!mapped) {
    return;
  }
  const safe = Boolean(safeModeEnabled);
  const spawnBase = Number.isFinite(mapped.spawnRate) ? mapped.spawnRate : SIM_PARAMS_DEFAULT.spawnRate;
  const fieldBase = Number.isFinite(mapped.fieldStrength) ? mapped.fieldStrength : SIM_PARAMS_DEFAULT.fieldStrength;
  const cohesionBase = Number.isFinite(mapped.cohesion) ? mapped.cohesion : SIM_PARAMS_DEFAULT.cohesion;
  const repelBase = Number.isFinite(mapped.repelImpulse) ? mapped.repelImpulse : SIM_PARAMS_DEFAULT.repelImpulse;
  const vortexBase = Number.isFinite(mapped.vortexAmount) ? mapped.vortexAmount : SIM_PARAMS_DEFAULT.vortexAmount;

  const trailBase = Number.isFinite(mapped.trailFade) ? mapped.trailFade : RENDER_PARAMS_DEFAULT.trailFade;
  const glowBase = Number.isFinite(mapped.glow) ? mapped.glow : RENDER_PARAMS_DEFAULT.glow;
  const jitterBase = Number.isFinite(mapped.sizeJitter) ? mapped.sizeJitter : RENDER_PARAMS_DEFAULT.sizeJitter;
  const hueBase = Number.isFinite(mapped.hueShift) ? mapped.hueShift : RENDER_PARAMS_DEFAULT.hueShift;
  const sparkleBase = Number.isFinite(mapped.sparkleDensity) ? mapped.sparkleDensity : RENDER_PARAMS_DEFAULT.sparkleDensity;
  const zoomBase = Number.isFinite(mapped.zoom) ? mapped.zoom : RENDER_PARAMS_DEFAULT.zoom;

  const spawnMin = 0.05;
  const spawnMax = safe ? 0.8 : 1.2;
  const glowMax = safe ? 0.6 : 1;
  const sparkleMax = safe ? 0.65 : 1;
  const zoomMin = 0.5;
  const zoomMax = safe ? 1.5 : 2;

  const spawnAdjusted = spawnBase + manualAdjustments.spawnOffset;
  const glowAdjusted = glowBase + manualAdjustments.glowOffset;
  const sparkleAdjusted = sparkleBase + manualAdjustments.sparkleOffset;
  const hueAdjusted = hueBase + manualAdjustments.hueOffset;

  simParams.spawnRate = clamp(spawnAdjusted, spawnMin, spawnMax);
  simParams.fieldStrength = clamp(fieldBase, 0, 1.5);
  simParams.cohesion = clamp(cohesionBase, 0.1, 1.2);
  simParams.repelImpulse = clamp(repelBase, 0, 1);
  simParams.vortexAmount = clamp(vortexBase, 0, 1.2);

  renderParams.trailFade = clamp(trailBase, 0.2, 0.98);
  renderParams.glow = clamp(glowAdjusted, 0, glowMax);
  renderParams.sizeJitter = clamp(jitterBase, 0, 0.8);
  renderParams.hueShift = wrapHue(hueAdjusted);
  renderParams.sparkleDensity = clamp(sparkleAdjusted, 0, sparkleMax);
  renderParams.zoom = clamp(zoomBase, zoomMin, zoomMax);
}

function cacheEntryIsPromise(entry) {
  return entry && typeof entry === 'object' && typeof entry.then === 'function';
}

async function fetchModelDefinition(index) {
  const existing = modelCache.get(index);
  if (existing) {
    if (cacheEntryIsPromise(existing)) {
      return existing;
    }
    return existing;
  }

  const url = MODEL_FILES[index];
  if (!url) {
    throw new RangeError(`Model path missing for playlist index ${index}`);
  }

  const fetchPromise = fetch(url)
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Failed to fetch model "${url}" (${response.status} ${response.statusText}).`);
      }
      return response.json();
    })
    .then((json) => {
      validateModelDefinition(json, url);
      modelCache.set(index, json);
      return json;
    })
    .catch((error) => {
      modelCache.delete(index);
      throw error;
    });

  modelCache.set(index, fetchPromise);
  return fetchPromise;
}

async function prepareModel(index) {
  const token = ++modelLoadToken;
  try {
    const definition = await fetchModelDefinition(index);
    if (token !== modelLoadToken) {
      return null;
    }

    const info = await nn.loadModel(definition);
    if (token !== modelLoadToken) {
      return info;
    }

    audio.frame();
    const features = audio.getFeatureVector();
    const normalized = nn.normalize(features);
    const warmupOutputs = nn.forward(normalized);
    lastModelOutputs = warmupOutputs || FALLBACK_NN_OUTPUTS;
    activeModelIndex = index;
    if (info) {
      console.info(`[app] Model ready for "${tracks[index]?.title ?? index}" (${info.layers} layers)`);
    }
    return info;
  } catch (error) {
    console.error(`[app] Failed to load model for "${tracks[index]?.title ?? index}"`, error);
    return null;
  }
}

function setTrack(index, options = {}) {
  clearAutoAdvanceTimer();
  if (!Number.isInteger(index) || index < 0 || index >= tracks.length) {
    console.warn('[app] Ignoring out-of-range track index', index);
    return;
  }
  const target = tracks[index];
  const autoplay = options.autoplay ?? !audioElement.paused;

  if (index === currentTrackIndex && activeModelIndex === index) {
    return;
  }

  currentTrackIndex = index;
  playlistSelect.selectedIndex = index;
  playlistSelect.value = String(index);
  audioElement.src = resolveUrl(index);
  writeStorage(STORAGE_KEYS.TRACK_INDEX, String(index));
  const preset = applyPresetForTrack(index);
  render.setTrackTitle(target?.title ?? `Track ${index + 1}`);
  render.updateTrackTime(0, Number.isFinite(audioElement.duration) ? audioElement.duration : NaN);
  updateSeekUi(0, NaN);
  playback.status = autoplay ? 'Buffering' : 'Idle';
  updateStatus(physics.getMetrics());
  lastModelOutputs = FALLBACK_NN_OUTPUTS;

  if (preset) {
    console.info('[app] Applied preset:', preset.title);
  }

  void prepareModel(index);

  if (!autoplay) {
    audioElement.pause();
  }

  if (autoplay) {
    audioElement.play().catch((error) => {
      playback.status = 'Idle';
      updateStatus(physics.getMetrics());
      console.warn('[app] Autoplay blocked', error);
    });
  }

  updatePlayButtonUi();
}

function nextTrack(step = 1, options = {}) {
  if (tracks.length === 0) {
    return;
  }
  const nextIndex = (currentTrackIndex + step + tracks.length) % tracks.length;
  const autoplay = options.autoplay ?? !audioElement.paused;
  setTrack(nextIndex, { autoplay });
}

function prevTrack() {
  nextTrack(-1);
}

function togglePlayback() {
  if (audioElement.paused) {
    audioElement.play().catch((error) => {
      console.warn('[app] Playback start blocked', error);
    });
  } else {
    audioElement.pause();
  }
}

function seekBy(seconds) {
  if (!Number.isFinite(seconds)) {
    return;
  }
  if (!Number.isFinite(audioElement.duration) || audioElement.duration <= 0) {
    audioElement.currentTime = Math.max(0, audioElement.currentTime + seconds);
  } else {
    const next = clamp(audioElement.currentTime + seconds, 0, audioElement.duration);
    audioElement.currentTime = next;
  }
  render.updateTrackTime(audioElement.currentTime, audioElement.duration);
  updateSeekUi(audioElement.currentTime, audioElement.duration);
}

// Default to the stored track (or first) and ensure the audio element points to bundled media only.
setTrack(initialTrackIndex, { autoplay: false });

const restoredVolume = audio.init(audioElement);
const initialVolume = Number.isFinite(restoredVolume) ? restoredVolume : Number(volumeSlider.value);
volumeSlider.value = initialVolume.toFixed(2);
audio.setVolume(initialVolume);
render.updateVolume(initialVolume);
updatePlayButtonUi();

volumeSlider.addEventListener('input', () => {
  const nextVolume = Number(volumeSlider.value);
  if (Number.isNaN(nextVolume)) {
    return;
  }
  audio.setVolume(nextVolume);
  render.updateVolume(nextVolume);
});

playButton.addEventListener('click', () => {
  togglePlayback();
});

prevButton.addEventListener('click', () => {
  prevTrack();
});

nextButton.addEventListener('click', () => {
  nextTrack(1);
});

seekSlider.addEventListener('input', handleSeekInput);
seekSlider.addEventListener('change', handleSeekInput);

playlistSelect.addEventListener('change', (event) => {
  const target = event.target;
  if (!(target instanceof HTMLSelectElement)) {
    return;
  }
  const selected = Number(target.value);
  if (Number.isNaN(selected)) {
    return;
  }
  setTrack(selected, { autoplay: !audioElement.paused });
});

render.on('playToggle', togglePlayback);
render.on('nextTrack', () => nextTrack(1));
render.on('prevTrack', prevTrack);
render.on('seekForward', ({ seconds }) => {
  seekBy(Math.abs(Number.isFinite(seconds) ? seconds : 5));
});
render.on('seekBackward', ({ seconds }) => {
  seekBy(-Math.abs(Number.isFinite(seconds) ? seconds : 5));
});
render.on('selectTrack', ({ index }) => {
  if (!Number.isInteger(index)) {
    return;
  }
  setTrack((index + tracks.length) % tracks.length, { autoplay: !audioElement.paused });
});
render.on('adjustParticles', ({ delta }) => {
  if (!Number.isFinite(delta)) {
    return;
  }
  manualAdjustments.spawnOffset = clamp(manualAdjustments.spawnOffset + delta, -0.4, 0.6);
  constrainManualAdjustmentsForSafeMode(safeModeEnabled);
});
render.on('adjustIntensity', ({ delta }) => {
  if (!Number.isFinite(delta)) {
    return;
  }
  manualAdjustments.glowOffset = clamp(manualAdjustments.glowOffset + delta, -0.5, 0.5);
  manualAdjustments.sparkleOffset = clamp(manualAdjustments.sparkleOffset + delta * 0.6, -0.6, 0.6);
  constrainManualAdjustmentsForSafeMode(safeModeEnabled);
});
render.on('cyclePalette', ({ direction }) => {
  const dir = direction >= 0 ? 1 : -1;
  manualAdjustments.hueOffset = wrapHue(manualAdjustments.hueOffset + dir * 20);
});
render.on('safeModeChange', (enabled) => {
  safeModeEnabled = Boolean(enabled);
  map.setSafeMode(safeModeEnabled);
  writeStoredBoolean(STORAGE_KEYS.SAFE_MODE, safeModeEnabled);
  constrainManualAdjustmentsForSafeMode(safeModeEnabled);
});
render.on('nnBypassChange', (enabled) => {
  nnBypass = Boolean(enabled);
  writeStoredBoolean(STORAGE_KEYS.NN_BYPASS, nnBypass);
  if (nnBypass) {
    lastModelOutputs = FALLBACK_NN_OUTPUTS;
  }
});
render.on('toggle', ({ name, value }) => {
  if (name === 'hud') {
    writeStoredBoolean(STORAGE_KEYS.HUD_VISIBLE, Boolean(value));
  }
});

render.setToggle('hud', storedHudVisible);
render.setToggle('safe', storedSafeMode);
render.setToggle('bypass', storedBypass);

audioElement.addEventListener('play', () => {
  clearAutoAdvanceTimer();
  playback.status = 'Playing';
  updateStatus(physics.getMetrics());
  updatePlayButtonUi();
});

audioElement.addEventListener('pause', () => {
  if (audioElement.ended) {
    playback.status = 'Ended';
  } else if (audioElement.currentTime > 0) {
    playback.status = 'Paused';
  } else {
    playback.status = 'Idle';
  }
  updateStatus(physics.getMetrics());
  updatePlayButtonUi();
});

audioElement.addEventListener('ended', () => {
  playback.status = 'Ended';
  updateStatus(physics.getMetrics());
  updatePlayButtonUi();
  clearAutoAdvanceTimer();
  autoAdvanceTimer = window.setTimeout(() => {
    autoAdvanceTimer = 0;
    nextTrack(1, { autoplay: true });
  }, 1000);
});

const updateTrackTime = () => {
  const { currentTime, duration } = audioElement;
  render.updateTrackTime(currentTime, duration);
  updateSeekUi(currentTime, duration);
};

audioElement.addEventListener('timeupdate', updateTrackTime);
audioElement.addEventListener('loadedmetadata', updateTrackTime);

const blockFileInput = (event) => {
  event.preventDefault();
  if (event.dataTransfer) {
    event.dataTransfer.dropEffect = 'none';
    event.dataTransfer.effectAllowed = 'none';
  }
};

window.addEventListener('dragenter', blockFileInput);
window.addEventListener('dragover', blockFileInput);
window.addEventListener('drop', blockFileInput);

document.addEventListener('paste', (event) => {
  if (event.clipboardData && event.clipboardData.files && event.clipboardData.files.length > 0) {
    event.preventDefault();
  }
});

document.addEventListener('visibilitychange', () => {
  queueVisibilityUpdate(document.visibilityState === 'hidden');
});

let lastFrameTime = performance.now();

function frame(now) {
  const dtMsRaw = now - lastFrameTime;
  lastFrameTime = now;
  const dtSeconds = clamp(dtMsRaw / 1000, 1 / 240, 1 / 20);
  const frameTimeMs = dtSeconds * 1000;

  fpsMonitor.sample(frameTimeMs);
  const averageFps = fpsMonitor.getAverageFps();
  const averageFrameTime = fpsMonitor.getAverageFrameTime();
  const instantaneousFps = fpsMonitor.getInstantaneousFps();

  updatePerformanceScaling(averageFps);

  const audioState = audio.frame();
  const features = audioState?.features ?? audio.getFeatureVector();
  const activity = Number.isFinite(audioState?.rms) ? Math.max(0, audioState.rms) : 0;

  let nnOutputs = lastModelOutputs;
  if (!nnBypass && activeModelIndex === currentTrackIndex) {
    try {
      const normalized = nn.normalize(features);
      nnOutputs = nn.forward(normalized);
      lastModelOutputs = nnOutputs;
    } catch (error) {
      console.warn('[app] NN inference failed; using fallback outputs.', error);
      nnOutputs = FALLBACK_NN_OUTPUTS;
      lastModelOutputs = FALLBACK_NN_OUTPUTS;
    }
  } else if (nnBypass) {
    nnOutputs = FALLBACK_NN_OUTPUTS;
    lastModelOutputs = FALLBACK_NN_OUTPUTS;
  }

  const mappedParams = map.update(nnOutputs, {
    dt: dtSeconds,
    timestamp: now,
    activity,
    features,
  });
  applyMappedParams(mappedParams);

  physics.step(simParams, { dt: dtSeconds, frameTime: frameTimeMs, frameTimeAvg: averageFrameTime });
  const particles = physics.getParticles();
  const metrics = physics.getMetrics();

  render.renderFrame(particles, renderParams, {
    dt: dtSeconds,
    frameTime: frameTimeMs,
    frameTimeAvg: averageFrameTime,
    fps: instantaneousFps,
    fpsAvg: averageFps,
  });
  updateStatus(metrics);

  updateDebugOverlay({
    fps: instantaneousFps,
    fpsAvg: averageFps,
    activity,
    features,
    outputs: nnOutputs,
    modelInfo: nn.getCurrentModelInfo(),
    params: {
      spawnRate: simParams.spawnRate,
      fieldStrength: simParams.fieldStrength,
      cohesion: simParams.cohesion,
      repelImpulse: simParams.repelImpulse,
      vortexAmount: simParams.vortexAmount,
      trailFade: renderParams.trailFade,
      glow: renderParams.glow,
      sizeJitter: renderParams.sizeJitter,
      hueShift: renderParams.hueShift,
      sparkleDensity: renderParams.sparkleDensity,
      zoom: renderParams.zoom,
      spawnOffset: manualAdjustments.spawnOffset,
      glowOffset: manualAdjustments.glowOffset,
      sparkleOffset: manualAdjustments.sparkleOffset,
      hueOffset: manualAdjustments.hueOffset,
      safeMode: safeModeEnabled ? 1 : 0,
      nnBypass: nnBypass ? 1 : 0,
    },
  });

  if (!audioElement.paused && audioElement.readyState >= 1) {
    const { currentTime, duration } = audioElement;
    render.updateTrackTime(currentTime, duration);
    updateSeekUi(currentTime, duration);
  }

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
