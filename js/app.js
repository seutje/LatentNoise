import * as audio from './audio.js';
import * as nn from './nn.js';
import * as physics from './physics.js';
import * as map from './map.js';
import * as render from './render.js';
import { applyPreset as applyPresetScaling, getDefaultPreset, getPreset } from './presets.js';
import { getList, resolveUrl } from './playlist.js';
import { initDebugOverlay, runStartupDiagnostics, updateDebugOverlay } from './diagnostics.js';
import * as byom from './byom.js';
import { FRESH_MODEL_ID, FRESH_MODEL_LABEL } from './byom-constants.js';
import { createController as createTrainingController } from './training.js';
import * as byomStorage from './byom-storage.js';
import { init as initNotifications, notify } from './notifications.js';
import { formatCorrelation } from './correlation-math.js';
import { createRepeatController } from './repeat-controller.js';
import * as videoExport from './video-export.js';

const MODEL_FILES = Object.freeze([
  'models/meditation.json',
  'models/built-on-the-steppers.json',
  'models/unsound.json',
  'models/system-js.json',
  'models/binary-mirage.json',
  'models/traffic-jam.json',
  'models/backpack.json',
  'models/last-pack.json',
  'models/clouds.json',
  'models/ease-up.json',
  'models/epoch-infinity.json',
]);

const STORAGE_KEYS = Object.freeze({
  TRACK_INDEX: 'ln.lastTrack',
  SAFE_MODE: 'ln.safeMode',
  NN_BYPASS: 'ln.nnBypass',
  REPEAT: 'ln.repeat',
});

const EXPORT_STORAGE_KEY = 'ln.exportOptions';
const EXPORT_MIN_DIMENSION = 320;
const EXPORT_MAX_DIMENSION = 4320;
const EXPORT_MIN_FRAME_RATE = 12;
const EXPORT_MAX_FRAME_RATE = 120;
const EXPORT_DEFAULT_FRAME_RATE = 60;
const EXPORT_MIN_BITRATE = 6_000_000;
const EXPORT_MAX_BITRATE = 30_000_000;
const MEGABIT = 1_000_000;

const MAP_PARAM_COUNT = map.PARAM_NAMES.length;
const FALLBACK_NN_OUTPUTS = new Float32Array(MAP_PARAM_COUNT);
const FEATURE_LABELS = audio.getFeatureLabels();
const OUTPUT_LABELS = map.PARAM_NAMES.slice();

function resolveCorrelationOrientation(metrics) {
  if (!metrics) {
    return 'Direct';
  }
  if (metrics.orientation === 'inverse' || metrics.inverse === true || metrics.orientationSign === -1) {
    return 'Inverse';
  }
  return 'Direct';
}

function buildCorrelationNotification(stats) {
  const perCorrelation = Array.isArray(stats?.correlations)
    ? stats.correlations
    : Array.isArray(stats?.correlationMetrics?.perCorrelation)
      ? stats.correlationMetrics.perCorrelation
      : [];
  if (perCorrelation.length === 0) {
    return 'Training complete! BYOM entry saved.';
  }
  const lines = [];
  perCorrelation.forEach((metrics, index) => {
    if (!metrics) {
      return;
    }
    const featureFallback = Number.isFinite(metrics.featureIndex)
      ? `Feature ${metrics.featureIndex}`
      : `Feature ${index}`;
    const outputFallback = Number.isFinite(metrics.outputIndex)
      ? `Output ${metrics.outputIndex}`
      : 'Output';
    const featureName = metrics.featureName ?? featureFallback;
    const outputName = metrics.outputName ?? outputFallback;
    const label = index === 0 ? 'Primary' : `Secondary #${index}`;
    const orientation = resolveCorrelationOrientation(metrics);
    const value = formatCorrelation(
      typeof metrics.correlation === 'number' ? metrics.correlation : Number(metrics.correlation),
    );
    lines.push(`${label} (${featureName} → ${outputName}, ${orientation}): ${value}`);
  });
  if (lines.length === 0) {
    return 'Training complete! BYOM entry saved.';
  }
  const combinedFitness = Number(stats?.correlationMetrics?.combinedFitness);
  const messageLines = ['Training complete!', ...lines];
  if (Number.isFinite(combinedFitness)) {
    messageLines.push(`Combined fitness: ${combinedFitness.toFixed(4)}`);
  }
  return messageLines.join('\n');
}

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

const ANIMATION_LOOKAHEAD_MS = 0;
const TRACK_INTERMISSION_MS = 1000;

const BASE_PARTICLE_CAP = 5200;
const MIN_PARTICLE_CAP = 800;
const HIDDEN_VISIBILITY_SCALE = 0.5;
const VISIBILITY_DEBOUNCE_MS = 220;

const FEEDBACK_DEBOUNCE_MS = 450;
const FEEDBACK_STATUS_RESET_MS = 3200;
const FEEDBACK_STATUS_MESSAGES = Object.freeze({
  offline: 'Feedback offline',
  ready: 'Ready for feedback',
  pending: 'Sending feedback…',
  success: 'Feedback received',
  error: 'Feedback failed',
});

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
let particleIntermissionUntil = 0;

const adaptiveFeedbackState = {
  controller: null,
  available: false,
  lastSentAt: 0,
  resetTimer: 0,
};

const ZOOM_SPEC = map.getParamSpec('zoom') ?? {};
const DEFAULT_ZOOM_SOURCE_MIN = 0.05;
const DEFAULT_ZOOM_SOURCE_MAX = 20;
const ZOOM_SOURCE_MIN = Number.isFinite(ZOOM_SPEC.min) ? ZOOM_SPEC.min : DEFAULT_ZOOM_SOURCE_MIN;
const ZOOM_SOURCE_MAX = Number.isFinite(ZOOM_SPEC.max) ? ZOOM_SPEC.max : DEFAULT_ZOOM_SOURCE_MAX;
const ZOOM_SOURCE_RANGE = Math.max(ZOOM_SOURCE_MAX - ZOOM_SOURCE_MIN, 1e-6);
const ZOOM_OUTPUT_MIN = 0.1;
const ZOOM_OUTPUT_MAX = 20;

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

function nowMs() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function startParticleIntermission(durationMs = TRACK_INTERMISSION_MS) {
  const clampedDuration = Number.isFinite(durationMs) && durationMs > 0 ? durationMs : 0;
  if (clampedDuration <= 0) {
    particleIntermissionUntil = 0;
    physics.reset();
    return;
  }
  particleIntermissionUntil = nowMs() + clampedDuration;
  physics.reset();
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

function resolveSourceDimensions(canvas) {
  const width = Math.round(Number(canvas?.width));
  const height = Math.round(Number(canvas?.height));
  if (Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0) {
    return { width, height };
  }
  return { width: 1920, height: 1080 };
}

function resolvePresetDimensions(preset, canvas, customDimensions) {
  switch (preset) {
    case '720p':
      return { width: 1280, height: 720 };
    case '1080p':
      return { width: 1920, height: 1080 };
    case '1440p':
      return { width: 2560, height: 1440 };
    case '2160p':
      return { width: 3840, height: 2160 };
    case 'custom': {
      if (customDimensions && Number.isFinite(customDimensions.width) && Number.isFinite(customDimensions.height)) {
        return { width: customDimensions.width, height: customDimensions.height };
      }
      break;
    }
    default:
      break;
  }
  return resolveSourceDimensions(canvas);
}

function clampExportDimension(value, fallback) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return clamp(fallback, EXPORT_MIN_DIMENSION, EXPORT_MAX_DIMENSION);
  }
  return clamp(Math.round(numeric), EXPORT_MIN_DIMENSION, EXPORT_MAX_DIMENSION);
}

function clampExportFrameRate(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return EXPORT_DEFAULT_FRAME_RATE;
  }
  return clamp(Math.round(numeric), EXPORT_MIN_FRAME_RATE, EXPORT_MAX_FRAME_RATE);
}

function clampExportBitrate(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return null;
  }
  return clamp(Math.round(numeric), EXPORT_MIN_BITRATE, EXPORT_MAX_BITRATE);
}

function normalizeStoredExportSettings(raw = {}, canvas) {
  const sourceDimensions = resolveSourceDimensions(canvas);
  const preset = typeof raw.resolutionPreset === 'string' ? raw.resolutionPreset : 'source';
  const storedCustomWidth = clampExportDimension(raw.customWidth, sourceDimensions.width);
  const storedCustomHeight = clampExportDimension(raw.customHeight, sourceDimensions.height);
  const customDimensions = {
    width: storedCustomWidth,
    height: storedCustomHeight,
  };

  let width = raw.width;
  let height = raw.height;

  if (preset === 'custom') {
    width = clampExportDimension(width ?? storedCustomWidth, sourceDimensions.width);
    height = clampExportDimension(height ?? storedCustomHeight, sourceDimensions.height);
    customDimensions.width = width;
    customDimensions.height = height;
  } else {
    const presetDimensions = resolvePresetDimensions(preset, canvas, customDimensions);
    width = clampExportDimension(presetDimensions.width, sourceDimensions.width);
    height = clampExportDimension(presetDimensions.height, sourceDimensions.height);
  }

  const frameRate = clampExportFrameRate(raw.frameRate);
  const format = raw.format === 'webm' ? 'webm' : 'mp4';
  const videoBitsPerSecond = clampExportBitrate(raw.videoBitsPerSecond ?? raw.videoBitrate ?? raw.bitrate);

  return {
    resolutionPreset: preset,
    width,
    height,
    frameRate,
    format,
    videoBitsPerSecond,
    customWidth: customDimensions.width,
    customHeight: customDimensions.height,
  };
}

function loadExportSettings(canvas) {
  const stored = readStorage(EXPORT_STORAGE_KEY);
  if (!stored) {
    return normalizeStoredExportSettings({}, canvas);
  }
  try {
    const parsed = JSON.parse(stored);
    return normalizeStoredExportSettings(parsed, canvas);
  } catch {
    return normalizeStoredExportSettings({}, canvas);
  }
}

function persistExportSettings(settings) {
  if (!settings) {
    return;
  }
  const payload = {
    resolutionPreset: settings.resolutionPreset,
    width: settings.width,
    height: settings.height,
    frameRate: settings.frameRate,
    format: settings.format,
    videoBitsPerSecond: settings.videoBitsPerSecond ?? null,
    customWidth: settings.customWidth,
    customHeight: settings.customHeight,
  };
  writeStorage(EXPORT_STORAGE_KEY, JSON.stringify(payload));
}

function readStoredBoolean(key, defaultValue) {
  const stored = readStorage(key);
  if (stored === null) {
    return defaultValue;
  }
  return stored === '1' || stored.toLowerCase() === 'true';
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

const canvasElement = document.getElementById('c');
const playlistSelect = document.getElementById('playlist');
const playlistAttachButton = document.getElementById('playlist-attach');
const playlistRenameButton = document.getElementById('playlist-rename');
const playlistDeleteButton = document.getElementById('playlist-delete');
const exportButton = document.getElementById('export-video');
const exportDialog = document.getElementById('export-settings');
const exportForm = document.getElementById('export-settings-form');
const exportPresetSelect = document.getElementById('export-resolution');
const exportWidthInput = document.getElementById('export-width');
const exportHeightInput = document.getElementById('export-height');
const exportFpsInput = document.getElementById('export-fps');
const exportBitrateInput = document.getElementById('export-bitrate');
const exportFormatRadios = Array.from(document.querySelectorAll("input[name='export-format']"));
const exportCancelButton = document.getElementById('export-cancel');
const exportCloseButton = document.getElementById('export-close');
const audioElement = document.getElementById('player');
const volumeSlider = document.getElementById('volume');
const playButton = document.getElementById('play');
const prevButton = document.getElementById('prev');
const nextButton = document.getElementById('next');
const seekSlider = document.getElementById('seek');
const fullscreenButton = document.getElementById('fullscreen');
const repeatButton = document.getElementById('repeat');
const byomAttachInput = document.getElementById('byom-attach-input');
const introOverlay = document.getElementById('intro-overlay');
const introPlayButton = document.getElementById('intro-play');
const byomToggleButton = document.getElementById('byom-toggle');
const byomDrawer = document.getElementById('byom-drawer');

const exportSettings = loadExportSettings(canvasElement);

function dismissIntroOverlay() {
  if (!introOverlay || introOverlay.dataset.hidden === 'true') {
    return;
  }
  introOverlay.dataset.hidden = 'true';
  introOverlay.setAttribute('aria-hidden', 'true');
}

function setExportDimensionsDisabled(disabled) {
  if (!exportWidthInput || !exportHeightInput) {
    return;
  }
  exportWidthInput.disabled = disabled;
  exportHeightInput.disabled = disabled;
  exportWidthInput.readOnly = disabled;
  exportHeightInput.readOnly = disabled;
  if (disabled) {
    exportWidthInput.setAttribute('aria-disabled', 'true');
    exportHeightInput.setAttribute('aria-disabled', 'true');
  } else {
    exportWidthInput.removeAttribute('aria-disabled');
    exportHeightInput.removeAttribute('aria-disabled');
  }
}

function refreshExportSettingsFromSource() {
  if (!exportSettings) {
    return;
  }
  const source = resolveSourceDimensions(canvasElement);
  if (exportSettings.resolutionPreset === 'source') {
    exportSettings.width = source.width;
    exportSettings.height = source.height;
  }
  if (!Number.isFinite(exportSettings.customWidth) || exportSettings.customWidth <= 0) {
    exportSettings.customWidth = source.width;
  }
  if (!Number.isFinite(exportSettings.customHeight) || exportSettings.customHeight <= 0) {
    exportSettings.customHeight = source.height;
  }
}

function syncExportResolutionInputs(preset) {
  if (!exportSettings) {
    return;
  }
  const targetPreset = preset || exportSettings.resolutionPreset || 'source';
  exportSettings.resolutionPreset = targetPreset;
  const disableCustom = targetPreset !== 'custom';
  setExportDimensionsDisabled(disableCustom);
  let width;
  let height;
  if (targetPreset === 'custom') {
    width = clampExportDimension(exportSettings.customWidth, exportSettings.width);
    height = clampExportDimension(exportSettings.customHeight, exportSettings.height);
    exportSettings.customWidth = width;
    exportSettings.customHeight = height;
  } else {
    const dims = resolvePresetDimensions(targetPreset, canvasElement, {
      width: exportSettings.customWidth,
      height: exportSettings.customHeight,
    });
    width = clampExportDimension(dims.width, exportSettings.width);
    height = clampExportDimension(dims.height, exportSettings.height);
  }
  exportSettings.width = width;
  exportSettings.height = height;
  if (exportWidthInput) {
    exportWidthInput.value = width;
  }
  if (exportHeightInput) {
    exportHeightInput.value = height;
  }
}

function updateExportFormatRadios(format) {
  const normalized = format === 'webm' ? 'webm' : 'mp4';
  exportFormatRadios.forEach((radio) => {
    if (!radio) {
      return;
    }
    radio.checked = radio.value === normalized;
  });
}

function populateExportDialog() {
  if (!exportSettings) {
    return;
  }
  refreshExportSettingsFromSource();
  const preset = exportSettings.resolutionPreset;
  if (exportPresetSelect) {
    const hasOption = Array.from(exportPresetSelect.options ?? []).some((option) => option.value === preset);
    exportPresetSelect.value = hasOption ? preset : 'source';
    if (!hasOption) {
      exportSettings.resolutionPreset = 'source';
    }
  }
  syncExportResolutionInputs(exportSettings.resolutionPreset);
  if (exportFpsInput) {
    exportSettings.frameRate = clampExportFrameRate(exportSettings.frameRate);
    exportFpsInput.value = exportSettings.frameRate;
  }
  if (exportBitrateInput) {
    exportBitrateInput.value = exportSettings.videoBitsPerSecond
      ? Math.round(exportSettings.videoBitsPerSecond / MEGABIT)
      : '';
    exportBitrateInput.setCustomValidity('');
  }
  updateExportFormatRadios(exportSettings.format);
}

function openExportDialog() {
  if (!exportDialog) {
    return {};
  }
  populateExportDialog();
  exportDialog.dataset.open = 'true';
  if (typeof exportDialog.showModal === 'function') {
    try {
      exportDialog.showModal();
    } catch {
      exportDialog.setAttribute('open', 'true');
    }
  } else {
    exportDialog.setAttribute('open', 'true');
  }
  if (exportPresetSelect) {
    exportPresetSelect.focus({ preventScroll: true });
  }
  return false;
}

function closeExportDialog() {
  if (!exportDialog) {
    return;
  }
  exportDialog.dataset.open = 'false';
  if (typeof exportDialog.close === 'function') {
    try {
      exportDialog.close();
    } catch {
      exportDialog.removeAttribute('open');
    }
  } else {
    exportDialog.removeAttribute('open');
  }
  if (exportButton && typeof exportButton.focus === 'function') {
    exportButton.focus({ preventScroll: true });
  }
}

function handleExportPresetChange() {
  if (!exportSettings) {
    return;
  }
  const preset = exportPresetSelect ? exportPresetSelect.value : exportSettings.resolutionPreset;
  exportSettings.resolutionPreset = preset || 'source';
  syncExportResolutionInputs(exportSettings.resolutionPreset);
}

function handleExportWidthInput() {
  if (!exportSettings || exportSettings.resolutionPreset !== 'custom' || !exportWidthInput) {
    return;
  }
  const width = clampExportDimension(exportWidthInput.value, exportSettings.customWidth);
  exportSettings.customWidth = width;
  exportSettings.width = width;
  exportWidthInput.value = width;
}

function handleExportHeightInput() {
  if (!exportSettings || exportSettings.resolutionPreset !== 'custom' || !exportHeightInput) {
    return;
  }
  const height = clampExportDimension(exportHeightInput.value, exportSettings.customHeight);
  exportSettings.customHeight = height;
  exportSettings.height = height;
  exportHeightInput.value = height;
}

function handleExportFpsChange() {
  if (!exportSettings || !exportFpsInput) {
    return;
  }
  const fps = clampExportFrameRate(exportFpsInput.value);
  exportSettings.frameRate = fps;
  exportFpsInput.value = fps;
}

function handleExportBitrateChange() {
  if (!exportSettings || !exportBitrateInput) {
    return;
  }
  if (exportBitrateInput.value === '') {
    exportSettings.videoBitsPerSecond = null;
    exportBitrateInput.setCustomValidity('');
    return;
  }
  const numeric = Number(exportBitrateInput.value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    exportSettings.videoBitsPerSecond = null;
    exportBitrateInput.setCustomValidity('Bitrate must be greater than zero.');
    return;
  }
  const bits = clampExportBitrate(Math.round(numeric * MEGABIT));
  if (!bits) {
    exportSettings.videoBitsPerSecond = null;
    exportBitrateInput.value = '';
    exportBitrateInput.setCustomValidity('');
    return;
  }
  exportSettings.videoBitsPerSecond = bits;
  exportBitrateInput.value = Math.round(bits / MEGABIT);
  exportBitrateInput.setCustomValidity('');
}

function handleExportFormatChange(event) {
  if (!exportSettings || !event?.target) {
    return;
  }
  const value = event.target.value === 'webm' ? 'webm' : 'mp4';
  exportSettings.format = value;
}

function handleExportFormSubmit(event) {
  if (event) {
    event.preventDefault();
  }
  if (!exportSettings) {
    return;
  }
  const preset = exportPresetSelect ? exportPresetSelect.value : exportSettings.resolutionPreset;
  exportSettings.resolutionPreset = preset || 'source';

  if (exportSettings.resolutionPreset === 'custom') {
    handleExportWidthInput();
    handleExportHeightInput();
  } else {
    const dims = resolvePresetDimensions(exportSettings.resolutionPreset, canvasElement, {
      width: exportSettings.customWidth,
      height: exportSettings.customHeight,
    });
    exportSettings.width = clampExportDimension(dims.width, exportSettings.width);
    exportSettings.height = clampExportDimension(dims.height, exportSettings.height);
  }

  handleExportFpsChange();
  handleExportBitrateChange();

  if (exportBitrateInput && exportBitrateInput.validationMessage) {
    exportBitrateInput.reportValidity();
    return;
  }

  const selectedFormat = exportFormatRadios.find((radio) => radio && radio.checked);
  exportSettings.format = selectedFormat && selectedFormat.value === 'webm' ? 'webm' : 'mp4';

  persistExportSettings(exportSettings);

  const startOptions = {
    resolutionPreset: exportSettings.resolutionPreset,
    width: exportSettings.width,
    height: exportSettings.height,
    frameRate: exportSettings.frameRate,
    format: exportSettings.format,
  };
  if (exportSettings.videoBitsPerSecond) {
    startOptions.videoBitsPerSecond = exportSettings.videoBitsPerSecond;
  }

  const started = videoExport.start(startOptions);
  if (started) {
    closeExportDialog();
  }
}

if (
  !canvasElement ||
  !playlistSelect ||
  !audioElement ||
  !volumeSlider ||
  !playButton ||
  !prevButton ||
  !nextButton ||
  !seekSlider ||
  !fullscreenButton ||
  !repeatButton ||
  !exportButton ||
  !playlistAttachButton ||
  !playlistRenameButton ||
  !playlistDeleteButton ||
  !byomAttachInput ||
  !byomToggleButton ||
  !byomDrawer
) {
  throw new Error(
    'Required controls missing from DOM (canvas, playlist, audio, volume, play, prev, next, seek, repeat, export, playlist actions, fullscreen, or BYOM).',
  );
}

if (exportDialog) {
  exportDialog.addEventListener('cancel', (event) => {
    event.preventDefault();
    closeExportDialog();
  });
  exportDialog.addEventListener('close', () => {
    exportDialog.dataset.open = 'false';
  });
}

if (exportForm) {
  exportForm.addEventListener('submit', handleExportFormSubmit);
}

if (exportCancelButton) {
  exportCancelButton.addEventListener('click', () => {
    closeExportDialog();
  });
}

if (exportCloseButton) {
  exportCloseButton.addEventListener('click', () => {
    closeExportDialog();
  });
}

if (exportPresetSelect) {
  exportPresetSelect.addEventListener('change', handleExportPresetChange);
}

if (exportWidthInput) {
  exportWidthInput.addEventListener('change', handleExportWidthInput);
  exportWidthInput.addEventListener('blur', handleExportWidthInput);
}

if (exportHeightInput) {
  exportHeightInput.addEventListener('change', handleExportHeightInput);
  exportHeightInput.addEventListener('blur', handleExportHeightInput);
}

if (exportFpsInput) {
  exportFpsInput.addEventListener('change', handleExportFpsChange);
  exportFpsInput.addEventListener('blur', handleExportFpsChange);
}

if (exportBitrateInput) {
  exportBitrateInput.addEventListener('change', handleExportBitrateChange);
  exportBitrateInput.addEventListener('blur', handleExportBitrateChange);
}

if (exportFormatRadios.length > 0) {
  exportFormatRadios.forEach((radio) => {
    if (!radio) {
      return;
    }
    radio.addEventListener('change', handleExportFormatChange);
  });
}

populateExportDialog();

initNotifications(document);

render.init();
render.setWorldSize(2, 2);
render.setStatus('Idle · Particles 0');
initAdaptiveFeedback();
updateFullscreenButtonUi(render.getToggles().fullscreen);

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
render.setPalette(activePreset?.palette);
videoExport.init({
  canvas: canvasElement,
  audio: audioElement,
  button: exportButton,
  notify,
  getFileName: resolveExportTitle,
  getStartOptions: openExportDialog,
});
const manualAdjustments = {
  spawnOffset: 0,
  glowOffset: 0,
  sparkleOffset: 0,
  hueOffset: 0,
};
const nnOffsets = {
  spawnOffset: 0,
  glowOffset: 0,
  sparkleOffset: 0,
  hueOffset: 0,
  repelImpulse: 0,
};


function resetManualAdjustments() {
  manualAdjustments.spawnOffset = 0;
  manualAdjustments.glowOffset = 0;
  manualAdjustments.sparkleOffset = 0;
  manualAdjustments.hueOffset = 0;
}

function resetNnOffsets() {
  nnOffsets.spawnOffset = 0;
  nnOffsets.glowOffset = 0;
  nnOffsets.sparkleOffset = 0;
  nnOffsets.hueOffset = 0;
  nnOffsets.repelImpulse = 0;
}

function constrainManualAdjustmentsForSafeMode(enabled) {
  if (!enabled) {
    return;
  }
  manualAdjustments.spawnOffset = Math.min(manualAdjustments.spawnOffset, 0);
  manualAdjustments.glowOffset = Math.min(manualAdjustments.glowOffset, 0);
  manualAdjustments.sparkleOffset = Math.min(manualAdjustments.sparkleOffset, 0);
}

const albumTracks = getList();
if (albumTracks.length === 0) {
  throw new Error('Playlist is empty; Phase 2 requires 11 static tracks.');
}
if (MODEL_FILES.length !== albumTracks.length) {
  throw new Error('Model placeholder count mismatch with playlist length.');
}

const albumEntries = albumTracks.map((track, index) => {
  const basePreset = getPreset(index);
  return {
    id: `album-${index}`,
    type: 'album',
    title: track.title,
    albumIndex: index,
    audioUrl: resolveUrl(index),
    modelUrl: MODEL_FILES[index],
    presetId: basePreset?.id ?? null,
    presetTitle: basePreset?.title ?? track.title ?? `Track ${index + 1}`,
    listIndex: index,
  };
});

let byomEntries = [];
let playlistEntries = [...albumEntries];
rebuildPlaylistOrder();

const storedTrackPreference = readStorage(STORAGE_KEYS.TRACK_INDEX);
const sessionObjectUrls = new Map();

let pendingAttachEntryId = '';

const baseModelOptions = [
  { id: FRESH_MODEL_ID, label: FRESH_MODEL_LABEL },
  ...albumEntries.map((entry) => ({
    id: entry.modelUrl,
    label: entry.title ?? entry.modelUrl,
    url: entry.modelUrl,
  })),
];

function buildStoredModelOption(entry) {
  if (!entry || !entry.id) {
    return null;
  }
  const label = entry.title ?? entry.file?.name ?? entry.id;
  return {
    id: `byom:${entry.id}`,
    label,
    entryId: entry.id,
    modelDefinition: entry.modelDefinition ?? null,
  };
}

function computeModelOptions() {
  const storedOptions = byomEntries
    .map((entry) => buildStoredModelOption(entry))
    .filter(Boolean);
  return [...baseModelOptions, ...storedOptions];
}

function syncByomModelOptions() {
  if (typeof byom.setModelOptions === 'function') {
    byom.setModelOptions(computeModelOptions());
  }
}

function getStoredModelDefinition(modelId) {
  if (typeof modelId !== 'string' || !modelId.startsWith('byom:')) {
    return null;
  }
  const entryId = modelId.slice('byom:'.length);
  const entry = byomEntries.find((candidate) => candidate.id === entryId);
  return entry?.modelDefinition ?? null;
}

function rebuildPlaylistOrder() {
  playlistEntries = [...albumEntries, ...byomEntries];
  playlistEntries.forEach((entry, index) => {
    entry.listIndex = index;
  });
  return playlistEntries;
}

function getEntryByIndex(index) {
  if (!Number.isInteger(index) || index < 0 || index >= playlistEntries.length) {
    return null;
  }
  return playlistEntries[index];
}

function getCurrentEntry() {
  return getEntryByIndex(currentTrackIndex);
}

function isByomEntry(entry) {
  return entry && entry.type === 'byom';
}

function resolveExportTitle() {
  const entry = getCurrentEntry();
  return entry?.title ?? 'Latent Noise';
}

function renderPlaylistOptions(activeIndex = currentTrackIndex) {
  if (!playlistSelect) {
    return;
  }
  playlistSelect.innerHTML = '';

  if (albumEntries.length > 0) {
    const albumGroup = document.createElement('optgroup');
    albumGroup.label = 'Album';
    albumEntries.forEach((entry) => {
      const option = document.createElement('option');
      option.value = String(entry.listIndex);
      option.dataset.entryId = entry.id;
      option.textContent = entry.title;
      albumGroup.append(option);
    });
    playlistSelect.append(albumGroup);
  }

  if (byomEntries.length > 0) {
    const byomGroup = document.createElement('optgroup');
    byomGroup.label = 'BYOM Library';
    byomEntries.forEach((entry) => {
      const option = document.createElement('option');
      option.value = String(entry.listIndex);
      option.dataset.entryId = entry.id;
      option.textContent = entry.requiresFile ? `${entry.title} (attach file)` : entry.title;
      byomGroup.append(option);
    });
    playlistSelect.append(byomGroup);
  }

  if (Number.isInteger(activeIndex) && activeIndex >= 0 && activeIndex < playlistEntries.length) {
    playlistSelect.value = String(activeIndex);
  }
}

function updatePlaylistControls(entry) {
  const isByom = isByomEntry(entry);
  [playlistAttachButton, playlistRenameButton, playlistDeleteButton].forEach((button) => {
    if (!button) {
      return;
    }
    button.hidden = !isByom;
    button.disabled = !isByom;
  });
  if (playlistAttachButton && isByom) {
    playlistAttachButton.textContent = entry && entry.objectUrl ? 'Replace File' : 'Attach File';
  }
}

function storeTrackSelection(entry) {
  if (!entry) {
    return;
  }
  const serialized = isByomEntry(entry)
    ? `byom:${entry.id}`
    : `album:${entry.albumIndex}`;
  writeStorage(STORAGE_KEYS.TRACK_INDEX, serialized);
}

function setEntryObjectUrl(entry, objectUrl, fileInfo) {
  if (!entry || !isByomEntry(entry)) {
    return;
  }
  if (entry.objectUrl && entry.objectUrl !== objectUrl) {
    try {
      URL.revokeObjectURL(entry.objectUrl);
    } catch {
      // Ignore revoke failures.
    }
  }
  entry.objectUrl = objectUrl || '';
  entry.requiresFile = !entry.objectUrl;
  if (fileInfo instanceof File) {
    const signature = `${fileInfo.name}:${fileInfo.size}:${Number.isFinite(fileInfo.lastModified) ? fileInfo.lastModified : 0}`;
    entry.file = {
      name: fileInfo.name,
      size: fileInfo.size,
      lastModified: Number.isFinite(fileInfo.lastModified) ? fileInfo.lastModified : 0,
      signature,
    };
  }
  if (entry.objectUrl) {
    sessionObjectUrls.set(entry.id, entry.objectUrl);
  } else {
    sessionObjectUrls.delete(entry.id);
  }
  renderPlaylistOptions(currentTrackIndex);
  updatePlaylistControls(entry);
}

function promptAttachForEntry(entry, reason = 'attach-file') {
  if (!entry || !isByomEntry(entry)) {
    return;
  }
  pendingAttachEntryId = entry.id;
  const label = entry.file?.name ?? entry.title ?? 'your track';
  const message =
    reason === 'attach-file'
      ? `Select the original MP3 for "${label}" via Attach File to enable playback.`
      : reason === 'object-url-expired'
        ? `The file reference for "${label}" expired. Please re-attach the MP3 to continue.`
        : `Please attach the local MP3 for "${label}".`;
  console.info('[byom] %s', message);
  const notification = notify(message, { tone: 'warning' });
  if (!notification && typeof window !== 'undefined' && typeof window.alert === 'function') {
    window.alert(message);
  }
  updatePlaylistControls(entry);
}

function resolveStoredTrackIndex(reference) {
  rebuildPlaylistOrder();
  if (!reference || typeof reference !== 'string') {
    return 0;
  }
  if (reference.startsWith('byom:')) {
    const id = reference.slice(5);
    const entry = playlistEntries.find((candidate) => candidate.id === id);
    return entry ? entry.listIndex : 0;
  }
  if (reference.startsWith('album:')) {
    const parsed = parseInt(reference.slice(6), 10);
    if (Number.isInteger(parsed) && parsed >= 0 && parsed < albumEntries.length) {
      return albumEntries[parsed].listIndex;
    }
  }
  const fallback = parseInt(reference, 10);
  if (Number.isInteger(fallback) && fallback >= 0 && fallback < playlistEntries.length) {
    return fallback;
  }
  return 0;
}

function createFileMetadata(file, summary) {
  if (file instanceof File) {
    const lastModified = Number.isFinite(file.lastModified) ? file.lastModified : 0;
    return {
      name: file.name,
      size: file.size,
      lastModified,
      signature: `${file.name}:${file.size}:${lastModified}`,
    };
  }
  if (summary) {
    return {
      name: summary.fileName ?? 'unknown',
      size: Number(summary.fileSizeBytes) || 0,
      lastModified: 0,
      signature: '',
    };
  }
  return null;
}

function buildRuntimeByomEntry(record, objectUrl = '') {
  if (!record || typeof record !== 'object') {
    throw new Error('Invalid BYOM record.');
  }
  return {
    id: record.id,
    type: 'byom',
    title: record.name ?? record.file?.name ?? `BYOM ${byomEntries.length + 1}`,
    modelDefinition: record.model ?? null,
    baseline: record.baseline ?? null,
    presetId: record.baseline?.presetId ?? null,
    presetOverrides: record.presetOverrides ?? null,
    summary: record.summary ?? null,
    stats: record.stats ?? null,
    file: record.file ?? null,
    objectUrl: objectUrl || '',
    requiresFile: !objectUrl,
    listIndex: 0,
  };
}

function mergeStoredByomRecords(records, { replace = false } = {}) {
  if (!Array.isArray(records)) {
    return false;
  }
  if (replace) {
    sessionObjectUrls.clear();
    byomEntries = records.map((record) => buildRuntimeByomEntry(record));
    rebuildPlaylistOrder();
    renderPlaylistOptions(currentTrackIndex);
    syncByomModelOptions();
    return byomEntries.length > 0;
  }
  let changed = false;
  records.forEach((record) => {
    if (!record || typeof record !== 'object' || !record.id) {
      return;
    }
    const existingIndex = byomEntries.findIndex((entry) => entry.id === record.id);
    if (existingIndex >= 0) {
      const existing = byomEntries[existingIndex];
      const runtime = buildRuntimeByomEntry(record, existing.objectUrl);
      runtime.listIndex = existing.listIndex;
      byomEntries[existingIndex] = runtime;
    } else {
      byomEntries.push(buildRuntimeByomEntry(record));
    }
    changed = true;
  });
  if (changed) {
    rebuildPlaylistOrder();
    renderPlaylistOptions(currentTrackIndex);
    syncByomModelOptions();
  }
  return changed;
}

async function loadStoredByomEntries() {
  try {
    const stored = await byomStorage.listEntries();
    if (!Array.isArray(stored) || stored.length === 0) {
      return;
    }
    mergeStoredByomRecords(stored, { replace: true });
    const message = 'BYOM models restored. Attach the original MP3 files via Attach File before playback.';
    const notification = notify(message, { tone: 'info' });
    if (!notification && typeof window !== 'undefined' && typeof window.alert === 'function') {
      window.alert(message);
    }
  } catch (error) {
    console.error('[byom] Failed to load stored BYOM entries', error);
  }
}

function handleImportedByomRecords(records) {
  if (!Array.isArray(records) || records.length === 0) {
    return;
  }
  const changed = mergeStoredByomRecords(records);
  if (changed) {
    console.info('[byom] Imported %d stored model%s.', records.length, records.length === 1 ? '' : 's');
  }
}

const initialModelOptions = computeModelOptions();

byom.mount({
  drawer: byomDrawer,
  toggle: byomToggleButton,
  modelOptions: initialModelOptions,
});

let latestTrainingResult = null;
let activeTrainingContext = null;

const trainingController = createTrainingController({
  onStatus: ({ status, detail }) => {
    const update = detail ? { ...detail } : {};
    if (status === 'preparing') {
      update.progress = 0;
      if (!update.message) {
        update.message = 'Preparing training…';
      }
    } else if (status === 'completed') {
      update.progress = 1;
    } else if (status === 'cancelled') {
      update.progress = 0;
      update.message = update.message ?? 'Training cancelled.';
    } else if (status === 'error') {
      update.error = update.error ?? detail;
      if (!update.message && detail) {
        update.message = detail.message ?? String(detail);
      }
      update.progress = 0;
    }
    byom.setTrainingStatus(status, update);
  },
  onProgress: (payload) => {
    byom.updateTrainingProgress(payload);
  },
  onComplete: async ({ modelDefinition, stats, warmup }) => {
    latestTrainingResult = { modelDefinition, stats, warmup };
    console.info('[byom] training completed', stats);
    if (warmup?.outputs) {
      console.info('[byom] warm-up outputs', warmup.outputs);
    }
    const completionMessage = buildCorrelationNotification(stats);
    if (completionMessage) {
      const notification = notify(completionMessage, {
        tone: 'success',
        duration: 8000,
      });
      if (!notification && typeof window !== 'undefined' && typeof window.alert === 'function') {
        window.alert(completionMessage);
      }
    }
    if (typeof window !== 'undefined') {
      window.__LN_LAST_TRAINING__ = latestTrainingResult;
    }
    await finalizeByomTraining({ modelDefinition, stats, warmup });
  },
  onCancelled: (detail) => {
    activeTrainingContext = null;
    byom.setTrainingStatus('cancelled', {
      progress: 0,
      message: detail?.reason === 'cancelled-before-start' ? 'Training cancelled.' : 'Training cancelled.',
    });
  },
  onError: (error) => {
    console.error('[byom] training error', error);
    activeTrainingContext = null;
    byom.setTrainingStatus('error', {
      error,
      message: error?.message ?? 'Training failed.',
      progress: 0,
    });
  },
  onWarmup: (warmup) => {
    if (warmup?.outputs) {
      console.debug('[byom] warm-up sample outputs', warmup.outputs);
    }
  },
});

byom.setHandlers({
  onTrain: ({ file, objectUrl, preset, dataset, summary, model, hyperparameters, correlations }) => {
    if (!dataset || !model) {
      byom.setTrainingStatus('error', { message: 'Training aborted — dataset is unavailable.', progress: 0 });
      return;
    }
    const isFreshModel = model === FRESH_MODEL_ID;
    activeTrainingContext = {
      file: file instanceof File ? file : null,
      objectUrl: typeof objectUrl === 'string' ? objectUrl : '',
      preset: preset || summary?.presetId || null,
      model,
      summary,
      hyperparameters,
      correlations: Array.isArray(correlations) ? correlations.slice() : [],
      mode: isFreshModel ? 'fresh' : 'tune',
    };
    byom.setTrainingStatus('preparing', { progress: 0, message: 'Preparing training…' });
    const inlineDefinition = isFreshModel ? null : getStoredModelDefinition(model);
    const startOptions = {
      dataset,
      summary,
      hyperparameters,
      correlations: Array.isArray(correlations) ? correlations : [],
      mode: isFreshModel ? 'fresh' : 'tune',
    };
    if (!isFreshModel) {
      startOptions.modelUrl = model;
      if (inlineDefinition) {
        startOptions.modelDefinition = inlineDefinition;
      }
    }
    trainingController
      .start(startOptions)
      .catch((error) => {
        console.error('[byom] training start failed', error);
        activeTrainingContext = null;
        byom.setTrainingStatus('error', {
          error,
          message: error?.message ?? 'Training could not start.',
          progress: 0,
        });
      });
  },
  onCancel: ({ training }) => {
    if (training) {
      const cancelled = trainingController.cancel();
      if (cancelled) {
        byom.setTrainingStatus('cancelling', { message: 'Stopping training…' });
        return true;
      }
      return false;
    }
    trainingController.cancel();
    return false;
  },
  onPause: () => {
    const paused = trainingController.pause();
    if (paused) {
      byom.setTrainingStatus('paused');
    }
  },
  onResume: () => {
    const resumed = trainingController.resume();
    if (resumed) {
      byom.setTrainingStatus('running');
    }
  },
  onImportEntries: (records) => {
    handleImportedByomRecords(records);
  },
});

const storedSafeMode = readStoredBoolean(STORAGE_KEYS.SAFE_MODE, false);
const storedBypass = readStoredBoolean(STORAGE_KEYS.NN_BYPASS, false);
const storedRepeat = readStoredBoolean(STORAGE_KEYS.REPEAT, false);

const safeModeEnabled = storedSafeMode;
const nnBypass = storedBypass;
let lastModelOutputs = FALLBACK_NN_OUTPUTS;
let lastHiddenLayers = [];
let currentTrackIndex = -1;

map.configure({ safeMode: safeModeEnabled });
if (safeModeEnabled) {
  constrainManualAdjustmentsForSafeMode(true);
}

initDebugOverlay({ search: typeof window !== 'undefined' ? window.location.search : '' });
await runStartupDiagnostics({ safeMode: safeModeEnabled });

rebuildPlaylistOrder();
renderPlaylistOptions(currentTrackIndex);

await loadStoredByomEntries();

const initialTrackIndex = resolveStoredTrackIndex(storedTrackPreference);
const initialEntry = getEntryByIndex(initialTrackIndex);
render.setTrackTitle(initialEntry?.title ?? 'Latent Noise');
updatePlaylistControls(initialEntry);

const modelCache = new Map();
let activeModelEntryId = '';
let modelLoadToken = 0;

const playback = {
  status: 'Idle',
  lastStatusText: '',
};

const repeatController = createRepeatController({
  initialEnabled: storedRepeat,
  onChange(enabled) {
    updateRepeatButtonUi(enabled);
    writeStorage(STORAGE_KEYS.REPEAT, enabled ? '1' : '0');
  },
});

let autoAdvanceTimer = 0;
let pendingPlayTimer = 0;
let pendingPlayToken = 0;

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

function resolveAdaptiveFeedbackController() {
  if (typeof window === 'undefined') {
    return null;
  }
  return window.lnAdaptiveFeedback ?? window.__LN_ADAPTIVE_FEEDBACK__ ?? null;
}

function parseAvailabilityPayload(payload) {
  if (payload == null) {
    return false;
  }
  if (typeof payload === 'boolean') {
    return payload;
  }
  if (typeof payload === 'number') {
    return payload !== 0;
  }
  if (typeof payload === 'string') {
    const normalized = payload.trim().toLowerCase();
    if (!normalized) {
      return false;
    }
    if (['false', '0', 'off', 'disabled', 'no'].includes(normalized)) {
      return false;
    }
    if (['true', '1', 'on', 'enabled', 'yes', 'ready', 'available'].includes(normalized)) {
      return true;
    }
    return Boolean(normalized);
  }
  if (typeof payload === 'object') {
    if (typeof payload.available !== 'undefined') {
      return Boolean(payload.available);
    }
    if (typeof payload.value !== 'undefined') {
      return parseAvailabilityPayload(payload.value);
    }
    if (typeof payload.detail !== 'undefined') {
      return parseAvailabilityPayload(payload.detail);
    }
    if (typeof payload.type === 'string') {
      return adaptiveFeedbackState.available;
    }
  }
  return Boolean(payload);
}

function resolveAdaptiveFeedbackAvailability(controller) {
  if (!controller) {
    return false;
  }
  try {
    if (typeof controller.isAvailable === 'function') {
      return Boolean(controller.isAvailable());
    }
    if (typeof controller.getAvailability === 'function') {
      return Boolean(controller.getAvailability());
    }
    if (typeof controller.available === 'boolean') {
      return controller.available;
    }
    if (controller.state && typeof controller.state.available === 'boolean') {
      return controller.state.available;
    }
  } catch (error) {
    console.warn('[app] adaptive feedback availability check failed', error);
    return false;
  }
  return true;
}

function cancelFeedbackResetTimer() {
  if (adaptiveFeedbackState.resetTimer && typeof window !== 'undefined') {
    window.clearTimeout(adaptiveFeedbackState.resetTimer);
    adaptiveFeedbackState.resetTimer = 0;
  }
}

function scheduleFeedbackReset() {
  if (typeof window === 'undefined') {
    return;
  }
  cancelFeedbackResetTimer();
  adaptiveFeedbackState.resetTimer = window.setTimeout(() => {
    adaptiveFeedbackState.resetTimer = 0;
    if (adaptiveFeedbackState.available) {
      render.setFeedbackStatus(FEEDBACK_STATUS_MESSAGES.ready, 'ready');
    } else {
      render.setFeedbackStatus(FEEDBACK_STATUS_MESSAGES.offline, 'offline');
    }
    render.setFeedbackSelection('none');
  }, FEEDBACK_STATUS_RESET_MS);
}

function updateAdaptiveFeedbackAvailability(available, options = {}) {
  const normalized = Boolean(available);
  adaptiveFeedbackState.available = normalized;
  render.setFeedbackAvailability(normalized, {
    text: options.text ?? FEEDBACK_STATUS_MESSAGES[normalized ? 'ready' : 'offline'],
    status: options.status ?? (normalized ? 'ready' : 'offline'),
  });
  if (!normalized) {
    cancelFeedbackResetTimer();
  }
  render.setFeedbackSelection('none');
}

function bindAdaptiveFeedbackController(controller) {
  if (!controller) {
    return;
  }
  const handler = (value) => {
    updateAdaptiveFeedbackAvailability(parseAvailabilityPayload(value));
  };

  let bound = false;
  if (typeof controller.on === 'function') {
    try {
      controller.on('availabilitychange', handler);
      bound = true;
    } catch (error) {
      console.warn('[app] adaptive feedback controller.on binding failed', error);
    }
  }
  if (!bound && typeof controller.addEventListener === 'function') {
    try {
      controller.addEventListener('availabilitychange', handler);
      bound = true;
    } catch (error) {
      console.warn('[app] adaptive feedback controller.addEventListener binding failed', error);
    }
  }
  if (!bound && typeof controller.addListener === 'function') {
    try {
      controller.addListener('availabilitychange', handler);
      bound = true;
    } catch (error) {
      console.warn('[app] adaptive feedback controller.addListener binding failed', error);
    }
  }
  if (!bound && typeof controller.onAvailabilityChange === 'function') {
    try {
      controller.onAvailabilityChange(handler);
    } catch (error) {
      console.warn('[app] adaptive feedback controller.onAvailabilityChange binding failed', error);
    }
  }
}

function submitAdaptiveFeedback(direction, payload) {
  const controller = adaptiveFeedbackState.controller;
  if (!controller) {
    return false;
  }
  if (typeof controller.submitFeedback === 'function') {
    return controller.submitFeedback(payload);
  }
  if (typeof controller.sendFeedback === 'function') {
    return controller.sendFeedback(payload);
  }
  if (typeof controller.handleFeedback === 'function') {
    return controller.handleFeedback(payload);
  }
  if (typeof controller.send === 'function') {
    return controller.send(payload);
  }
  if (typeof controller.submit === 'function') {
    return controller.submit(payload);
  }
  if (typeof controller.dispatch === 'function') {
    return controller.dispatch('feedback', payload);
  }
  if (typeof controller === 'function') {
    return controller(payload);
  }
  return true;
}

function handleFeedbackEvent(detail = {}) {
  const direction = detail?.direction;
  if (direction !== 'positive' && direction !== 'negative') {
    return;
  }
  if (!adaptiveFeedbackState.available) {
    return;
  }
  const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
  if (now - adaptiveFeedbackState.lastSentAt < FEEDBACK_DEBOUNCE_MS) {
    return;
  }
  adaptiveFeedbackState.lastSentAt = now;
  cancelFeedbackResetTimer();
  render.setFeedbackStatus(FEEDBACK_STATUS_MESSAGES.pending, 'pending');

  const payload = {
    direction,
    source: detail?.source ?? 'hud',
    timestamp: Number.isFinite(detail?.timestamp) ? detail.timestamp : now,
  };

  let result;
  try {
    result = submitAdaptiveFeedback(direction, payload);
  } catch (error) {
    console.error('[app] adaptive feedback dispatch failed', error);
    render.setFeedbackStatus(FEEDBACK_STATUS_MESSAGES.error, 'error');
    render.setFeedbackSelection('none');
    scheduleFeedbackReset();
    return;
  }

  if (result && typeof result.then === 'function') {
    result
      .then((response) => {
        if (response === false) {
          render.setFeedbackStatus(FEEDBACK_STATUS_MESSAGES.error, 'error');
        } else {
          render.setFeedbackStatus(FEEDBACK_STATUS_MESSAGES.success, 'success');
        }
        render.setFeedbackSelection('none');
        scheduleFeedbackReset();
      })
      .catch((error) => {
        console.error('[app] adaptive feedback rejected', error);
        render.setFeedbackStatus(FEEDBACK_STATUS_MESSAGES.error, 'error');
        render.setFeedbackSelection('none');
        scheduleFeedbackReset();
      });
    return;
  }

  if (result === false) {
    render.setFeedbackStatus(FEEDBACK_STATUS_MESSAGES.error, 'error');
  } else {
    render.setFeedbackStatus(FEEDBACK_STATUS_MESSAGES.success, 'success');
  }
  render.setFeedbackSelection('none');
  scheduleFeedbackReset();
}

function initAdaptiveFeedback() {
  const controller = resolveAdaptiveFeedbackController();
  adaptiveFeedbackState.controller = controller;
  updateAdaptiveFeedbackAvailability(resolveAdaptiveFeedbackAvailability(controller));
  bindAdaptiveFeedbackController(controller);

  if (typeof window !== 'undefined') {
    window.addEventListener('ln:feedback-availability', (event) => {
      updateAdaptiveFeedbackAvailability(parseAvailabilityPayload(event));
    });
    window.addEventListener('ln:feedback-controller', (event) => {
      const detail = event?.detail ?? {};
      const candidate = detail.controller ?? detail.instance ?? detail;
      if (!candidate) {
        return;
      }
      adaptiveFeedbackState.controller = candidate;
      updateAdaptiveFeedbackAvailability(resolveAdaptiveFeedbackAvailability(candidate));
      bindAdaptiveFeedbackController(candidate);
    });
  }
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

function applyPresetForEntry(entry, options = {}) {
  let preset = null;
  if (entry) {
    if (isByomEntry(entry)) {
      if (entry.presetId) {
        preset = getPreset(entry.presetId);
      }
      if (!preset && entry.presetTitle) {
        preset = getPreset(entry.presetTitle);
      }
    } else if (entry.type === 'album') {
      preset = getPreset(entry.albumIndex);
    }
  }
  if (!preset) {
    preset = activePreset ?? getDefaultPreset();
  }

  activePreset = preset;
  if (preset?.palette) {
    render.setPalette(preset.palette);
  }

  const forceSilence = options.forceSilence === true;

  resetManualAdjustments();
  resetNnOffsets();
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

  if (entry && entry.presetOverrides) {
    if (entry.presetOverrides.sim) {
      copyParams(simParams, entry.presetOverrides.sim);
    }
    if (entry.presetOverrides.render) {
      copyParams(renderParams, entry.presetOverrides.render);
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

  if (forceSilence) {
    const restParams = map.update(FALLBACK_NN_OUTPUTS, {
      dt: 1 / 60,
      activity: 0,
      forceSilence: true,
    });
    applyMappedParams(restParams);
    physics.reset();
  }

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

function updateRepeatButtonUi(nextState = repeatController.isEnabled()) {
  if (!repeatButton) {
    return;
  }
  const active = Boolean(nextState);
  repeatButton.textContent = active ? 'Repeat On' : 'Repeat';
  repeatButton.setAttribute('aria-pressed', active ? 'true' : 'false');
  repeatButton.setAttribute('title', active ? 'Disable repeat' : 'Repeat current track');
}

function updateFullscreenButtonUi(active) {
  if (!fullscreenButton) {
    return;
  }
  const pressed = Boolean(active);
  fullscreenButton.textContent = pressed ? 'Exit Fullscreen' : 'Fullscreen';
  fullscreenButton.setAttribute('aria-pressed', pressed ? 'true' : 'false');
}

function updateSeekUi(currentSeconds, durationSeconds) {
  if (!seekSlider) {
    return;
  }
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
    seekSlider.value = '0';
    seekSlider.disabled = true;
    seekSlider.style.setProperty('--value', '0%');
    return;
  }
  const percent = clamp((currentSeconds / durationSeconds) * 100, 0, 100);
  const formatted = percent.toFixed(2);
  seekSlider.value = formatted;
  seekSlider.disabled = false;
  seekSlider.style.setProperty('--value', `${formatted}%`);
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
  const zoomBaseClamped = clamp(zoomBase, ZOOM_SOURCE_MIN, ZOOM_SOURCE_MAX);
  const zoomNormalized = (zoomBaseClamped - ZOOM_SOURCE_MIN) / ZOOM_SOURCE_RANGE;
  const zoomScaled = ZOOM_OUTPUT_MIN + (ZOOM_OUTPUT_MAX - ZOOM_OUTPUT_MIN) * zoomNormalized;

  nnOffsets.spawnOffset = Number.isFinite(mapped.spawnOffset) ? mapped.spawnOffset : 0;
  nnOffsets.glowOffset = Number.isFinite(mapped.glowOffset) ? mapped.glowOffset : 0;
  nnOffsets.sparkleOffset = Number.isFinite(mapped.sparkleOffset) ? mapped.sparkleOffset : 0;
  nnOffsets.hueOffset = Number.isFinite(mapped.hueOffset) ? mapped.hueOffset : 0;
  nnOffsets.repelImpulse = clamp(repelBase, 0, 1);

  const spawnMin = 0;
  const spawnMax = safe ? 0.8 : 1.2;
  const glowMax = safe ? 0.6 : 1;
  const sparkleMax = safe ? 0.65 : 1;
  const zoomMin = ZOOM_OUTPUT_MIN;
  const zoomMax = safe ? 1.5 : ZOOM_OUTPUT_MAX;

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
  renderParams.zoom = clamp(zoomScaled, zoomMin, zoomMax);
}

function cacheEntryIsPromise(entry) {
  return entry && typeof entry === 'object' && typeof entry.then === 'function';
}

async function fetchModelDefinitionForEntry(entry) {
  if (!entry) {
    throw new Error('Playlist entry is required to load a model.');
  }
  const cacheKey = entry.id;
  const existing = modelCache.get(cacheKey);
  if (existing) {
    if (cacheEntryIsPromise(existing)) {
      return existing;
    }
    return existing;
  }

  if (entry.type === 'album') {
    const url = entry.modelUrl;
    if (!url) {
      throw new RangeError(`Model path missing for playlist entry ${entry.id}`);
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
        modelCache.set(cacheKey, json);
        return json;
      })
      .catch((error) => {
        modelCache.delete(cacheKey);
        throw error;
      });

    modelCache.set(cacheKey, fetchPromise);
    return fetchPromise;
  }

  if (isByomEntry(entry)) {
    if (!entry.modelDefinition) {
      throw new Error(`BYOM entry "${entry.title ?? entry.id}" is missing a model definition.`);
    }
    try {
      validateModelDefinition(entry.modelDefinition, entry.title ?? entry.id);
    } catch (error) {
      console.error('[app] Invalid stored BYOM model', error);
      throw error;
    }
    modelCache.set(cacheKey, entry.modelDefinition);
    return entry.modelDefinition;
  }

  throw new Error(`Unsupported playlist entry type "${entry.type}".`);
}

async function prepareModelForEntry(entry) {
  if (!entry) {
    return null;
  }
  const token = ++modelLoadToken;
  try {
    const definition = await fetchModelDefinitionForEntry(entry);
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
    lastHiddenLayers = warmupOutputs ? nn.getHiddenLayerActivations() : [];
    activeModelEntryId = entry.id;
    if (info) {
      console.info(`[app] Model ready for "${entry.title ?? entry.id}" (${info.layers} layers)`);
    }
    return info;
  } catch (error) {
    console.error(`[app] Failed to load model for "${entry.title ?? entry.id}"`, error);
    return null;
  }
}

async function finalizeByomTraining({ modelDefinition, stats }) {
  if (!modelDefinition) {
    console.warn('[byom] Training result missing model definition; cannot persist entry.');
    activeTrainingContext = null;
    return;
  }
  if (!activeTrainingContext) {
    console.warn('[byom] Training context lost; skipping BYOM persistence.');
    return;
  }
  const context = activeTrainingContext;
  activeTrainingContext = null;

  let playbackUrl = '';
  if (context.file instanceof File) {
    try {
      playbackUrl = URL.createObjectURL(context.file);
    } catch (error) {
      console.warn('[byom] Failed to create playback Object URL from File', error);
      playbackUrl = context.objectUrl || '';
    }
  } else if (context.objectUrl) {
    playbackUrl = context.objectUrl;
  }

  const fileMeta = createFileMetadata(context.file, context.summary);
  const entryName =
    context.summary?.fileName
    || fileMeta?.name
    || `BYOM ${new Date().toLocaleTimeString()}`;

  const baseline = {
    presetId: context.preset ?? null,
    modelId: context.model,
  };
  if (context.hyperparameters) {
    baseline.hyperparameters = { ...context.hyperparameters };
  }

  let persisted;
  try {
    const payload = byomStorage.createEntryPayload({
      name: entryName,
      baseline,
      file: fileMeta,
      summary: context.summary ?? null,
      stats: stats ?? null,
      model: modelDefinition,
      version: 1,
    });
    persisted = await byomStorage.putEntry(payload, {
      name: entryName,
      inputs: modelDefinition?.input,
      outputs: Array.isArray(modelDefinition?.layers)
        ? modelDefinition.layers.at(-1)?.bias?.length
        : undefined,
    });
  } catch (error) {
    console.error('[byom] Failed to persist trained BYOM entry', error);
    return;
  }

  const runtimeEntry = buildRuntimeByomEntry(persisted, playbackUrl);
  if (context.presetOverrides) {
    runtimeEntry.presetOverrides = { ...context.presetOverrides };
  }
  if (playbackUrl) {
    sessionObjectUrls.set(runtimeEntry.id, playbackUrl);
    runtimeEntry.requiresFile = false;
  }
  byomEntries.push(runtimeEntry);
  await byom.refreshManagerEntries({ silent: true });
  rebuildPlaylistOrder();
  renderPlaylistOptions(runtimeEntry.listIndex);
  updatePlaylistControls(runtimeEntry);
  syncByomModelOptions();
  console.info('[byom] Stored BYOM entry "%s".', runtimeEntry.title);

  if (typeof byom.close === 'function') {
    byom.close({ restoreFocus: false });
  }

  if (playbackUrl) {
    currentTrackIndex = -1;
    setTrack(runtimeEntry.listIndex, { autoplay: true, autoplayDelayMs: 0 });
  } else {
    promptAttachForEntry(runtimeEntry, 'object-url-expired');
  }
}

function setTrack(index, options = {}) {
  clearAutoAdvanceTimer();
  videoExport.cancel({ silent: true });
  if (!Number.isInteger(index) || index < 0 || index >= playlistEntries.length) {
    console.warn('[app] Ignoring out-of-range track index', index);
    return;
  }
  if (pendingPlayTimer) {
    window.clearTimeout(pendingPlayTimer);
    pendingPlayTimer = 0;
  }
  const playToken = ++pendingPlayToken;
  const entry = getEntryByIndex(index);
  if (!entry) {
    return;
  }
  const autoplay = options.autoplay ?? !audioElement.paused;
  const autoplayDelayMs = Number.isFinite(options.autoplayDelayMs)
    ? Math.max(0, options.autoplayDelayMs)
    : 0;
  const forceReload = options.forceReload === true;

  if (isByomEntry(entry) && !entry.objectUrl) {
    currentTrackIndex = index;
    playlistSelect.value = String(index);
    storeTrackSelection(entry);
    updatePlaylistControls(entry);
    promptAttachForEntry(entry, 'attach-file');
    audioElement.pause();
    audioElement.removeAttribute('src');
    audioElement.load();
    activeModelEntryId = '';
    lastModelOutputs = FALLBACK_NN_OUTPUTS;
    lastHiddenLayers = [];
    playback.status = 'Idle';
    updateStatus(physics.getMetrics());
    render.setTrackTitle(entry.title ?? `Track ${index + 1}`);
    render.updateTrackTime(0, NaN);
    updateSeekUi(0, NaN);
    updatePlayButtonUi();
    return;
  }

  if (!forceReload && index === currentTrackIndex && activeModelEntryId === entry.id && entry.type === 'album') {
    return;
  }

  currentTrackIndex = index;
  playlistSelect.value = String(index);
  storeTrackSelection(entry);
  updatePlaylistControls(entry);

  if (isByomEntry(entry)) {
    audioElement.src = entry.objectUrl;
  } else {
    audioElement.src = entry.audioUrl;
  }
  if (forceReload) {
    audioElement.load();
  }

  const preset = applyPresetForEntry(entry, { forceSilence: !autoplay });
  render.setTrackTitle(entry.title ?? `Track ${index + 1}`);
  render.updateTrackTime(0, Number.isFinite(audioElement.duration) ? audioElement.duration : NaN);
  updateSeekUi(0, NaN);
  playback.status = autoplay ? 'Buffering' : 'Idle';
  updateStatus(physics.getMetrics());
  lastModelOutputs = FALLBACK_NN_OUTPUTS;
  lastHiddenLayers = [];

  if (preset) {
    console.info('[app] Applied preset:', preset.title);
  }

  void prepareModelForEntry(entry);

  if (!autoplay || autoplayDelayMs > 0) {
    audioElement.pause();
  }

  if (autoplay) {
    if (autoplayDelayMs > 0) {
      pendingPlayTimer = window.setTimeout(() => {
        pendingPlayTimer = 0;
        if (playToken !== pendingPlayToken || currentTrackIndex !== index) {
          return;
        }
        audioElement.play().catch((error) => {
          playback.status = 'Idle';
          updateStatus(physics.getMetrics());
          console.warn('[app] Autoplay blocked', error);
        });
      }, autoplayDelayMs);
    } else {
      audioElement.play().catch((error) => {
        playback.status = 'Idle';
        updateStatus(physics.getMetrics());
        console.warn('[app] Autoplay blocked', error);
      });
    }
  }

  updatePlayButtonUi();
}

function nextTrack(step = 1, options = {}) {
  if (playlistEntries.length === 0) {
    return;
  }
  const nextIndex = (currentTrackIndex + step + playlistEntries.length) % playlistEntries.length;
  const autoplay = options.autoplay ?? !audioElement.paused;
  const autoplayDelayMs = Number.isFinite(options.autoplayDelayMs)
    ? Math.max(0, options.autoplayDelayMs)
    : 0;
  if (!options.skipIntermission && nextIndex !== currentTrackIndex && currentTrackIndex >= 0) {
    const duration = Number.isFinite(options.intermissionDuration)
      ? options.intermissionDuration
      : TRACK_INTERMISSION_MS;
    startParticleIntermission(duration);
  }
  setTrack(nextIndex, { autoplay, autoplayDelayMs });
}

function prevTrack(options = {}) {
  nextTrack(-1, options);
}

function togglePlayback() {
  dismissIntroOverlay();
  if (audioElement.paused) {
    audioElement.play().catch((error) => {
      console.warn('[app] Playback start blocked', error);
    });
  } else {
    audioElement.pause();
  }
}

function startExperience() {
  audio
    .unlock()
    .catch(() => {
      // Mobile Safari may throw when attempting to unlock before the context exists.
    });
  render.setToggle('fullscreen', true);
  if (!audioElement.paused) {
    dismissIntroOverlay();
    return;
  }

  dismissIntroOverlay();

  if (pendingPlayTimer) {
    window.clearTimeout(pendingPlayTimer);
    pendingPlayTimer = 0;
  }

  const playToken = ++pendingPlayToken;
  const introDelayMs = TRACK_INTERMISSION_MS;
  playback.status = 'Buffering';
  updateStatus(physics.getMetrics());

  pendingPlayTimer = window.setTimeout(() => {
    pendingPlayTimer = 0;
    if (playToken !== pendingPlayToken || !audioElement.paused) {
      return;
    }
    audioElement.play().catch((error) => {
      playback.status = 'Idle';
      updateStatus(physics.getMetrics());
      console.warn('[app] Playback start blocked', error);
    });
  }, introDelayMs);
}

function handleIntroStart(event) {
  if (event && typeof event.preventDefault === 'function' && event.type !== 'click') {
    event.preventDefault();
  }
  if (introOverlay && introOverlay.dataset.hidden === 'true' && !audioElement.paused) {
    return;
  }
  startExperience();
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
updateRepeatButtonUi();

if (introPlayButton) {
  const supportsPointer = typeof window !== 'undefined' && 'PointerEvent' in window;
  if (supportsPointer) {
    introPlayButton.addEventListener('pointerup', handleIntroStart, { passive: false });
  } else {
    introPlayButton.addEventListener('touchend', handleIntroStart, { passive: false });
  }
  introPlayButton.addEventListener('click', handleIntroStart);
}

volumeSlider.addEventListener('input', () => {
  const nextVolume = Number(volumeSlider.value);
  if (Number.isNaN(nextVolume)) {
    return;
  }
  audio.setVolume(nextVolume);
  render.updateVolume(nextVolume);
});

fullscreenButton.addEventListener('click', () => {
  const toggles = render.getToggles();
  render.setToggle('fullscreen', !toggles.fullscreen);
});

repeatButton.addEventListener('click', () => {
  repeatController.toggle();
});

playButton.addEventListener('click', () => {
  togglePlayback();
});

prevButton.addEventListener('click', () => {
  prevTrack({ autoplayDelayMs: TRACK_INTERMISSION_MS });
});

nextButton.addEventListener('click', () => {
  nextTrack(1, { autoplayDelayMs: TRACK_INTERMISSION_MS });
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
  const entry = getEntryByIndex(selected);
  updatePlaylistControls(entry);
  if (selected !== currentTrackIndex && currentTrackIndex >= 0) {
    startParticleIntermission(TRACK_INTERMISSION_MS);
  }
  setTrack(selected, {
    autoplay: !audioElement.paused,
    autoplayDelayMs: TRACK_INTERMISSION_MS,
  });
});

playlistAttachButton.addEventListener('click', () => {
  const entry = getCurrentEntry();
  if (!isByomEntry(entry)) {
    return;
  }
  pendingAttachEntryId = entry.id;
  byomAttachInput.value = '';
  byomAttachInput.click();
});

playlistRenameButton.addEventListener('click', async () => {
  const entry = getCurrentEntry();
  if (!isByomEntry(entry)) {
    return;
  }
  const currentName = entry.title ?? entry.file?.name ?? '';
  const nextName = typeof window !== 'undefined' && typeof window.prompt === 'function'
    ? window.prompt('Rename BYOM entry', currentName)
    : currentName;
  if (!nextName) {
    return;
  }
  const trimmed = nextName.trim();
  if (!trimmed || trimmed === entry.title) {
    return;
  }
  entry.title = trimmed;
  try {
    await byomStorage.renameEntry(entry.id, trimmed);
  } catch (error) {
    console.error('[byom] Failed to rename entry', error);
  }
  renderPlaylistOptions(currentTrackIndex);
  if (currentTrackIndex >= 0 && playlistEntries[currentTrackIndex]?.id === entry.id) {
    render.setTrackTitle(entry.title);
  }
  updatePlaylistControls(entry);
  syncByomModelOptions();
});

playlistDeleteButton.addEventListener('click', async () => {
  const entry = getCurrentEntry();
  if (!isByomEntry(entry)) {
    return;
  }
  const confirmed = typeof window !== 'undefined' && typeof window.confirm === 'function'
    ? window.confirm(`Delete "${entry.title ?? entry.file?.name ?? 'BYOM entry'}"? This cannot be undone.`)
    : true;
  if (!confirmed) {
    return;
  }

  if (entry.objectUrl) {
    try {
      URL.revokeObjectURL(entry.objectUrl);
    } catch {
      // Ignore revoke errors.
    }
  }
  sessionObjectUrls.delete(entry.id);
  modelCache.delete(entry.id);
  if (activeModelEntryId === entry.id) {
    activeModelEntryId = '';
  }

  const activeEntryBeforeDelete = getCurrentEntry();
  const wasPlayingDeleted = activeEntryBeforeDelete && activeEntryBeforeDelete.id === entry.id;

  byomEntries = byomEntries.filter((candidate) => candidate.id !== entry.id);
  rebuildPlaylistOrder();
  renderPlaylistOptions(currentTrackIndex);
  syncByomModelOptions();

  try {
    await byomStorage.deleteEntry(entry.id);
  } catch (error) {
    console.error('[byom] Failed to delete BYOM entry', error);
  }

  if (playlistEntries.length === 0) {
    currentTrackIndex = -1;
    audioElement.pause();
    render.setTrackTitle('Latent Noise');
    updatePlaylistControls(null);
    updateStatus(physics.getMetrics());
    return;
  }

  if (wasPlayingDeleted) {
    const fallbackIndex = Math.min(entry.listIndex ?? 0, playlistEntries.length - 1);
    currentTrackIndex = -1;
    setTrack(fallbackIndex, { autoplay: false });
  } else if (activeEntryBeforeDelete) {
    const activeIndex = playlistEntries.findIndex((candidate) => candidate.id === activeEntryBeforeDelete.id);
    if (activeIndex >= 0) {
      currentTrackIndex = activeIndex;
      playlistSelect.value = String(activeIndex);
    } else {
      currentTrackIndex = -1;
    }
    updatePlaylistControls(getCurrentEntry());
  } else {
    updatePlaylistControls(getCurrentEntry());
  }
});

byomAttachInput.addEventListener('change', async () => {
  const files = byomAttachInput.files;
  const file = files && files.length > 0 ? files[0] : null;
  const entryId = pendingAttachEntryId || getCurrentEntry()?.id;
  pendingAttachEntryId = '';
  byomAttachInput.value = '';
  if (!file || !entryId) {
    return;
  }
  const entry = playlistEntries.find((candidate) => candidate.id === entryId);
  if (!entry || !isByomEntry(entry)) {
    return;
  }
  let objectUrl = '';
  try {
    objectUrl = URL.createObjectURL(file);
  } catch (error) {
    console.error('[byom] Failed to create Object URL for BYOM file', error);
    return;
  }
  setEntryObjectUrl(entry, objectUrl, file);
  try {
    await byomStorage.updateEntry(entry.id, { file: entry.file });
  } catch (error) {
    console.warn('[byom] Failed to persist BYOM file metadata', error);
  }
  if (currentTrackIndex === entry.listIndex) {
    setTrack(entry.listIndex, {
      autoplay: !audioElement.paused,
      autoplayDelayMs: 0,
    });
  }
});

render.on('playToggle', togglePlayback);
render.on('nextTrack', () => nextTrack(1, { autoplayDelayMs: TRACK_INTERMISSION_MS }));
render.on('prevTrack', () => prevTrack({ autoplayDelayMs: TRACK_INTERMISSION_MS }));
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
  if (playlistEntries.length === 0) {
    return;
  }
  const nextIndex = (index + playlistEntries.length) % playlistEntries.length;
  const entry = getEntryByIndex(nextIndex);
  updatePlaylistControls(entry);
  if (nextIndex !== currentTrackIndex && currentTrackIndex >= 0) {
    startParticleIntermission(TRACK_INTERMISSION_MS);
  }
  setTrack(nextIndex, { autoplay: !audioElement.paused });
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
render.on('toggle', ({ name, value }) => {
  if (name === 'fullscreen') {
    updateFullscreenButtonUi(Boolean(value));
  }
});
render.on('feedback', handleFeedbackEvent);

audioElement.addEventListener('play', () => {
  dismissIntroOverlay();
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
  const endedIndex = currentTrackIndex;
  startParticleIntermission(TRACK_INTERMISSION_MS);
  autoAdvanceTimer = window.setTimeout(() => {
    autoAdvanceTimer = 0;
    if (repeatController.isEnabled()) {
      setTrack(endedIndex, {
        autoplay: true,
        autoplayDelayMs: 0,
        forceReload: true,
      });
      return;
    }
    nextTrack(1, { autoplay: true, skipIntermission: true });
  }, TRACK_INTERMISSION_MS);
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

  const lookAheadTimestamp = now + ANIMATION_LOOKAHEAD_MS;

  fpsMonitor.sample(frameTimeMs);
  const averageFps = fpsMonitor.getAverageFps();
  const averageFrameTime = fpsMonitor.getAverageFrameTime();
  const instantaneousFps = fpsMonitor.getInstantaneousFps();

  updatePerformanceScaling(averageFps);

  const audioState = audio.frame();
  const features = audioState?.features ?? audio.getFeatureVector();
  const activity = Number.isFinite(audioState?.activity)
    ? Math.min(Math.max(audioState.activity, 0), 1)
    : audio.getActivityLevel(audioState?.rms ?? 0);

  const currentEntry = getCurrentEntry();
  let nnOutputs = lastModelOutputs;
  let hiddenActivations = lastHiddenLayers;
  if (!nnBypass && currentEntry && activeModelEntryId === currentEntry.id) {
    try {
      const normalized = nn.normalize(features);
      nnOutputs = nn.forward(normalized);
      hiddenActivations = nn.getHiddenLayerActivations();
      lastModelOutputs = nnOutputs;
      lastHiddenLayers = hiddenActivations;
    } catch (error) {
      console.warn('[app] NN inference failed; using fallback outputs.', error);
      nnOutputs = FALLBACK_NN_OUTPUTS;
      hiddenActivations = [];
      lastModelOutputs = FALLBACK_NN_OUTPUTS;
      lastHiddenLayers = [];
    }
  } else if (nnBypass) {
    nnOutputs = FALLBACK_NN_OUTPUTS;
    hiddenActivations = [];
    lastModelOutputs = FALLBACK_NN_OUTPUTS;
    lastHiddenLayers = [];
  }

  const playbackSilent =
    !audioElement
    || audioElement.paused
    || audioElement.ended
    || audioElement.readyState < 2;

  const mappedParams = map.update(nnOutputs, {
    dt: dtSeconds,
    timestamp: lookAheadTimestamp,
    activity,
    features,
    forceSilence: playbackSilent,
  });
  applyMappedParams(mappedParams);

  const intermissionActive = particleIntermissionUntil > now;
  if (intermissionActive) {
    simParams.spawnRate = 0;
  } else if (particleIntermissionUntil !== 0) {
    particleIntermissionUntil = 0;
  }

  physics.step(simParams, { dt: dtSeconds, frameTime: frameTimeMs, frameTimeAvg: averageFrameTime });
  const particles = physics.getParticles();
  const metrics = physics.getMetrics();

  render.renderFrame(particles, renderParams, {
    dt: dtSeconds,
    frameTime: frameTimeMs,
    frameTimeAvg: averageFrameTime,
    fps: instantaneousFps,
    fpsAvg: averageFps,
  }, {
    features,
    featureLabels: FEATURE_LABELS,
    outputs: nnOutputs,
    outputLabels: OUTPUT_LABELS,
    hiddenLayers: hiddenActivations,
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
      spawnOffset: manualAdjustments.spawnOffset + nnOffsets.spawnOffset,
      glowOffset: manualAdjustments.glowOffset + nnOffsets.glowOffset,
      sparkleOffset: manualAdjustments.sparkleOffset + nnOffsets.sparkleOffset,
      hueOffset: wrapHue(nnOffsets.hueOffset + manualAdjustments.hueOffset),
      nnSpawnOffset: nnOffsets.spawnOffset,
      nnGlowOffset: nnOffsets.glowOffset,
      nnSparkleOffset: nnOffsets.sparkleOffset,
      nnHueOffset: wrapHue(nnOffsets.hueOffset),
      manualSpawnOffset: manualAdjustments.spawnOffset,
      manualGlowOffset: manualAdjustments.glowOffset,
      manualSparkleOffset: manualAdjustments.sparkleOffset,
      manualHueOffset: manualAdjustments.hueOffset,
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
