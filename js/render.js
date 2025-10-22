const TAU = Math.PI * 2;
const MAX_PIXEL_RATIO = 3;
const MIN_DYNAMIC_SCALE = 0.55;
const MAX_DYNAMIC_SCALE = 1;
const SCALE_STEP = 0.1;
const TARGET_FPS = 58;
const TARGET_FRAME_MS = 1000 / TARGET_FPS;
const DROP_THRESHOLD_MS = 20.5;
const RECOVER_THRESHOLD_MS = 17;
const RESIZE_DEBOUNCE_MS = 150;
const TRAIL_BASE_ALPHA = 0.12;
const DEFAULT_BASE_HUE = 218;
const DEFAULT_PALETTE = Object.freeze({
  background: '#050505',
  accents: ['#a78bfa', '#c4b5fd', '#ede9fe'],
  baseHue: DEFAULT_BASE_HUE,
});
const DEFAULT_GRID_COLOR = 'rgba(170, 180, 220, 0.08)';
const VOLUME_MIN = 0;
const VOLUME_MAX = 1;
const PARAM_SCRATCH = {
  trailFade: 0.65,
  glow: 0.5,
  sizeJitter: 0.25,
  hueShift: 0,
  sparkleDensity: 0.05,
  zoom: 1,
};

const CONNECTION_FRACTION = 0.9;

const TOGGLE_DEFAULTS = /** @type {const} */ ({
  fullscreen: false,
  bloom: true,
  trails: true,
  grid: false,
  safe: false,
  bypass: false,
});

const FULLSCREEN_CLASS = 'fullscreen-active';

const KEY_PLAYLIST_MAP = {
  Digit1: 0,
  Digit2: 1,
  Digit3: 2,
  Digit4: 3,
  Digit5: 4,
  Digit6: 5,
  Digit7: 6,
  Digit8: 7,
  Digit9: 8,
  Digit0: 9,
  Minus: 10,
};

function normalizeHexColor(color) {
  if (typeof color !== 'string') {
    return null;
  }
  const trimmed = color.trim();
  if (trimmed.length === 0) {
    return null;
  }
  const withoutHash = trimmed.startsWith('#') ? trimmed.slice(1) : trimmed;
  if (/^[0-9a-fA-F]{6}$/.test(withoutHash)) {
    return `#${withoutHash.toLowerCase()}`;
  }
  if (/^[0-9a-fA-F]{3}$/.test(withoutHash)) {
    const expanded = withoutHash
      .toLowerCase()
      .split('')
      .map((ch) => ch + ch)
      .join('');
    return `#${expanded}`;
  }
  return null;
}

function hexToRgb(hex) {
  const normalized = normalizeHexColor(hex);
  if (!normalized) {
    return { r: 0, g: 0, b: 0 };
  }
  const value = parseInt(normalized.slice(1), 16);
  return {
    r: (value >> 16) & 0xff,
    g: (value >> 8) & 0xff,
    b: value & 0xff,
  };
}

function rgbToHex(rgb) {
  const r = clamp(Math.round(rgb.r ?? 0), 0, 255);
  const g = clamp(Math.round(rgb.g ?? 0), 0, 255);
  const b = clamp(Math.round(rgb.b ?? 0), 0, 255);
  return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`;
}

function rgbToHsl(rgb) {
  const r = (rgb.r ?? 0) / 255;
  const g = (rgb.g ?? 0) / 255;
  const b = (rgb.b ?? 0) / 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;
  let h = 0;
  if (delta !== 0) {
    if (max === r) {
      h = ((g - b) / delta) % 6;
    } else if (max === g) {
      h = (b - r) / delta + 2;
    } else {
      h = (r - g) / delta + 4;
    }
  }
  h = (h * 60 + 360) % 360;
  const l = (max + min) * 0.5;
  const s = delta === 0 ? 0 : delta / (1 - Math.abs(2 * l - 1));
  return {
    h,
    s: s * 100,
    l: l * 100,
  };
}

function mixRgb(a, b, amount) {
  const t = clamp(Number.isFinite(amount) ? amount : 0, 0, 1);
  return {
    r: Math.round((a.r ?? 0) + ((b.r ?? 0) - (a.r ?? 0)) * t),
    g: Math.round((a.g ?? 0) + ((b.g ?? 0) - (a.g ?? 0)) * t),
    b: Math.round((a.b ?? 0) + ((b.b ?? 0) - (a.b ?? 0)) * t),
  };
}

function wrapHue360(value) {
  if (!Number.isFinite(value)) {
    return DEFAULT_BASE_HUE;
  }
  let result = value % 360;
  if (result < 0) {
    result += 360;
  }
  return result;
}

function buildPalette(paletteInput = {}) {
  const backgroundHex = normalizeHexColor(paletteInput.background) ?? DEFAULT_PALETTE.background;
  const backgroundRgb = hexToRgb(backgroundHex);

  const accentSource = Array.isArray(paletteInput.accents) && paletteInput.accents.length > 0
    ? paletteInput.accents
    : DEFAULT_PALETTE.accents;
  const accentHexes = accentSource
    .map((value) => normalizeHexColor(value))
    .filter((value) => value !== null);
  if (accentHexes.length === 0) {
    accentHexes.push(DEFAULT_PALETTE.accents[0]);
  }
  const accentRgb = accentHexes.map((hex) => hexToRgb(hex));
  const accentHsl = accentRgb.map((rgb) => rgbToHsl(rgb));

  const baseHueInput = Number.isFinite(paletteInput.baseHue) ? paletteInput.baseHue : DEFAULT_PALETTE.baseHue;
  const baseHue = wrapHue360(baseHueInput);

  const gradientAccent = accentRgb[0] ?? accentRgb[1] ?? backgroundRgb;
  const gradientInner = mixRgb(backgroundRgb, gradientAccent, 0.35);
  const canvasBackground = `radial-gradient(circle at 20% 20%, ${rgbToHex(gradientInner)}, ${backgroundHex} 68%)`;

  const gridBase = accentRgb[1] ?? accentRgb[0] ?? hexToRgb('#aab4dc');
  const gridMix = mixRgb(gridBase, backgroundRgb, 0.55);
  const gridColor = `rgba(${gridMix.r}, ${gridMix.g}, ${gridMix.b}, 0.12)`;

  const panelColor = `rgba(${backgroundRgb.r}, ${backgroundRgb.g}, ${backgroundRgb.b}, 0.82)`;
  const accentPrimary = accentHexes[0];

  return {
    backgroundHex,
    backgroundRgb,
    accentHexes,
    accentHsl,
    baseHue,
    canvasBackground,
    gridColor,
    panelColor,
    accentPrimary,
  };
}

function applyPaletteToDom() {
  const root = document.documentElement;
  if (root) {
    root.style.setProperty('--bg', state.palette.backgroundHex);
    root.style.setProperty('--panel', state.palette.panelColor);
    const accent = state.palette.accentPrimary ?? DEFAULT_PALETTE.accents[0];
    root.style.setProperty('--accent', accent);
  }
  if (state.canvas) {
    state.canvas.style.background = state.palette.canvasBackground;
  }
}

const listeners = new Map();

const paletteState = buildPalette(DEFAULT_PALETTE);

const state = {
  initialized: false,
  canvas: /** @type {HTMLCanvasElement|null} */ (null),
  ctx: /** @type {CanvasRenderingContext2D|null} */ (null),
  hud: {
    root: /** @type {HTMLElement|null} */ (null),
    title: /** @type {HTMLElement|null} */ (null),
    time: /** @type {HTMLElement|null} */ (null),
    status: /** @type {HTMLElement|null} */ (null),
    fps: /** @type {HTMLElement|null} */ (null),
    volumeSlider: /** @type {HTMLInputElement|null} */ (null),
    volumeDisplay: /** @type {HTMLElement|null} */ (null),
    toggleInputs: new Map(),
  },
  toggles: { ...TOGGLE_DEFAULTS },
  pixelRatio: 1,
  dynamicScale: 1,
  logicalWidth: 0,
  logicalHeight: 0,
  renderScale: 1,
  lastTime: 0,
  frameCounter: 0,
  fpsSamples: new Float32Array(120),
  fpsIndex: 0,
  fpsSum: 0,
  fpsSampleCount: 0,
  fps: 0,
  statusText: 'Idle',
  trackTitle: 'Latent Noise',
  trackTime: '00:00',
  volume: 0.7,
  performance: {
    overloadFrames: 0,
    recoveryFrames: 0,
    rollingFrameTime: 1000 / 60,
  },
  world: {
    width: 2,
    height: 2,
  },
  glow: {
    enabled: true,
    scale: 0.5,
    strength: 0.65,
    canvas: /** @type {HTMLCanvasElement|null} */ (null),
    ctx: /** @type {CanvasRenderingContext2D|null} */ (null),
  },
  keyHandlersBound: false,
  resizeHandlerBound: false,
  resizeTimerId: 0,
  fullscreenChangeBound: false,
  frameSeed: 0,
  palette: paletteState,
};

applyPaletteToDom();

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

function fract(value) {
  return value - Math.floor(value);
}

function hash(index, frameSeed = 0) {
  const x = Math.sin(index * 12.9898 + frameSeed * 78.233);
  return fract(x * 43758.5453);
}

function emit(eventName, detail) {
  const handlers = listeners.get(eventName);
  if (handlers) {
    for (const handler of handlers) {
      try {
        handler(detail);
      } catch (error) {
        console.error('[render] listener error', error);
      }
    }
  }
  const wildcard = listeners.get('*');
  if (wildcard) {
    for (const handler of wildcard) {
      try {
        handler({ event: eventName, detail });
      } catch (error) {
        console.error('[render] listener error', error);
      }
    }
  }
}

function registerListener(eventName, handler) {
  if (!listeners.has(eventName)) {
    listeners.set(eventName, new Set());
  }
  listeners.get(eventName).add(handler);
}

function unregisterListener(eventName, handler) {
  const handlers = listeners.get(eventName);
  if (!handlers) {
    return;
  }
  handlers.delete(handler);
  if (handlers.size === 0) {
    listeners.delete(eventName);
  }
}

function formatClock(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return '00:00';
  }
  const total = Math.floor(seconds);
  const m = Math.floor(total / 60);
  const s = total % 60;
  const mm = m < 10 ? `0${m}` : `${m}`;
  const ss = s < 10 ? `0${s}` : `${s}`;
  return `${mm}:${ss}`;
}

function formatTrackTime(current, duration) {
  const cur = formatClock(current);
  if (!Number.isFinite(duration) || duration <= 0) {
    return `${cur}`;
  }
  const dur = formatClock(duration);
  return `${cur} / ${dur}`;
}

function ensureGlowCanvas() {
  if (!state.canvas || !state.ctx) {
    return;
  }
  if (!state.glow.canvas) {
    state.glow.canvas = document.createElement('canvas');
    state.glow.ctx = state.glow.canvas.getContext('2d', { alpha: true });
  }
  const glowCanvas = state.glow.canvas;
  const glowCtx = state.glow.ctx;
  if (!glowCanvas || !glowCtx) {
    state.glow.enabled = false;
    return;
  }

  const pixelWidth = Math.max(1, Math.round(state.canvas.width * state.glow.scale));
  const pixelHeight = Math.max(1, Math.round(state.canvas.height * state.glow.scale));
  if (glowCanvas.width !== pixelWidth || glowCanvas.height !== pixelHeight) {
    glowCanvas.width = pixelWidth;
    glowCanvas.height = pixelHeight;
  }

  const scale = state.pixelRatio * state.dynamicScale * state.glow.scale;
  glowCtx.setTransform(scale, 0, 0, scale, 0, 0);
  glowCtx.globalCompositeOperation = 'source-over';
}

function ensureCanvasSize(force = false) {
  if (!state.canvas || !state.ctx) {
    return;
  }
  const cssWidth = state.canvas.clientWidth || window.innerWidth || 1;
  const cssHeight = state.canvas.clientHeight || window.innerHeight || 1;
  const pixelRatio = clamp(window.devicePixelRatio || 1, 1, MAX_PIXEL_RATIO);
  state.pixelRatio = pixelRatio;

  const scale = state.dynamicScale;
  const desiredWidth = Math.max(1, Math.round(cssWidth * pixelRatio * scale));
  const desiredHeight = Math.max(1, Math.round(cssHeight * pixelRatio * scale));

  if (force || state.canvas.width !== desiredWidth || state.canvas.height !== desiredHeight) {
    state.canvas.width = desiredWidth;
    state.canvas.height = desiredHeight;
    state.ctx.setTransform(pixelRatio * scale, 0, 0, pixelRatio * scale, 0, 0);
    state.ctx.globalCompositeOperation = 'source-over';
    ensureGlowCanvas();
  }

  state.logicalWidth = cssWidth;
  state.logicalHeight = cssHeight;
  state.renderScale = Math.min(cssWidth, cssHeight) * 0.5;
}

function updateVolumeDisplay(value) {
  state.volume = clamp(value, VOLUME_MIN, VOLUME_MAX);
  if (state.hud.volumeDisplay) {
    const percent = Math.round(state.volume * 100);
    state.hud.volumeDisplay.textContent = `${percent}%`;
  }
}

function getControlsElement() {
  if (typeof document === 'undefined') {
    return null;
  }
  const controls = document.getElementById('controls');
  return controls instanceof HTMLElement ? controls : null;
}

function getDebugOverlayElement() {
  if (typeof document === 'undefined') {
    return null;
  }
  const overlay = document.getElementById('debug-overlay');
  return overlay instanceof HTMLElement ? overlay : null;
}

function applyFullscreenState() {
  const active = Boolean(state.toggles.fullscreen);
  if (typeof document !== 'undefined' && document.body) {
    document.body.classList.toggle(FULLSCREEN_CLASS, active);
  }
  if (state.hud.root) {
    state.hud.root.setAttribute('aria-hidden', active ? 'true' : 'false');
  }
  const controls = getControlsElement();
  if (controls) {
    controls.setAttribute('aria-hidden', active ? 'true' : 'false');
  }
  const overlay = getDebugOverlayElement();
  if (overlay) {
    overlay.setAttribute('aria-hidden', active ? 'true' : 'false');
  }
}

function applySafeModeClass() {
  if (typeof document === 'undefined' || !document.body) {
    return;
  }
  document.body.classList.toggle('safe-mode', state.toggles.safe);
}

function requestFullscreen() {
  if (typeof document === 'undefined') {
    return Promise.reject(new Error('Fullscreen API unavailable'));
  }
  const target = document.documentElement;
  if (target && typeof target.requestFullscreen === 'function') {
    return target.requestFullscreen();
  }
  return Promise.reject(new Error('Fullscreen API not supported'));
}

function exitFullscreen() {
  if (typeof document === 'undefined' || typeof document.exitFullscreen !== 'function') {
    return Promise.resolve();
  }
  if (!document.fullscreenElement) {
    return Promise.resolve();
  }
  return document.exitFullscreen();
}

function handleFullscreenChange() {
  if (typeof document === 'undefined') {
    return;
  }
  const active = Boolean(document.fullscreenElement);
  if (state.toggles.fullscreen !== active) {
    state.toggles.fullscreen = active;
    const input = state.hud.toggleInputs.get('fullscreen');
    if (input) {
      input.checked = active;
    }
    applyFullscreenState();
    emit('toggle', { name: 'fullscreen', value: active, source: 'system' });
    return;
  }
  applyFullscreenState();
}

function bindFullscreenChange() {
  if (state.fullscreenChangeBound || typeof document === 'undefined') {
    return;
  }
  document.addEventListener('fullscreenchange', handleFullscreenChange);
  state.fullscreenChangeBound = true;
}

function unbindFullscreenChange() {
  if (!state.fullscreenChangeBound || typeof document === 'undefined') {
    return;
  }
  document.removeEventListener('fullscreenchange', handleFullscreenChange);
  state.fullscreenChangeBound = false;
}

function handleToggleChange(name, value, source) {
  if (!(name in state.toggles)) {
    return;
  }
  const nextValue = Boolean(value);
  if (state.toggles[name] === nextValue && source === 'api') {
    return;
  }
  state.toggles[name] = nextValue;

  const input = state.hud.toggleInputs.get(name);
  if (input && input.checked !== nextValue) {
    input.checked = nextValue;
  }

  if (name === 'fullscreen') {
    applyFullscreenState();
    if (source !== 'system') {
      if (nextValue) {
        requestFullscreen().catch((error) => {
          console.warn('[render] Failed to enter fullscreen', error);
          handleToggleChange('fullscreen', false, 'system');
        });
      } else {
        exitFullscreen().catch((error) => {
          console.warn('[render] Failed to exit fullscreen', error);
        });
      }
    }
  } else if (name === 'safe') {
    applySafeModeClass();
    emit('safeModeChange', nextValue);
  } else if (name === 'bypass') {
    emit('nnBypassChange', nextValue);
  }

  emit('toggle', { name, value: nextValue, source });
}

function handleToggleInput(event) {
  const target = event.target;
  if (!(target instanceof HTMLInputElement)) {
    return;
  }
  const name = target.dataset.toggle;
  if (!name) {
    return;
  }
  handleToggleChange(name, target.checked, 'ui');
}

function updateFps(frameTimeMs, fpsInstantHint, fpsAverageHint) {
  let instantaneous = Number.isFinite(fpsInstantHint) && fpsInstantHint > 0 ? fpsInstantHint : 0;
  if (!Number.isFinite(instantaneous) || instantaneous <= 0) {
    instantaneous = frameTimeMs > 0 ? 1000 / frameTimeMs : 0;
  }

  const samples = state.fpsSamples;
  const index = state.fpsIndex;

  if (state.fpsSampleCount === samples.length) {
    state.fpsSum -= samples[index];
  }

  samples[index] = instantaneous;
  state.fpsSum += instantaneous;

  if (state.fpsSampleCount < samples.length) {
    state.fpsSampleCount += 1;
  }

  state.fpsIndex = (index + 1) % samples.length;

  const windowAverage = state.fpsSampleCount > 0 ? state.fpsSum / state.fpsSampleCount : instantaneous;
  const average = Number.isFinite(fpsAverageHint) && fpsAverageHint > 0 ? fpsAverageHint : windowAverage;

  state.fps = average;
  state.performance.rollingFrameTime = average > 0 ? 1000 / average : frameTimeMs > 0 ? frameTimeMs : Infinity;

  if (state.hud.fps) {
    const fpsValue = Number.isFinite(state.fps) && state.fps > 0 ? Math.round(state.fps) : 0;
    state.hud.fps.textContent = fpsValue > 0 ? `FPS: ${fpsValue}` : 'FPS: --';
  }
}

function adjustDynamicScale(frameTimeMs, rollingFrameTimeMs) {
  const hasInstant = Number.isFinite(frameTimeMs);
  const hasRolling = Number.isFinite(rollingFrameTimeMs);
  if (!hasInstant && !hasRolling) {
    return;
  }

  const overloadCheck = hasInstant && hasRolling
    ? Math.max(frameTimeMs, rollingFrameTimeMs)
    : hasInstant
      ? frameTimeMs
      : rollingFrameTimeMs;
  const recoveryCheck = hasInstant && hasRolling
    ? Math.min(frameTimeMs, rollingFrameTimeMs)
    : hasRolling
      ? rollingFrameTimeMs
      : frameTimeMs;

  const overloadByAverage = hasRolling && rollingFrameTimeMs > TARGET_FRAME_MS;
  const overloadBySpike = Number.isFinite(overloadCheck) && overloadCheck > DROP_THRESHOLD_MS;

  if (overloadByAverage || overloadBySpike) {
    state.performance.overloadFrames += 1;
    state.performance.recoveryFrames = 0;
    if (state.performance.overloadFrames > 8 && state.dynamicScale > MIN_DYNAMIC_SCALE) {
      state.dynamicScale = Math.max(MIN_DYNAMIC_SCALE, state.dynamicScale - SCALE_STEP);
      state.performance.overloadFrames = 0;
      ensureCanvasSize(true);
      emit('resolutionChange', { scale: state.dynamicScale });
      console.info('[render] dynamicScale drop', state.dynamicScale);
    }
  } else if (
    Number.isFinite(recoveryCheck)
    && recoveryCheck < RECOVER_THRESHOLD_MS
    && (!hasRolling || rollingFrameTimeMs < TARGET_FRAME_MS)
    && state.dynamicScale < MAX_DYNAMIC_SCALE
  ) {
    state.performance.recoveryFrames += 1;
    if (state.performance.recoveryFrames > 90) {
      state.dynamicScale = Math.min(MAX_DYNAMIC_SCALE, state.dynamicScale + SCALE_STEP);
      state.performance.recoveryFrames = 0;
      ensureCanvasSize(true);
      emit('resolutionChange', { scale: state.dynamicScale });
      console.info('[render] dynamicScale recover', state.dynamicScale);
    }
  } else {
    state.performance.overloadFrames = Math.max(0, state.performance.overloadFrames - 1);
  }
}

function fadeCanvas(alpha) {
  if (!state.ctx || !state.canvas) {
    return;
  }
  state.ctx.save();
  state.ctx.setTransform(1, 0, 0, 1, 0, 0);
  state.ctx.globalCompositeOperation = 'source-over';
  const bg = state.palette.backgroundRgb;
  state.ctx.fillStyle = `rgba(${bg.r}, ${bg.g}, ${bg.b}, ${alpha})`;
  state.ctx.fillRect(0, 0, state.canvas.width, state.canvas.height);
  state.ctx.restore();
  state.ctx.setTransform(state.pixelRatio * state.dynamicScale, 0, 0, state.pixelRatio * state.dynamicScale, 0, 0);
}

function fadeGlow(alpha) {
  if (!state.glow.ctx || !state.glow.canvas) {
    return;
  }
  const glowCtx = state.glow.ctx;
  glowCtx.save();
  glowCtx.setTransform(1, 0, 0, 1, 0, 0);
  glowCtx.globalCompositeOperation = 'source-over';
  glowCtx.fillStyle = `rgba(0, 0, 0, ${alpha})`;
  glowCtx.fillRect(0, 0, state.glow.canvas.width, state.glow.canvas.height);
  glowCtx.restore();
  const scale = state.pixelRatio * state.dynamicScale * state.glow.scale;
  glowCtx.setTransform(scale, 0, 0, scale, 0, 0);
  glowCtx.globalCompositeOperation = 'lighter';
}

function drawGrid() {
  if (!state.ctx) {
    return;
  }
  const ctx = state.ctx;
  const width = state.logicalWidth;
  const height = state.logicalHeight;
  const spacing = Math.max(60, Math.min(width, height) / 10);

  ctx.save();
  ctx.globalCompositeOperation = 'screen';
  ctx.strokeStyle = state.palette.gridColor ?? DEFAULT_GRID_COLOR;
  ctx.lineWidth = 1;

  for (let x = spacing * 0.5; x < width; x += spacing) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }

  for (let y = spacing * 0.5; y < height; y += spacing) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  ctx.restore();
}

function compositeGlow() {
  if (!state.ctx || !state.glow.canvas || !state.glow.ctx) {
    return;
  }
  const ctx = state.ctx;
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  ctx.globalAlpha = clamp(state.glow.strength, 0, 1);
  ctx.filter = 'blur(18px)';
  ctx.drawImage(state.glow.canvas, 0, 0, state.logicalWidth, state.logicalHeight);
  ctx.filter = 'none';
  ctx.restore();
}

function updateHudText() {
  if (state.hud.title) {
    state.hud.title.textContent = state.trackTitle;
  }
  if (state.hud.time) {
    state.hud.time.textContent = state.trackTime;
  }
  if (state.hud.status) {
    state.hud.status.textContent = state.statusText;
  }
}

function handleKeyDown(event) {
  if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) {
    return;
  }
  const code = event.code;
  switch (code) {
    case 'Space':
      event.preventDefault();
      emit('playToggle');
      break;
    case 'KeyN':
      emit('nextTrack');
      break;
    case 'KeyP':
      if (event.shiftKey) {
        handleToggleChange('safe', !state.toggles.safe, 'keyboard');
      } else {
        emit('prevTrack');
      }
      break;
    case 'ArrowRight':
      emit('seekForward', { seconds: event.shiftKey ? 10 : 5 });
      break;
    case 'ArrowLeft':
      emit('seekBackward', { seconds: event.shiftKey ? 10 : 5 });
      break;
    case 'KeyF':
      handleToggleChange('fullscreen', !state.toggles.fullscreen, 'keyboard');
      break;
    case 'KeyB':
      handleToggleChange('bloom', !state.toggles.bloom, 'keyboard');
      break;
    case 'KeyT':
      handleToggleChange('trails', !state.toggles.trails, 'keyboard');
      break;
    case 'KeyG':
      handleToggleChange('grid', !state.toggles.grid, 'keyboard');
      break;
    case 'KeyK':
      handleToggleChange('bypass', !state.toggles.bypass, 'keyboard');
      break;
    case 'BracketLeft':
      emit('adjustParticles', { delta: event.shiftKey ? -0.2 : -0.08 });
      break;
    case 'BracketRight':
      emit('adjustParticles', { delta: event.shiftKey ? 0.2 : 0.08 });
      break;
    case 'Semicolon':
      emit('adjustIntensity', { delta: event.shiftKey ? -0.15 : -0.06 });
      break;
    case 'Quote':
      emit('adjustIntensity', { delta: event.shiftKey ? 0.15 : 0.06 });
      break;
    case 'Comma':
      emit('cyclePalette', { direction: -1 });
      break;
    case 'Period':
      emit('cyclePalette', { direction: 1 });
      break;
    default:
      if (code in KEY_PLAYLIST_MAP) {
        emit('selectTrack', { index: KEY_PLAYLIST_MAP[code] });
      }
      break;
  }
}

function bindKeyboard() {
  if (state.keyHandlersBound) {
    return;
  }
  window.addEventListener('keydown', handleKeyDown);
  state.keyHandlersBound = true;
}

function unbindKeyboard() {
  if (!state.keyHandlersBound) {
    return;
  }
  window.removeEventListener('keydown', handleKeyDown);
  state.keyHandlersBound = false;
}

function flushResize() {
  if (!state.initialized) {
    state.resizeTimerId = 0;
    return;
  }
  state.resizeTimerId = 0;
  ensureCanvasSize(true);
}

function handleResize() {
  if (!state.initialized) {
    return;
  }
  if (state.resizeTimerId) {
    window.clearTimeout(state.resizeTimerId);
  }
  state.resizeTimerId = window.setTimeout(flushResize, RESIZE_DEBOUNCE_MS);
}

function bindResize() {
  if (state.resizeHandlerBound) {
    return;
  }
  window.addEventListener('resize', handleResize);
  state.resizeHandlerBound = true;
}

function unbindResize() {
  if (!state.resizeHandlerBound) {
    return;
  }
  window.removeEventListener('resize', handleResize);
  state.resizeHandlerBound = false;
  if (state.resizeTimerId) {
    window.clearTimeout(state.resizeTimerId);
    state.resizeTimerId = 0;
  }
}

function assertElement(element, message) {
  if (!element) {
    throw new Error(message);
  }
  return element;
}

export function init(options = {}) {
  if (state.initialized) {
    return;
  }
  const canvas = assertElement(options.canvas || document.getElementById('c'), 'Canvas element #c is required.');
  const ctx = canvas.getContext('2d', { alpha: true, desynchronized: true });
  if (!ctx) {
    throw new Error('Failed to acquire 2D rendering context.');
  }

  state.canvas = canvas;
  state.ctx = ctx;
  applyPaletteToDom();

  const hudRoot = assertElement(options.hud || document.getElementById('hud'), 'HUD element #hud is required.');
  const title = assertElement(options.trackTitle || document.getElementById('track-title'), 'HUD track title element missing.');
  const time = assertElement(options.trackTime || document.getElementById('track-time'), 'HUD track time element missing.');
  const status = assertElement(options.status || document.getElementById('hud-status'), 'HUD status element missing.');
  const fps = assertElement(options.fps || document.getElementById('hud-fps'), 'HUD FPS element missing.');
  const volumeSlider = assertElement(options.volumeSlider || document.getElementById('volume'), 'Volume slider #volume missing.');
  const volumeDisplay = assertElement(options.volumeDisplay || document.getElementById('volume-display'), 'HUD volume display missing.');
  const toggleContainer = assertElement(options.toggleContainer || document.getElementById('hud-toggles'), 'HUD toggles container missing.');

  state.hud.root = hudRoot;
  state.hud.title = title;
  state.hud.time = time;
  state.hud.status = status;
  state.hud.fps = fps;
  state.hud.volumeSlider = volumeSlider;
  state.hud.volumeDisplay = volumeDisplay;

  state.hud.toggleInputs.clear();
  const inputs = toggleContainer.querySelectorAll('input[data-toggle]');
  inputs.forEach((input) => {
    if (input instanceof HTMLInputElement) {
      const name = input.dataset.toggle;
      if (name) {
        state.hud.toggleInputs.set(name, input);
        const defaultValue = name in TOGGLE_DEFAULTS ? TOGGLE_DEFAULTS[name] : input.checked;
        input.checked = defaultValue;
        state.toggles[name] = defaultValue;
      }
      input.addEventListener('change', handleToggleInput);
    }
  });

  toggleContainer.addEventListener('change', handleToggleInput);

  volumeSlider.addEventListener('input', () => {
    updateVolumeDisplay(Number(volumeSlider.value));
  });
  updateVolumeDisplay(Number(volumeSlider.value));

  ensureCanvasSize(true);
  bindKeyboard();
  bindResize();
  bindFullscreenChange();
  applyFullscreenState();
  applySafeModeClass();
  updateHudText();

  state.initialized = true;
  emit('ready');
}

export function destroy() {
  if (!state.initialized) {
    return;
  }
  unbindKeyboard();
  unbindResize();
  unbindFullscreenChange();
  if (state.resizeTimerId) {
    window.clearTimeout(state.resizeTimerId);
    state.resizeTimerId = 0;
  }

  if (state.hud.toggleInputs.size > 0) {
    for (const input of state.hud.toggleInputs.values()) {
      input.removeEventListener('change', handleToggleInput);
    }
  }
  const toggleContainer = state.hud.root ? state.hud.root.querySelector('#hud-toggles') : null;
  if (toggleContainer) {
    toggleContainer.removeEventListener('change', handleToggleInput);
  }

  state.hud.toggleInputs.clear();
  state.initialized = false;
  state.canvas = null;
  state.ctx = null;
  state.glow.canvas = null;
  state.glow.ctx = null;
  state.hud.root = null;
  state.hud.title = null;
  state.hud.time = null;
  state.hud.status = null;
  state.hud.fps = null;
  state.hud.volumeSlider = null;
  state.hud.volumeDisplay = null;
}

export function getPalette() {
  return {
    background: state.palette.backgroundHex,
    accents: [...state.palette.accentHexes],
    baseHue: state.palette.baseHue,
  };
}

export function setPalette(palette) {
  state.palette = buildPalette(palette ?? {});
  applyPaletteToDom();
  return getPalette();
}

export function setWorldSize(width, height) {
  if (!Number.isFinite(width) || !Number.isFinite(height)) {
    return;
  }
  state.world.width = Math.max(width, 0.1);
  state.world.height = Math.max(height, 0.1);
}

export function setTrackTitle(title) {
  state.trackTitle = title || 'Latent Noise';
  updateHudText();
}

export function updateTrackTime(currentSeconds, durationSeconds) {
  state.trackTime = formatTrackTime(currentSeconds, durationSeconds);
  updateHudText();
}

export function setStatus(text) {
  state.statusText = text || 'Idle';
  updateHudText();
}

export function updateVolume(value) {
  if (!state.hud.volumeSlider) {
    return;
  }
  const clamped = clamp(value, VOLUME_MIN, VOLUME_MAX);
  state.hud.volumeSlider.value = clamped.toFixed(2);
  updateVolumeDisplay(clamped);
}

export function getToggles() {
  return { ...state.toggles };
}

export function setToggle(name, value) {
  handleToggleChange(name, Boolean(value), 'api');
}

export function on(eventName, handler) {
  if (typeof handler !== 'function') {
    throw new TypeError('Listener must be a function.');
  }
  registerListener(eventName, handler);
  return () => off(eventName, handler);
}

export function off(eventName, handler) {
  unregisterListener(eventName, handler);
}

function resolveParams(input = {}) {
  PARAM_SCRATCH.trailFade = clamp(input.trailFade ?? 0.65, 0, 0.98);
  PARAM_SCRATCH.glow = clamp(input.glow ?? 0.5, 0, 1);
  PARAM_SCRATCH.sizeJitter = clamp(input.sizeJitter ?? 0.25, 0, 0.8);
  PARAM_SCRATCH.hueShift = Number.isFinite(input.hueShift) ? input.hueShift : 0;
  PARAM_SCRATCH.sparkleDensity = clamp(input.sparkleDensity ?? 0.05, 0, 1);
  const zoom = Number.isFinite(input.zoom) ? input.zoom : 1;
  PARAM_SCRATCH.zoom = clamp(zoom, 0.5, 20);
  return PARAM_SCRATCH;
}

function prepareGlow(glowLevel) {
  state.glow.enabled = state.toggles.bloom && glowLevel > 0.01;
  if (state.glow.enabled) {
    state.glow.strength = clamp(glowLevel * 0.85, 0.1, 0.9);
    ensureGlowCanvas();
    fadeGlow(1 - glowLevel * 0.6);
  }
}

function drawParticles(particles, params, dt) {
  if (!particles || !state.ctx) {
    return;
  }
  const positionsX = particles.positions?.x;
  const positionsY = particles.positions?.y;
  const life = particles.life;
  const maxLife = particles.maxLife;
  const masses = particles.masses;
  const alive = particles.alive;
  const indices = particles.indices;
  const count = Math.min(particles.count ?? 0, indices ? indices.length : 0);

  if (!positionsX || !positionsY || !life || !maxLife || !indices || count === 0) {
    return;
  }

  const ctx = state.ctx;
  const glowCtx = state.glow.enabled ? state.glow.ctx : null;
  if (glowCtx) {
    glowCtx.globalCompositeOperation = 'lighter';
  }

  const centerX = state.logicalWidth * 0.5;
  const centerY = state.logicalHeight * 0.5;
  const zoom = Number.isFinite(params.zoom) ? params.zoom : 1;
  const zoomFactor = clamp(zoom, 0.5, 20);
  const worldWidth = Math.max(state.world.width, 1e-3);
  const worldHeight = Math.max(state.world.height, 1e-3);
  const scaleX = (state.logicalWidth / worldWidth) * zoomFactor;
  const scaleY = (state.logicalHeight / worldHeight) * zoomFactor;

  const jitter = params.sizeJitter;
  const sparkle = params.sparkleDensity;
  const palette = state.palette;
  const accentCount = palette.accentHsl.length;
  const hueBase = wrapHue360((palette.baseHue ?? DEFAULT_BASE_HUE) + params.hueShift);

  const maxConnections =
    CONNECTION_FRACTION > 0 ? Math.min(Math.floor(count * CONNECTION_FRACTION * 0.5), count) : 0;
  const connectionStride = maxConnections > 0 ? Math.max(1, Math.round(count / maxConnections)) : count + 1;
  const pixelScale = Math.max(0.001, state.pixelRatio * state.dynamicScale);
  const connectionLineWidth = clamp(0.7 / pixelScale, 0.25, 1.15);
  let connectionsDrawn = 0;

  ctx.save();
  ctx.globalCompositeOperation = 'lighter';

  const sparkleThreshold = 1 - sparkle * 0.6;
  const fadeBoost = dt ? Math.exp(-dt * 1.6) : 1;
  const connectionAlphaBase = clamp(0.08 + (1 - params.trailFade) * 0.22, 0.06, 0.35) * fadeBoost;

  for (let i = 0; i < count; i += 1) {
    const index = indices[i];
    if (alive && alive[index] === 0) {
      continue;
    }

    const px = positionsX[index];
    const py = positionsY[index];

    const sx = centerX + px * scaleX;
    const sy = centerY + py * scaleY;

    const max = maxLife[index] || 1;
    const current = life[index] || 0;
    const lifeT = clamp(current / max, 0, 1);
    const baseMass = masses ? masses[index] : 1;
    const rng = hash(index, state.frameSeed);
    const sizeJitter = 1 + (rng - 0.5) * 2 * jitter;
    const radius = (1.6 + baseMass * 1.1) * sizeJitter * (0.6 + (1 - lifeT) * 0.6);

    const accent = accentCount > 0 ? palette.accentHsl[Math.floor(rng * accentCount)] : null;
    let hue = wrapHue360(hueBase + rng * 36 + lifeT * 72);
    let saturation = 78;
    let bodyLight = 48 + (1 - lifeT) * 32;
    let glowLight = 60;
    let sparkleLight = clamp(bodyLight + 16, 0, 100);
    if (accent) {
      hue = wrapHue360(accent.h + params.hueShift + (rng - 0.5) * 24 + lifeT * 14);
      saturation = clamp(accent.s * (0.85 + (1 - lifeT) * 0.18), 20, 100);
      const accentLight = clamp(accent.l, 10, 88);
      bodyLight = clamp(accentLight * 0.7 + (1 - lifeT) * 28, 8, 94);
      glowLight = clamp(accentLight + 24, 14, 98);
      sparkleLight = clamp(bodyLight + 14, 0, 100);
    }
    const alpha = clamp(0.25 + (1 - lifeT) * 0.55, 0.1, 0.85) * fadeBoost;

    if (
      maxConnections > 0 &&
      connectionsDrawn < maxConnections &&
      (i % connectionStride === 0 || connectionsDrawn < maxConnections * 0.4)
    ) {
      const partnerSeed = fract(rng * 97.417 + i * 0.611 + state.frameSeed * 0.733);
      let partnerSlot = Math.floor(partnerSeed * count);
      if (partnerSlot === i) {
        partnerSlot = (partnerSlot + 1) % count;
      }
      const partnerIndex = indices[partnerSlot];
      if (partnerIndex !== index && (!alive || alive[partnerIndex] !== 0)) {
        const connectionAlpha = clamp(connectionAlphaBase * (1.15 - lifeT * 0.6), 0.025, 0.38);
        const connectionHue = wrapHue360(hue + (partnerSeed - 0.5) * 18);
        const connectionSat = Math.round(clamp(saturation * 0.6 + 18, 10, 95));
        const connectionLight = Math.round(clamp(bodyLight * 0.82 + 12, 12, 92));
        ctx.lineWidth = connectionLineWidth;
        ctx.strokeStyle = `hsl(${Math.round(connectionHue)}, ${connectionSat}%, ${connectionLight}%)`;
        ctx.globalAlpha = connectionAlpha;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(centerX + positionsX[partnerIndex] * scaleX, centerY + positionsY[partnerIndex] * scaleY);
        ctx.stroke();
        connectionsDrawn += 1;
      }
    }

    ctx.fillStyle = `hsl(${Math.round(hue)}, ${Math.round(saturation)}%, ${Math.round(bodyLight)}%)`;
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    ctx.arc(sx, sy, radius, 0, TAU);
    ctx.fill();

    if (glowCtx) {
      glowCtx.fillStyle = `hsl(${Math.round(hue)}, ${Math.round(Math.min(100, saturation + 8))}%, ${Math.round(glowLight)}%)`;
      glowCtx.globalAlpha = clamp(alpha * 0.8, 0.05, 0.6);
      glowCtx.beginPath();
      glowCtx.arc(sx, sy, radius * 1.6, 0, TAU);
      glowCtx.fill();
    }

    if (rng > sparkleThreshold) {
      const sparkleAlpha = clamp((rng - sparkleThreshold) * 5, 0.1, 0.8);
      const sparkleHue = wrapHue360(hue + (accent ? (rng - 0.5) * 12 : 0));
      const sparkleSat = Math.round(clamp(saturation + 12, 0, 100));
      ctx.fillStyle = `hsl(${Math.round(sparkleHue)}, ${sparkleSat}%, ${Math.round(sparkleLight)}%)`;
      ctx.globalAlpha = sparkleAlpha;
      ctx.beginPath();
      ctx.arc(sx, sy, radius * 0.45, 0, TAU);
      ctx.fill();
    }
  }

  ctx.restore();
  ctx.globalAlpha = 1;
  if (glowCtx) {
    glowCtx.globalAlpha = 1;
  }
}

export function renderFrame(particles, renderParams = {}, metrics = {}) {
  if (!state.initialized || !state.ctx) {
    return;
  }

  const now = performance.now();
  const dt = Number.isFinite(metrics.dt) ? Math.max(metrics.dt, 1 / 120) : state.lastTime > 0 ? (now - state.lastTime) / 1000 : 1 / 60;
  const frameTimeInstant = Number.isFinite(metrics.frameTime) ? metrics.frameTime : dt * 1000;
  const frameTimeAverage = Number.isFinite(metrics.frameTimeAvg) ? metrics.frameTimeAvg : frameTimeInstant;
  const fpsInstant = Number.isFinite(metrics.fps) && metrics.fps > 0
    ? metrics.fps
    : frameTimeInstant > 0
      ? 1000 / frameTimeInstant
      : 0;
  const fpsAverage = Number.isFinite(metrics.fpsAvg) && metrics.fpsAvg > 0
    ? metrics.fpsAvg
    : frameTimeAverage > 0
      ? 1000 / frameTimeAverage
      : fpsInstant;

  state.lastTime = now;
  state.frameCounter += 1;
  state.frameSeed = state.frameCounter * 0.37;

  ensureCanvasSize();
  adjustDynamicScale(frameTimeInstant, frameTimeAverage);
  updateFps(frameTimeInstant, fpsInstant, fpsAverage);

  const params = resolveParams(renderParams);
  const fadeAlpha = state.toggles.trails ? clamp(1 - params.trailFade, 0.02, 0.35) : 1;

  if (state.toggles.trails) {
    fadeCanvas(TRAIL_BASE_ALPHA + fadeAlpha * 0.75);
  } else if (state.ctx && state.canvas) {
    state.ctx.save();
    state.ctx.setTransform(1, 0, 0, 1, 0, 0);
    state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
    state.ctx.restore();
    state.ctx.setTransform(state.pixelRatio * state.dynamicScale, 0, 0, state.pixelRatio * state.dynamicScale, 0, 0);
  }

  prepareGlow(params.glow);
  drawParticles(particles, params, dt);

  if (state.toggles.grid) {
    drawGrid();
  }

  if (state.glow.enabled) {
    compositeGlow();
  }
}

export default {
  init,
  destroy,
  renderFrame,
  setWorldSize,
  setTrackTitle,
  updateTrackTime,
  setStatus,
  updateVolume,
  getToggles,
  setToggle,
  on,
  off,
};
