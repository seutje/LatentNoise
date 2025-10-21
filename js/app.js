import * as audio from './audio.js';
import * as nn from './nn.js';
import * as physics from './physics.js';
import * as render from './render.js';
import { getList, resolveUrl } from './playlist.js';

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

const RENDER_PARAMS_DEFAULT = Object.freeze({
  trailFade: 0.68,
  glow: 0.55,
  sizeJitter: 0.32,
  hueShift: 0,
  sparkleDensity: 0.14,
});

const SIM_PARAMS_DEFAULT = Object.freeze({
  spawnRate: 0.45,
  fieldStrength: 0.62,
  cohesion: 0.54,
  repelImpulse: 0,
  vortexAmount: 0.28,
});

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

const playlistSelect = document.getElementById('playlist');
const audioElement = document.getElementById('player');
const volumeSlider = document.getElementById('volume');

if (!playlistSelect || !audioElement || !volumeSlider) {
  throw new Error('Required controls missing from DOM (playlist, audio, or volume).');
}

render.init();
render.setWorldSize(2, 2);
render.setStatus('Idle · Particles 0');

physics.configure({
  bounds: { width: 2, height: 2, mode: 'wrap' },
  baseCap: 5200,
  minCap: 800,
  defaults: {
    spawnRate: SIM_PARAMS_DEFAULT.spawnRate,
    fieldStrength: SIM_PARAMS_DEFAULT.fieldStrength,
    cohesion: SIM_PARAMS_DEFAULT.cohesion,
    repelImpulse: SIM_PARAMS_DEFAULT.repelImpulse,
    vortexAmount: SIM_PARAMS_DEFAULT.vortexAmount,
  },
});

const renderParams = { ...RENDER_PARAMS_DEFAULT };
const simParams = { ...SIM_PARAMS_DEFAULT };

const tracks = getList();
if (tracks.length === 0) {
  throw new Error('Playlist is empty; Phase 2 requires 11 static tracks.');
}
if (MODEL_FILES.length !== tracks.length) {
  throw new Error('Model placeholder count mismatch with playlist length.');
}

// Remove any stray options before populating the locked playlist.
playlistSelect.innerHTML = '';
tracks.forEach((track, index) => {
  const option = document.createElement('option');
  option.value = String(index);
  option.textContent = track.title;
  option.dataset.src = resolveUrl(index);
  playlistSelect.append(option);
});

render.setTrackTitle(tracks[0]?.title ?? 'Latent Noise');

const modelCache = new Map();
let currentTrackIndex = -1;
let activeModelIndex = -1;
let modelLoadToken = 0;

const playback = {
  status: 'Idle',
  lastStatusText: '',
};

function updateStatus(metrics) {
  const count = metrics?.count ?? 0;
  const cap = metrics?.dynamicCap ?? 0;
  const statusText = `${playback.status} · Particles ${count}/${cap}`;
  if (statusText !== playback.lastStatusText) {
    render.setStatus(statusText);
    playback.lastStatusText = statusText;
  }
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
    nn.forward(normalized);
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
  render.setTrackTitle(target?.title ?? `Track ${index + 1}`);
  render.updateTrackTime(0, Number.isFinite(audioElement.duration) ? audioElement.duration : NaN);
  playback.status = autoplay ? 'Buffering' : 'Idle';
  updateStatus(physics.getMetrics());

  void prepareModel(index);

  if (autoplay) {
    audioElement.play().catch((error) => {
      playback.status = 'Idle';
      updateStatus(physics.getMetrics());
      console.warn('[app] Autoplay blocked', error);
    });
  }
}

function nextTrack(step = 1) {
  if (tracks.length === 0) {
    return;
  }
  const nextIndex = (currentTrackIndex + step + tracks.length) % tracks.length;
  setTrack(nextIndex, { autoplay: !audioElement.paused });
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
}

// Default to the first track and ensure the audio element points to bundled media only.
setTrack(0, { autoplay: false });

const restoredVolume = audio.init(audioElement);
const initialVolume = Number.isFinite(restoredVolume) ? restoredVolume : Number(volumeSlider.value);
volumeSlider.value = initialVolume.toFixed(2);
audio.setVolume(initialVolume);
render.updateVolume(initialVolume);

volumeSlider.addEventListener('input', () => {
  const nextVolume = Number(volumeSlider.value);
  if (Number.isNaN(nextVolume)) {
    return;
  }
  audio.setVolume(nextVolume);
  render.updateVolume(nextVolume);
});

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
  simParams.spawnRate = clamp(simParams.spawnRate + delta, 0.05, 1.2);
});
render.on('adjustIntensity', ({ delta }) => {
  if (!Number.isFinite(delta)) {
    return;
  }
  renderParams.glow = clamp(renderParams.glow + delta, 0, 1);
  renderParams.sparkleDensity = clamp(renderParams.sparkleDensity + delta * 0.6, 0, 1);
});
render.on('cyclePalette', ({ direction }) => {
  const dir = direction >= 0 ? 1 : -1;
  renderParams.hueShift = (renderParams.hueShift + dir * 20) % 360;
});
render.on('safeModeChange', (enabled) => {
  if (enabled) {
    renderParams.glow = Math.min(renderParams.glow, 0.3);
    renderParams.sparkleDensity = Math.min(renderParams.sparkleDensity, 0.08);
    simParams.spawnRate = Math.min(simParams.spawnRate, 0.55);
  } else {
    renderParams.glow = Math.max(renderParams.glow, RENDER_PARAMS_DEFAULT.glow);
    renderParams.sparkleDensity = Math.max(renderParams.sparkleDensity, RENDER_PARAMS_DEFAULT.sparkleDensity);
    simParams.spawnRate = Math.max(simParams.spawnRate, SIM_PARAMS_DEFAULT.spawnRate);
  }
});
render.on('resolutionChange', ({ scale }) => {
  if (Number.isFinite(scale) && scale > 0.3) {
    physics.configure({ baseCap: Math.round(5200 * clamp(scale, 0.55, 1)) });
  }
});

audioElement.addEventListener('play', () => {
  playback.status = 'Playing';
  updateStatus(physics.getMetrics());
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
});

audioElement.addEventListener('ended', () => {
  playback.status = 'Ended';
  updateStatus(physics.getMetrics());
});

const updateTrackTime = () => {
  render.updateTrackTime(audioElement.currentTime, audioElement.duration);
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

let lastFrameTime = performance.now();

function frame(now) {
  const dtMs = now - lastFrameTime;
  lastFrameTime = now;
  const dtSeconds = clamp(dtMs / 1000, 1 / 240, 1 / 20);

  physics.step(simParams, { dt: dtSeconds, frameTime: dtMs });
  const particles = physics.getParticles();
  const metrics = physics.getMetrics();

  render.renderFrame(particles, renderParams, { dt: dtSeconds, frameTime: dtMs });
  updateStatus(metrics);

  if (!audioElement.paused && audioElement.readyState >= 1) {
    render.updateTrackTime(audioElement.currentTime, audioElement.duration);
  }

  audio.frame();

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
