import * as audio from './audio.js';
import * as nn from './nn.js';
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

const playlistSelect = document.getElementById('playlist');
const audioElement = document.getElementById('player');
const volumeSlider = document.getElementById('volume');

if (!playlistSelect || !audioElement || !volumeSlider) {
  throw new Error('Required controls missing from DOM (playlist, audio, or volume).');
}

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

const modelCache = new Map();
let currentTrackIndex = -1;
let activeModelIndex = -1;
let modelLoadToken = 0;

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

function setTrack(index) {
  if (!Number.isInteger(index) || index < 0 || index >= tracks.length) {
    console.warn('[app] Ignoring out-of-range track index', index);
    return;
  }
  if (index === currentTrackIndex && activeModelIndex === index) {
    return;
  }

  currentTrackIndex = index;
  playlistSelect.selectedIndex = index;
  playlistSelect.value = String(index);
  audioElement.src = resolveUrl(index);
  void prepareModel(index);
}

// Default to the first track and ensure the audio element points to bundled media only.
setTrack(0);

const restoredVolume = audio.init(audioElement);
const initialVolume = Number.isFinite(restoredVolume) ? restoredVolume : Number(volumeSlider.value);
volumeSlider.value = initialVolume.toFixed(2);
audio.setVolume(initialVolume);

volumeSlider.addEventListener('input', () => {
  const nextVolume = Number(volumeSlider.value);
  if (Number.isNaN(nextVolume)) {
    return;
  }
  audio.setVolume(nextVolume);
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
  setTrack(selected);
});

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
