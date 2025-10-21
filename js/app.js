import * as audio from './audio.js';
import { getList, resolveUrl } from './playlist.js';

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

// Remove any stray options before populating the locked playlist.
playlistSelect.innerHTML = '';

tracks.forEach((track, index) => {
  const option = document.createElement('option');
  option.value = String(index);
  option.textContent = track.title;
  option.dataset.src = resolveUrl(index);
  playlistSelect.append(option);
});

// Default to the first track and ensure the audio element points to bundled media only.
playlistSelect.selectedIndex = 0;
audioElement.src = resolveUrl(0);

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
  audioElement.src = resolveUrl(selected);
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
