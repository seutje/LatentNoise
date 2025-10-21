import { getList, resolveUrl } from './playlist.js';

const playlistSelect = document.getElementById('playlist');
const audioElement = document.getElementById('player');

if (!playlistSelect || !audioElement) {
  throw new Error('Playlist select or audio element missing from DOM.');
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
