// Latent Noise playlist is fixed to the bundled 11 album tracks.
// Titles mirror the design spec; filenames map to assets/audio.
const TRACKS = Object.freeze([
  Object.freeze({ title: 'Meditation', file: 'Meditation.mp3' }),
  Object.freeze({ title: 'Built on the Steppers', file: 'Built on the Steppers.mp3' }),
  Object.freeze({ title: 'Unsound', file: 'Unsound.mp3' }),
  Object.freeze({ title: 'System.js', file: 'System.js.mp3' }),
  Object.freeze({ title: 'Binary Mirage', file: 'Binary Mirage.mp3' }),
  Object.freeze({ title: 'Traffic Jam', file: 'Traffic Jam.mp3' }),
  Object.freeze({ title: 'Backpack', file: 'Backpack.mp3' }),
  Object.freeze({ title: 'Last Pack', file: 'Last Pack.mp3' }),
  Object.freeze({ title: 'Clouds', file: 'Clouds.mp3' }),
  Object.freeze({ title: 'Ease Up', file: 'Ease Up.mp3' }),
  Object.freeze({ title: 'Epoch ∞', file: 'Epoch ∞.mp3' }),
]);

export function getList() {
  // Return a shallow copy so callers cannot mutate the canonical list.
  return TRACKS.slice();
}

export function resolveUrl(index) {
  const track = TRACKS[index];
  if (!track) {
    throw new RangeError(`Track index out of bounds: ${index}`);
  }
  return `assets/audio/${encodeURI(track.file)}`;
}

export function count() {
  return TRACKS.length;
}
