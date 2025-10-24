export const TRACK_MODELS = Object.freeze([
  { index: 0, name: 'Meditation', slug: 'meditation', file: 'meditation.json' },
  { index: 1, name: 'Built on the Steppers', slug: 'built-on-the-steppers', file: 'built-on-the-steppers.json' },
  { index: 2, name: 'Unsound', slug: 'unsound', file: 'unsound.json' },
  { index: 3, name: 'System.js', slug: 'system-js', file: 'system-js.json' },
  { index: 4, name: 'Binary Mirage', slug: 'binary-mirage', file: 'binary-mirage.json' },
  { index: 5, name: 'Traffic Jam', slug: 'traffic-jam', file: 'traffic-jam.json' },
  { index: 6, name: 'Backpack', slug: 'backpack', file: 'backpack.json' },
  { index: 7, name: 'Last Pack', slug: 'last-pack', file: 'last-pack.json' },
  { index: 8, name: 'Clouds', slug: 'clouds', file: 'clouds.json' },
  { index: 9, name: 'Ease Up', slug: 'ease-up', file: 'ease-up.json' },
  { index: 10, name: 'Epoch ∞', slug: 'epoch-infinity', file: 'epoch-infinity.json' },
]);

/**
 * Resolve a track definition from a CLI token.
 * Accepts numeric indices, slugs, filenames, or exact names (case-insensitive).
 * @param {string|number} reference
 * @returns {{index:number,name:string,slug:string,file:string}|null}
 */
export function findTrack(reference) {
  if (reference === null || reference === undefined) {
    return null;
  }

  const token = String(reference).trim();
  if (token.length === 0) {
    return null;
  }

  const normalized = token.toLowerCase();

  if (/^\d+$/.test(normalized)) {
    const index = Number.parseInt(normalized, 10);
    const byIndex = TRACK_MODELS.find((entry) => entry.index === index);
    if (byIndex) {
      return byIndex;
    }
  }

  return (
    TRACK_MODELS.find((entry) =>
      entry.slug === normalized ||
      entry.file.toLowerCase() === normalized ||
      entry.name.toLowerCase() === normalized,
    ) || null
  );
}

/**
 * Provide a formatted list of track references for help messages.
 * @returns {string}
 */
export function formatTrackList() {
  return TRACK_MODELS.map((track) => `[${track.index}] ${track.name}`).join(', ');
}
