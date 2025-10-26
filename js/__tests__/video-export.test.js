import { buildExportFileName, selectMp4MimeType, MP4_MIME_CANDIDATES } from '../video-export.js';

describe('video-export helpers', () => {
  test('selectMp4MimeType returns the first supported candidate', () => {
    const supported = new Set([MP4_MIME_CANDIDATES[1]]);
    const result = selectMp4MimeType((type) => supported.has(type));
    expect(result).toBe(MP4_MIME_CANDIDATES[1]);
  });

  test('selectMp4MimeType returns null when no candidates supported', () => {
    const result = selectMp4MimeType(() => false);
    expect(result).toBeNull();
  });

  test('buildExportFileName normalizes strings into safe slugs', () => {
    expect(buildExportFileName('Sunrise Overdrive')).toBe('sunrise-overdrive.mp4');
    expect(buildExportFileName('   ')).toBe('latent-noise.mp4');
    expect(buildExportFileName('Night-Flight!', 'mov')).toBe('night-flight.mov');
  });
});
