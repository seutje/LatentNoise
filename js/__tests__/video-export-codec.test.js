import { selectCodec, __test } from '../video-export-codec.js';

describe('video-export-codec', () => {
  it('returns preferred codec when provided', () => {
    expect(selectCodec(640, 360, 'custom-codec')).toBe('custom-codec');
  });

  it('selects baseline profile for small resolutions', () => {
    expect(selectCodec(640, 360)).toBe('avc1.42001E');
  });

  it('selects main profile for 1080p', () => {
    expect(selectCodec(1920, 1080)).toBe('avc1.4D0028');
  });

  it('selects high profile with level 5.1 for 3128x1916 resolution', () => {
    expect(selectCodec(3128, 1916)).toBe('avc1.640033');
  });

  it('falls back to highest config when dimensions are extreme', () => {
    const lastConfig = __test.AVC_LEVEL_CONFIGS[__test.AVC_LEVEL_CONFIGS.length - 1];
    expect(selectCodec(8000, 8000)).toBe(`avc1.${lastConfig.profileHex}00${lastConfig.levelHex}`);
  });
});
