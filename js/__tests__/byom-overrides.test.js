import {
  clonePresetOverrides,
  mergePaletteOverrides,
  normalizeHexColor,
  normalizePresetOverrides,
} from '../byom-overrides.js';

describe('byom-overrides', () => {
  test('normalizeHexColor standardizes short and long hex values', () => {
    expect(normalizeHexColor('#ABC')).toBe('#aabbcc');
    expect(normalizeHexColor('123456')).toBe('#123456');
    expect(normalizeHexColor('   #ff00aa   ')).toBe('#ff00aa');
    expect(normalizeHexColor('not-a-color')).toBeNull();
  });

  test('clonePresetOverrides keeps numeric groups and palette overrides', () => {
    const overrides = {
      sim: { spawnRate: '0.5', invalid: 'nope' },
      render: { glow: 0.8 },
      palette: { background: '#ABC', baseHue: 480 },
    };
    const cloned = clonePresetOverrides(overrides);
    expect(cloned).toEqual({
      sim: { spawnRate: 0.5 },
      render: { glow: 0.8 },
      palette: { background: '#aabbcc', baseHue: 360 },
    });
  });

  test('normalizePresetOverrides returns null when nothing valid is supplied', () => {
    expect(normalizePresetOverrides(null)).toBeNull();
    expect(normalizePresetOverrides({ palette: { background: 'oops' } })).toBeNull();
  });

  test('mergePaletteOverrides applies overrides while preserving base accents', () => {
    const base = { background: '#101010', baseHue: 200, accents: ['#fff', '#000'] };
    const overrides = { background: '#3498db', baseHue: 120 };
    const merged = mergePaletteOverrides(base, overrides);
    expect(merged.background).toBe('#3498db');
    expect(merged.baseHue).toBe(120);
    expect(merged.accents).toEqual(['#fff', '#000']);
  });
});
