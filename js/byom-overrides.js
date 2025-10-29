const HEX_SHORT = /^[0-9a-fA-F]{3}$/;
const HEX_LONG = /^[0-9a-fA-F]{6}$/;
const BASE_HUE_MIN = 0;
const BASE_HUE_MAX = 360;

function clampHue(value) {
  if (!Number.isFinite(value)) {
    return BASE_HUE_MIN;
  }
  if (value < BASE_HUE_MIN) {
    return BASE_HUE_MIN;
  }
  if (value > BASE_HUE_MAX) {
    return BASE_HUE_MAX;
  }
  return value;
}

export function normalizeHexColor(color) {
  if (typeof color !== 'string') {
    return null;
  }
  const trimmed = color.trim();
  if (trimmed.length === 0) {
    return null;
  }
  const withoutHash = trimmed.startsWith('#') ? trimmed.slice(1) : trimmed;
  if (HEX_LONG.test(withoutHash)) {
    return `#${withoutHash.toLowerCase()}`;
  }
  if (HEX_SHORT.test(withoutHash)) {
    const expanded = withoutHash
      .toLowerCase()
      .split('')
      .map((char) => char + char)
      .join('');
    return `#${expanded}`;
  }
  return null;
}

function sanitizeNumericGroup(source) {
  if (!source || typeof source !== 'object') {
    return null;
  }
  const result = {};
  for (const [key, value] of Object.entries(source)) {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) {
      result[key] = numeric;
    }
  }
  return Object.keys(result).length > 0 ? result : null;
}

function sanitizePaletteGroup(source) {
  if (!source || typeof source !== 'object') {
    return null;
  }
  const result = {};
  if (typeof source.background === 'string') {
    const normalized = normalizeHexColor(source.background);
    if (normalized) {
      result.background = normalized;
    }
  }
  const hue = Number(source.baseHue);
  if (Number.isFinite(hue)) {
    result.baseHue = clampHue(hue);
  }
  return Object.keys(result).length > 0 ? result : null;
}

function sanitizePresetOverrides(overrides) {
  if (!overrides || typeof overrides !== 'object') {
    return null;
  }
  const normalized = {};
  let hasValue = false;

  const sim = sanitizeNumericGroup(overrides.sim);
  if (sim) {
    normalized.sim = sim;
    hasValue = true;
  }

  const render = sanitizeNumericGroup(overrides.render);
  if (render) {
    normalized.render = render;
    hasValue = true;
  }

  const palette = sanitizePaletteGroup(overrides.palette);
  if (palette) {
    normalized.palette = palette;
    hasValue = true;
  }

  return hasValue ? normalized : null;
}

export function normalizePresetOverrides(overrides) {
  return sanitizePresetOverrides(overrides);
}

export function clonePresetOverrides(overrides) {
  const sanitized = sanitizePresetOverrides(overrides);
  if (!sanitized) {
    return null;
  }
  const cloned = {};
  if (sanitized.sim) {
    cloned.sim = { ...sanitized.sim };
  }
  if (sanitized.render) {
    cloned.render = { ...sanitized.render };
  }
  if (sanitized.palette) {
    cloned.palette = { ...sanitized.palette };
  }
  return cloned;
}

export function mergePaletteOverrides(basePalette, overrides) {
  const palette = basePalette && typeof basePalette === 'object' ? { ...basePalette } : {};
  if (Array.isArray(basePalette?.accents)) {
    palette.accents = basePalette.accents.slice();
  }

  if (overrides && typeof overrides === 'object') {
    if (typeof overrides.background === 'string') {
      const normalizedBackground = normalizeHexColor(overrides.background);
      if (normalizedBackground) {
        palette.background = normalizedBackground;
      }
    }
    const hue = Number(overrides.baseHue);
    if (Number.isFinite(hue)) {
      palette.baseHue = clampHue(hue);
    }
  }

  if (typeof palette.background === 'string') {
    const normalized = normalizeHexColor(palette.background);
    palette.background = normalized ?? palette.background;
  }

  return palette;
}
