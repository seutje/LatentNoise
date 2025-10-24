const SIM_LIMITS = /** @type {const} */ ({
  spawnRate: { min: 0.05, max: 1.35 },
  fieldStrength: { min: 0.2, max: 1.5 },
  cohesion: { min: 0.2, max: 1.25 },
  repelImpulse: { min: 0, max: 1.3 },
  vortexAmount: { min: 0, max: 1.4 },
});

const RENDER_LIMITS = /** @type {const} */ ({
  trailFade: { min: 0.2, max: 0.98 },
  glow: { min: 0, max: 1 },
  sizeJitter: { min: 0, max: 0.8 },
  hueShift: { min: -90, max: 90 },
  sparkleDensity: { min: 0, max: 1 },
  zoom: { min: 0.5, max: 20 },
});

const PRESET_DATA = /** @type {const} */ ([
  {
    id: 'meditation',
    title: 'Meditation',
    description: 'Haze motif with slow pooling and luminous bloom.',
    palette: {
      background: '#040814',
      accents: ['#4ba3ff', '#76c8ff', '#d8f0ff'],
      baseHue: 212,
    },
    sim: {
      spawnRate: { scale: 0.85 },
      fieldStrength: { scale: 0.9 },
      cohesion: { scale: 1.15 },
      repelImpulse: { scale: 0.9 },
      vortexAmount: { scale: 0.8 },
    },
    render: {
      trailFade: { scale: 1.12 },
      glow: { scale: 1.25 },
      sizeJitter: { scale: 0.85 },
      hueShift: { offset: -12 },
      sparkleDensity: { scale: 0.7 },
    },
  },
  {
    id: 'built-on-the-steppers',
    title: 'Built on the Steppers',
    description: 'Structured lattice with strong fields and restrained sparkle.',
    palette: {
      background: '#03261c',
      accents: ['#f94144', '#ff7f11', '#ffd166'],
      baseHue: 80,
    },
    sim: {
      spawnRate: { scale: 3.1 },
      fieldStrength: { scale: 1.2 },
      cohesion: { scale: 1.05 },
      repelImpulse: { scale: 0.75 },
      vortexAmount: { scale: 0.9 },
    },
    render: {
      trailFade: { scale: 0.95 },
      glow: { scale: 1.5 },
      sizeJitter: { scale: 0.8 },
      hueShift: { offset: 8 },
      sparkleDensity: { scale: 0.6 },
    },
  },
  {
    id: 'unsound',
    title: 'Unsound',
    description: 'Transient-driven pulses with aggressive shockwave accents.',
    palette: {
      background: '#0a0811',
      accents: ['#ff6b7c', '#ffd35f', '#ffe8c0'],
      baseHue: 12,
    },
    sim: {
      spawnRate: { scale: 1 },
      fieldStrength: { scale: 1.1 },
      cohesion: { scale: 0.9 },
      repelImpulse: { scale: 1.35 },
      vortexAmount: { scale: 1.05 },
    },
    render: {
      trailFade: { scale: 0.88 },
      glow: { scale: 1.15 },
      sizeJitter: { scale: 1.05 },
      hueShift: { offset: 22 },
      sparkleDensity: { scale: 1.25 },
    },
  },
  {
    id: 'system-js',
    title: 'System.js',
    description: 'Vortex-heavy spin with cool digital gradients.',
    palette: {
      background: '#030c18',
      accents: ['#3e7fff', '#5fa8ff', '#bbf3ff'],
      baseHue: 228,
    },
    sim: {
      spawnRate: { scale: 0.95 },
      fieldStrength: { scale: 1.05 },
      cohesion: { scale: 0.92 },
      repelImpulse: { scale: 1.05 },
      vortexAmount: { scale: 1.4 },
    },
    render: {
      trailFade: { scale: 1 },
      glow: { scale: 1.05 },
      sizeJitter: { scale: 1.1 },
      hueShift: { offset: 48 },
      sparkleDensity: { scale: 0.95 },
    },
  },
  {
    id: 'binary-mirage',
    title: 'Binary Mirage',
    description: 'Inward-folding particles with high cohesion and warm bloom.',
    palette: {
      background: '#150805',
      accents: ['#ff7f50', '#ffc640', '#ffe5a6'],
      baseHue: 26,
    },
    sim: {
      spawnRate: { scale: 0.1 },
      fieldStrength: { scale: 1.15 },
      cohesion: { scale: 1.2 },
      repelImpulse: { scale: 0.9 },
      vortexAmount: { scale: 0.85 },
    },
    render: {
      trailFade: { scale: 1.05 },
      glow: { scale: 1.3 },
      sizeJitter: { scale: 0.95 },
      hueShift: { offset: -6 },
      sparkleDensity: { scale: 0.8 },
    },
  },
  {
    id: 'traffic-jam',
    title: 'Traffic Jam',
    description: 'Stop-go motion with hard-lane repulsion cues.',
    palette: {
      background: '#04321c',
      accents: ['#ff2d55', '#ff8a3d', '#ffc15e'],
      baseHue: 136,
    },
    sim: {
      spawnRate: { scale: 0.25 },
      fieldStrength: { scale: 1.4 },
      cohesion: { scale: 0.98 },
      repelImpulse: { scale: 1.1 },
      vortexAmount: { scale: 0.75 },
    },
    render: {
      trailFade: { scale: 0.9 },
      glow: { scale: 1.2 },
      sizeJitter: { scale: 0.7 },
      hueShift: { offset: 14 },
      sparkleDensity: { scale: 1.05 },
    },
  },
  {
    id: 'backpack',
    title: 'Backpack',
    description: 'Transient bursts with compact life and jitter.',
    palette: {
      background: '#4a3300',
      accents: ['#ffd700', '#ffe866', '#fff4b1'],
      baseHue: 44,
    },
    sim: {
      spawnRate: { scale: 1.5 },
      fieldStrength: { scale: 0.85 },
      cohesion: { scale: 0.85 },
      repelImpulse: { scale: 1.25 },
      vortexAmount: { scale: 0.9 },
    },
    render: {
      trailFade: { scale: 0.86 },
      glow: { scale: 1.5 },
      sizeJitter: { scale: 1.2 },
      hueShift: { offset: -18 },
      sparkleDensity: { scale: 1.15 },
    },
  },
  {
    id: 'last-pack',
    title: 'Last Pack',
    description: 'Arc-like ribbons with pronounced swirl and cohesion.',
    palette: {
      background: '#0d0911',
      accents: ['#8257ff', '#c68bff', '#f0d0ff'],
      baseHue: 268,
    },
    sim: {
      spawnRate: { scale: 1.02 },
      fieldStrength: { scale: 1.08 },
      cohesion: { scale: 1.12 },
      repelImpulse: { scale: 0.95 },
      vortexAmount: { scale: 1.2 },
    },
    render: {
      trailFade: { scale: 1.08 },
      glow: { scale: 1.18 },
      sizeJitter: { scale: 0.92 },
      hueShift: { offset: 32 },
      sparkleDensity: { scale: 0.85 },
    },
  },
  {
    id: 'clouds',
    title: 'Clouds',
    description: 'Soft buoyant plumes with dense bloom and muted sparkle.',
    palette: {
      background: '#061019',
      accents: ['#5bc7ff', '#91e0ff', '#d2f7ff'],
      baseHue: 204,
    },
    sim: {
      spawnRate: { scale: 0.88 },
      fieldStrength: { scale: 0.78 },
      cohesion: { scale: 0.9 },
      repelImpulse: { scale: 0.8 },
      vortexAmount: { scale: 0.95 },
    },
    render: {
      trailFade: { scale: 1.2 },
      glow: { scale: 1.35 },
      sizeJitter: { scale: 0.75 },
      hueShift: { offset: -8 },
      sparkleDensity: { scale: 0.6 },
    },
  },
  {
    id: 'ease-up',
    title: 'Ease Up',
    description: 'Relaxed sway with long trails and gentle warmth.',
    palette: {
      background: '#0b0906',
      accents: ['#ffb366', '#ffd9a3', '#fff1d5'],
      baseHue: 32,
    },
    sim: {
      spawnRate: { scale: 0.92 },
      fieldStrength: { scale: 0.9 },
      cohesion: { scale: 1.05 },
      repelImpulse: { scale: 0.85 },
      vortexAmount: { scale: 1.1 },
    },
    render: {
      trailFade: { scale: 1.18 },
      glow: { scale: 1.1 },
      sizeJitter: { scale: 0.88 },
      hueShift: { offset: 26 },
      sparkleDensity: { scale: 0.72 },
    },
  },
  {
    id: 'epoch-infinity',
    title: 'Epoch ∞',
    description: 'Sparse constellations with drifting hue and low density.',
    palette: {
      background: '#080b14',
      accents: ['#6cf0ff', '#9cf6ff', '#e0ffff'],
      baseHue: 190,
    },
    sim: {
      spawnRate: { scale: 0.65 },
      fieldStrength: { scale: 0.85 },
      cohesion: { scale: 1.08 },
      repelImpulse: { scale: 0.7 },
      vortexAmount: { scale: 0.6 },
    },
    render: {
      trailFade: { scale: 1.22 },
      glow: { scale: 1 },
      sizeJitter: { scale: 1.05 },
      hueShift: { offset: 64 },
      sparkleDensity: { scale: 0.5 },
    },
  },
]);

const PRESET_LOOKUP = new Map();
for (const preset of PRESET_DATA) {
  PRESET_LOOKUP.set(preset.id, preset);
  PRESET_LOOKUP.set(slugify(preset.title), preset);
}

function slugify(name) {
  return String(name)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)+/g, '');
}

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

function applyGroup(target, adjustments, limits) {
  if (!target || !adjustments) {
    return;
  }
  for (const [key, config] of Object.entries(adjustments)) {
    if (typeof target[key] !== 'number') {
      continue;
    }

    let scale = 1;
    let offset = 0;

    if (typeof config === 'number') {
      scale = config;
    } else {
      if (typeof config.scale === 'number') {
        scale = config.scale;
      }
      if (typeof config.offset === 'number') {
        offset = config.offset;
      }
    }

    const next = target[key] * scale + offset;
    const bounds = limits[key];
    if (bounds) {
      target[key] = clamp(next, bounds.min, bounds.max);
    } else {
      target[key] = next;
    }
  }
}

export function listPresets() {
  return PRESET_DATA.map((preset) => ({ ...preset }));
}

export function getPreset(nameOrIndex) {
  if (typeof nameOrIndex === 'number' && Number.isInteger(nameOrIndex)) {
    return PRESET_DATA[nameOrIndex] ?? null;
  }
  if (typeof nameOrIndex === 'string' && nameOrIndex.length > 0) {
    const key = slugify(nameOrIndex);
    return PRESET_LOOKUP.get(key) ?? null;
  }
  if (nameOrIndex == null) {
    return PRESET_DATA[0];
  }
  return null;
}

export function applyPreset(preset, params) {
  if (!preset || !params) {
    return params;
  }

  const hasGroups = typeof params === 'object' && params !== null && ('sim' in params || 'render' in params);
  const result = hasGroups
    ? {
        sim: params.sim ? { ...params.sim } : undefined,
        render: params.render ? { ...params.render } : undefined,
      }
    : { ...params };

  if (hasGroups) {
    if (result.sim) {
      applyGroup(result.sim, preset.sim, SIM_LIMITS);
    }
    if (result.render) {
      applyGroup(result.render, preset.render, RENDER_LIMITS);
    }
  } else {
    applyGroup(result, preset.sim, SIM_LIMITS);
    applyGroup(result, preset.render, RENDER_LIMITS);
  }

  return result;
}

export function getDefaultPreset() {
  return PRESET_DATA[0];
}

export function resolvePreset(nameOrIndex, fallback = null) {
  const preset = getPreset(nameOrIndex);
  return preset ?? fallback;
}

export function describePreset(nameOrIndex) {
  const preset = getPreset(nameOrIndex);
  if (!preset) {
    return null;
  }
  return {
    id: preset.id,
    title: preset.title,
    description: preset.description,
    palette: { ...preset.palette },
  };
}

export const PRESETS = PRESET_DATA;
