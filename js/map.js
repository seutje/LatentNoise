const PARAM_NAMES = Object.freeze([
  'spawnRate',
  'fieldStrength',
  'cohesion',
  'repelImpulse',
  'trailFade',
  'glow',
  'sizeJitter',
  'hueShift',
  'sparkleDensity',
  'vortexAmount',
  'zoom',
]);

const PARAM_SPECS = /** @type {const} */ ({
  spawnRate: {
    index: 0,
    type: 'continuous',
    baseline: 0.35,
    swing: 0.9,
    safeSwing: 0.6,
    min: 0,
    max: 1.2,
    safeMax: 0.8,
    rest: 0,
    smoothingHz: 2.6,
  },
  fieldStrength: {
    index: 1,
    type: 'continuous',
    baseline: 0.6,
    swing: 0.55,
    safeSwing: 0.35,
    min: 0,
    max: 1.5,
    safeMax: 1,
    rest: 0.45,
    smoothingHz: 3.4,
  },
  cohesion: {
    index: 2,
    type: 'continuous',
    baseline: 0.5,
    swing: 0.4,
    safeSwing: 0.28,
    min: 0.1,
    max: 1.1,
    safeMax: 0.85,
    rest: 0.46,
    smoothingHz: 3.1,
  },
  repelImpulse: {
    index: 3,
    type: 'impulse',
    baseline: 0,
    swing: 1,
    safeSwing: 0.7,
    min: 0,
    max: 1,
    safeMax: 0.7,
    rest: 0,
    thresholdOn: 0.55,
    thresholdOff: 0.32,
    hold: 0.12,
    decay: 7,
    floor: 0,
  },
  trailFade: {
    index: 4,
    type: 'continuous',
    baseline: 0.65,
    swing: 0.3,
    safeSwing: 0.2,
    min: 0.2,
    max: 0.95,
    safeMax: 0.8,
    rest: 0.68,
    smoothingHz: 1.9,
  },
  glow: {
    index: 5,
    type: 'continuous',
    baseline: 0.5,
    swing: 0.5,
    safeSwing: 0.35,
    min: 0,
    max: 1,
    safeMax: 0.6,
    rest: 0.45,
    smoothingHz: 2.3,
  },
  sizeJitter: {
    index: 6,
    type: 'continuous',
    baseline: 0.25,
    swing: 0.35,
    safeSwing: 0.22,
    min: 0,
    max: 0.8,
    safeMax: 0.5,
    rest: 0.2,
    smoothingHz: 4.2,
  },
  hueShift: {
    index: 7,
    type: 'continuous',
    baseline: 0,
    swing: 240,
    safeSwing: 160,
    min: -150,
    max: 150,
    safeMax: 80,
    rest: 0,
    smoothingHz: 1.3,
    symmetric: true,
  },
  sparkleDensity: {
    index: 8,
    type: 'impulse',
    baseline: 0.05,
    swing: 0.9,
    safeSwing: 0.55,
    min: 0,
    max: 1,
    safeMax: 0.65,
    rest: 0.04,
    thresholdOn: 0.45,
    thresholdOff: 0.24,
    hold: 0.1,
    decay: 5,
    floor: 0.01,
  },
  vortexAmount: {
    index: 9,
    type: 'continuous',
    baseline: 0.4,
    swing: 0.5,
    safeSwing: 0.32,
    min: 0,
    max: 1.2,
    safeMax: 0.8,
    rest: 0.36,
    smoothingHz: 2,
  },
  zoom: {
    index: 10,
    type: 'continuous',
    baseline: 1,
    swing: 1.2,
    safeSwing: 0.8,
    min: 0.5,
    max: 2,
    safeMax: 1.5,
    rest: 1,
    smoothingHz: 1.8,
  },
});

const DEFAULT_SILENCE_THRESHOLD = 0.03;
const MIN_DT = 1 / 240;
const MAX_DT = 0.5;
const SCRATCH_OUTPUTS = new Float32Array(PARAM_NAMES.length);

class CriticallyDampedSmoother {
  constructor(initialValue = 0, smoothingHz = 2) {
    this.value = initialValue;
    this.velocity = 0;
    this.smoothingHz = Math.max(0, smoothingHz);
  }

  reset(nextValue = 0) {
    this.value = nextValue;
    this.velocity = 0;
  }

  setFrequency(hz) {
    this.smoothingHz = Math.max(0, hz);
  }

  update(target, dt) {
    if (this.smoothingHz <= 0) {
      this.value = target;
      this.velocity = 0;
      return this.value;
    }

    const clampedDt = clamp(dt, MIN_DT, MAX_DT);
    const smoothTime = 1 / this.smoothingHz;
    const omega = 2 / Math.max(smoothTime, 1e-3);
    const x = omega * clampedDt;
    const exp = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);
    const change = this.value - target;
    const temp = (this.velocity + omega * change) * clampedDt;
    this.velocity = (this.velocity - omega * temp) * exp;
    const next = target + (change + temp) * exp;

    const overshoot = (target - this.value) > 0 === (next > target);
    this.value = overshoot ? target : next;
    if (!Number.isFinite(this.value)) {
      this.value = target;
      this.velocity = 0;
    }
    if (!Number.isFinite(this.velocity)) {
      this.velocity = 0;
    }
    return this.value;
  }
}

function clamp(value, min, max) {
  if (Number.isFinite(min) && value < min) {
    return min;
  }
  if (Number.isFinite(max) && value > max) {
    return max;
  }
  return value;
}

function sanitizeArrayLike(values) {
  const result = SCRATCH_OUTPUTS;
  for (let i = 0; i < result.length; i += 1) {
    result[i] = 0;
  }

  if (!values || typeof values.length !== 'number') {
    return result;
  }

  const len = Math.min(values.length, result.length);
  for (let i = 0; i < len; i += 1) {
    const value = Number(values[i]);
    result[i] = Number.isFinite(value) ? clamp(value, -1, 1) : 0;
  }
  return result;
}

function resolveBounds(spec, safeMode) {
  let min = typeof spec.min === 'number' ? spec.min : Number.NEGATIVE_INFINITY;
  let max = typeof spec.max === 'number' ? spec.max : Number.POSITIVE_INFINITY;

  if (safeMode) {
    if (spec.symmetric && typeof spec.safeMax === 'number') {
      max = Math.min(max, spec.safeMax);
      min = Math.max(min, -spec.safeMax);
    } else {
      if (typeof spec.safeMin === 'number') {
        min = Math.max(min, spec.safeMin);
      }
      if (typeof spec.safeMax === 'number') {
        max = Math.min(max, spec.safeMax);
      }
    }
  }

  return { min, max };
}

function resolveSwing(spec, safeMode) {
  const swing = safeMode && typeof spec.safeSwing === 'number' ? spec.safeSwing : spec.swing;
  return Number.isFinite(swing) ? swing : 0;
}

function getTimestamp() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function computeActivity(outputs, features, fallback) {
  if (features && typeof features === 'object') {
    if (typeof features.rms === 'number') {
      return Math.max(0, features.rms);
    }
    if (Array.isArray(features) || features instanceof Float32Array) {
      const candidate = Number(features[5]);
      if (Number.isFinite(candidate)) {
        return Math.max(0, candidate);
      }
    }
  }

  if (!outputs || outputs.length === 0) {
    return fallback;
  }

  let sum = 0;
  for (let i = 0; i < outputs.length; i += 1) {
    sum += Math.abs(outputs[i]);
  }
  return sum / outputs.length;
}

function createImpulseState(spec) {
  return {
    envelope: spec.rest ?? 0,
    thresholdOn: spec.thresholdOn ?? 0.5,
    thresholdOff: spec.thresholdOff ?? 0.25,
    holdTime: spec.hold ?? 0.1,
    decay: spec.decay ?? 6,
    floor: spec.floor ?? 0,
    active: false,
    hold: 0,
  };
}

function updateImpulseEnvelope(state, input, dt) {
  if (!Number.isFinite(input)) {
    input = 0;
  }

  if (input >= state.thresholdOn) {
    state.active = true;
    state.hold = state.holdTime;
    state.envelope = Math.max(state.envelope, input);
  } else if (state.active) {
    if (input >= state.thresholdOff) {
      state.hold = state.holdTime;
      state.envelope = Math.max(state.envelope, input);
    } else {
      state.hold -= dt;
      if (state.hold <= 0) {
        state.active = false;
      }
    }
  }

  const decayRate = Math.max(state.decay, 1);
  const decayFactor = Math.exp(-decayRate * clamp(dt, MIN_DT, MAX_DT));

  if (state.active) {
    const decayed = state.envelope * decayFactor;
    state.envelope = Math.max(decayed, input);
  } else {
    state.envelope *= decayFactor;
    if (state.envelope < state.floor) {
      state.envelope = state.floor;
    }
  }

  if (!Number.isFinite(state.envelope)) {
    state.envelope = state.floor;
  }
  return state.envelope;
}

function createDefaultParams() {
  const params = {};
  for (const name of PARAM_NAMES) {
    params[name] = PARAM_SPECS[name].baseline;
  }
  return params;
}

function createRestParams() {
  const rests = {};
  for (const name of PARAM_NAMES) {
    const spec = PARAM_SPECS[name];
    rests[name] = typeof spec.rest === 'number' ? spec.rest : spec.baseline;
  }
  return rests;
}

const state = {
  params: createDefaultParams(),
  baselines: createDefaultParams(),
  rests: createRestParams(),
  smoothers: new Map(),
  impulses: new Map(),
  safeMode: false,
  silenceThreshold: DEFAULT_SILENCE_THRESHOLD,
  lastTimestamp: 0,
  lastOutputs: new Float32Array(PARAM_NAMES.length),
};

function getSmoother(name, spec) {
  let smoother = state.smoothers.get(name);
  if (!smoother) {
    smoother = new CriticallyDampedSmoother(spec.baseline, spec.smoothingHz ?? 2);
    state.smoothers.set(name, smoother);
  } else if (typeof spec.smoothingHz === 'number') {
    smoother.setFrequency(spec.smoothingHz);
  }
  return smoother;
}

function getImpulseState(name, spec) {
  let impulse = state.impulses.get(name);
  if (!impulse) {
    impulse = createImpulseState(spec);
    state.impulses.set(name, impulse);
  }
  return impulse;
}

function applyContinuous(name, spec, rawValue, dt, silence) {
  const swing = resolveSwing(spec, state.safeMode);
  const { min, max } = resolveBounds(spec, state.safeMode);
  const smoother = getSmoother(name, spec);
  const baseline = state.baselines[name] ?? spec.baseline;
  const rest = state.rests[name] ?? spec.rest ?? baseline;
  const target = silence ? rest : baseline + rawValue * swing;
  const clampedTarget = clamp(target, min, max);
  const value = smoother.update(clampedTarget, dt);
  state.params[name] = clamp(value, min, max);
}

function applyImpulse(name, spec, rawValue, dt, silence) {
  const swing = resolveSwing(spec, state.safeMode);
  const { min, max } = resolveBounds(spec, state.safeMode);
  const impulseState = getImpulseState(name, spec);
  const baseline = state.baselines[name] ?? spec.baseline;
  const rest = state.rests[name] ?? spec.rest ?? baseline;

  if (silence) {
    impulseState.active = false;
    impulseState.hold = 0;
    impulseState.envelope = Math.max(rest, impulseState.envelope * Math.exp(-impulseState.decay * dt));
  } else {
    const positive = Math.max(0, rawValue);
    const normalized = positive > 1 ? 1 : positive;
    updateImpulseEnvelope(impulseState, normalized, dt);
  }

  const envelope = clamp(impulseState.envelope, 0, 1);
  const target = silence ? rest : baseline + envelope * swing;
  state.params[name] = clamp(target, min, max);
}

export function setSafeMode(enabled) {
  state.safeMode = Boolean(enabled);
}

export function configure(options = {}) {
  if (typeof options.safeMode === 'boolean') {
    setSafeMode(options.safeMode);
  }
  if (typeof options.silenceThreshold === 'number' && Number.isFinite(options.silenceThreshold)) {
    state.silenceThreshold = clamp(options.silenceThreshold, 0, 1);
  }
}

export function reset(params) {
  state.lastTimestamp = 0;
  state.lastOutputs.fill(0);
  for (const name of PARAM_NAMES) {
    const spec = PARAM_SPECS[name];
    const { min, max } = resolveBounds(spec, state.safeMode);
    const baseline = params && typeof params[name] === 'number' ? clamp(params[name], min, max) : spec.baseline;
    const restBase = typeof spec.rest === 'number' ? spec.rest : spec.baseline;
    const rest = clamp(restBase + (baseline - spec.baseline), min, max);
    state.baselines[name] = baseline;
    state.rests[name] = rest;
    state.params[name] = baseline;

    const smoother = state.smoothers.get(name);
    if (smoother) {
      smoother.reset(baseline);
    }

    const impulse = state.impulses.get(name);
    if (impulse) {
      impulse.envelope = rest;
      impulse.active = false;
      impulse.hold = 0;
    }
  }
}

export function update(outputs, options = {}) {
  const cleanedOutputs = sanitizeArrayLike(outputs);
  state.lastOutputs.set(cleanedOutputs);

  let dt = Number.isFinite(options.dt) ? options.dt : 0;
  let now = Number.isFinite(options.timestamp) ? options.timestamp : 0;

  if (!Number.isFinite(dt) || dt <= 0) {
    if (!Number.isFinite(now) || now <= 0) {
      now = getTimestamp();
    }
    if (state.lastTimestamp === 0) {
      dt = 1 / 60;
    } else {
      dt = (now - state.lastTimestamp) / 1000;
    }
  }

  if (!Number.isFinite(now) || now <= 0) {
    now = state.lastTimestamp > 0 ? state.lastTimestamp + dt * 1000 : getTimestamp();
  }

  state.lastTimestamp = now;
  dt = clamp(dt, MIN_DT, MAX_DT);

  const activity = Number.isFinite(options.activity)
    ? options.activity
    : computeActivity(cleanedOutputs, options.features, state.silenceThreshold);

  const silence = activity < state.silenceThreshold;

  for (const name of PARAM_NAMES) {
    const spec = PARAM_SPECS[name];
    const raw = cleanedOutputs[spec.index] ?? 0;
    if (spec.type === 'continuous') {
      applyContinuous(name, spec, raw, dt, silence);
    } else {
      applyImpulse(name, spec, raw, dt, silence);
    }
  }

  return state.params;
}

export function getParams() {
  return { ...state.params };
}

export function getLastOutputs() {
  return state.lastOutputs.slice();
}

export function getParamNames() {
  return PARAM_NAMES.slice();
}

export function getParamSpec(name) {
  return PARAM_SPECS[name] ? { ...PARAM_SPECS[name] } : null;
}

export { PARAM_NAMES };
