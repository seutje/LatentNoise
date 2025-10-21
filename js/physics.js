/**
 * Physics core (Phase 8)
 *
 * Implements a pooled particle system using typed arrays, a semi-implicit Euler
 * integrator, multiple force fields, and dynamic particle caps that respond to
 * frame time hints. The API is purposefully self-contained so later phases can
 * hook rendering and orchestration layers without mutating internal buffers.
 */

const DEFAULT_CAPACITY = 8192;
const BASE_MAX_PARTICLES = 5200;
const MIN_DYNAMIC_CAP = 600;
const DEFAULT_DRAG = 0.18;
const MIN_DT = 1 / 240;
const MAX_DT = 1 / 30;
const TARGET_FRAME_MS = 16.67;
const OVERLOAD_THRESHOLD_MS = 21;
const RECOVERY_THRESHOLD_MS = 16.7;
const SPAWN_RATE_MIN = 160; // particles per second at spawnRate=0
const SPAWN_RATE_MAX = 640; // particles per second at spawnRate=1
const LIFE_MIN = 2.8;
const LIFE_MAX = 7.2;
const FLOW_FREQ_X = 0.37;
const FLOW_FREQ_Y = 0.41;
const FLOW_TIME_SCALE = 0.18;
const EPSILON = 1e-6;
const BOUNDS_SCRATCH = { x: 0, y: 0, vx: 0, vy: 0 };

/**
 * @typedef {Object} PhysicsParams
 * @property {number} [spawnRate]
 * @property {number} [fieldStrength]
 * @property {number} [cohesion]
 * @property {number} [repelImpulse]
 * @property {number} [vortexAmount]
 */

/**
 * @typedef {Object} StepOptions
 * @property {number} [dt]
 * @property {number} [frameTime] Instantaneous frame time in ms.
 * @property {number} [frameTimeAvg] Rolling-average frame time in ms.
 */

const state = {
  capacity: 0,
  baseCap: BASE_MAX_PARTICLES,
  minCap: MIN_DYNAMIC_CAP,
  dynamicCap: BASE_MAX_PARTICLES,
  freeList: /** @type {Uint32Array|null} */ (null),
  freeTop: 0,
  liveList: /** @type {Uint32Array|null} */ (null),
  liveMap: /** @type {Int32Array|null} */ (null),
  killQueue: /** @type {Uint32Array|null} */ (null),
  killCount: 0,
  liveCount: 0,
  activeCount: 0,
  spawnAccumulator: 0,
  drag: DEFAULT_DRAG,
  time: 0,
  centerX: 0,
  centerY: 0,
  repelStrength: 0,
  rngState: 0x1234abcd,
  // Typed buffers
  posX: /** @type {Float32Array|null} */ (null),
  posY: /** @type {Float32Array|null} */ (null),
  velX: /** @type {Float32Array|null} */ (null),
  velY: /** @type {Float32Array|null} */ (null),
  life: /** @type {Float32Array|null} */ (null),
  maxLife: /** @type {Float32Array|null} */ (null),
  mass: /** @type {Float32Array|null} */ (null),
  seed: /** @type {Float32Array|null} */ (null),
  alive: /** @type {Uint8Array|null} */ (null),
  wells: [
    { x: 0, y: 0, strength: 0 },
    { x: 0, y: 0, strength: 0 },
    { x: 0, y: 0, strength: 0 },
  ],
  bounds: {
    minX: -1,
    maxX: 1,
    minY: -1,
    maxY: 1,
    width: 2,
    height: 2,
    mode: /** @type {'wrap' | 'reflect'} */ ('wrap'),
  },
  defaults: {
    spawnRate: 0.35,
    fieldStrength: 0.6,
    cohesion: 0.5,
    repelImpulse: 0,
    vortexAmount: 0.18,
  },
  metrics: {
    frameTime: TARGET_FRAME_MS,
    frameTimeInstant: TARGET_FRAME_MS,
    fps: 60,
    trimmedLastFrame: false,
    overloadedFrames: 0,
  },
};

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

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function ensurePool(capacity) {
  if (state.capacity === capacity && state.posX) {
    return;
  }

  state.capacity = capacity;
  state.posX = new Float32Array(capacity);
  state.posY = new Float32Array(capacity);
  state.velX = new Float32Array(capacity);
  state.velY = new Float32Array(capacity);
  state.life = new Float32Array(capacity);
  state.maxLife = new Float32Array(capacity);
  state.mass = new Float32Array(capacity);
  state.seed = new Float32Array(capacity);
  state.alive = new Uint8Array(capacity);
  state.freeList = new Uint32Array(capacity);
  state.liveList = new Uint32Array(capacity);
  state.liveMap = new Int32Array(capacity);
  state.killQueue = new Uint32Array(capacity);

  for (let i = 0; i < capacity; i++) {
    state.freeList[i] = capacity - 1 - i;
    state.liveMap[i] = -1;
  }
  state.freeTop = capacity;
  state.liveCount = 0;
  state.activeCount = 0;
  state.killCount = 0;
}

function resetPools() {
  if (!state.posX || !state.posY || !state.freeList || !state.alive) {
    return;
  }

  state.posX.fill(0);
  state.posY.fill(0);
  state.velX.fill(0);
  state.velY.fill(0);
  state.life.fill(0);
  state.maxLife.fill(0);
  state.mass.fill(0);
  state.seed.fill(0);
  state.alive.fill(0);

  for (let i = 0; i < state.capacity; i++) {
    state.freeList[i] = state.capacity - 1 - i;
    state.liveMap[i] = -1;
  }

  state.freeTop = state.capacity;
  state.liveCount = 0;
  state.activeCount = 0;
  state.killCount = 0;
}

function toBounds(width, height) {
  const w = Math.max(width, EPSILON);
  const h = Math.max(height, EPSILON);
  return {
    minX: -w * 0.5,
    maxX: w * 0.5,
    minY: -h * 0.5,
    maxY: h * 0.5,
    width: w,
    height: h,
    mode: state.bounds.mode,
  };
}

function sanitizeDt(dt) {
  if (!Number.isFinite(dt) || dt <= 0) {
    return 1 / 60;
  }
  if (dt < MIN_DT) {
    return MIN_DT;
  }
  if (dt > MAX_DT) {
    return MAX_DT;
  }
  return dt;
}

function sanitizeParams(input = /** @type {PhysicsParams} */ ({})) {
  const defaults = state.defaults;
  const spawnRate = clamp(
    Number.isFinite(input.spawnRate) ? input.spawnRate : defaults.spawnRate,
    0,
    1.5,
  );
  const fieldStrength = clamp(
    Number.isFinite(input.fieldStrength) ? input.fieldStrength : defaults.fieldStrength,
    0,
    1.5,
  );
  const cohesion = clamp(
    Number.isFinite(input.cohesion) ? input.cohesion : defaults.cohesion,
    0,
    1.2,
  );
  const repelImpulse = clamp(
    Number.isFinite(input.repelImpulse) ? input.repelImpulse : defaults.repelImpulse,
    0,
    1,
  );
  const vortexAmount = clamp(
    Number.isFinite(input.vortexAmount) ? input.vortexAmount : defaults.vortexAmount,
    0,
    1,
  );

  return {
    spawnRate,
    fieldStrength,
    cohesion,
    cohesionForce: 0.14 + cohesion * 1.25,
    repelImpulse,
    vortexAmount,
    vortexStrength: vortexAmount * 1.45,
    flowStrength: 0.18 + fieldStrength * 1.85,
  };
}

function updateWells(params, dt) {
  const { vortexAmount, fieldStrength } = params;
  const time = state.time;
  const radiusBase = 0.18 + vortexAmount * 0.32;
  const radius = radiusBase * Math.min(state.bounds.width, state.bounds.height);
  const speed = 0.25 + vortexAmount * 1.2;
  const centerStrength = 0.6 + fieldStrength * 1.95;
  const satelliteStrength = 0.32 + fieldStrength * 1.1;
  const angle = time * speed;

  state.wells[0].x = 0;
  state.wells[0].y = 0;
  state.wells[0].strength = centerStrength;

  const offsetX = Math.cos(angle) * radius * 0.5;
  const offsetY = Math.sin(angle) * radius * 0.5;
  state.wells[1].x = offsetX;
  state.wells[1].y = offsetY;
  state.wells[1].strength = satelliteStrength;

  state.wells[2].x = -offsetX;
  state.wells[2].y = -offsetY;
  state.wells[2].strength = satelliteStrength;

  // Subtly spin wells to keep motion lively.
  state.wells[1].x += Math.sin(time * 0.73) * dt * radius * 0.12;
  state.wells[1].y += Math.cos(time * 0.53) * dt * radius * 0.12;
  state.wells[2].x += Math.sin(time * 0.63 + Math.PI) * dt * radius * 0.12;
  state.wells[2].y += Math.cos(time * 0.49 + Math.PI) * dt * radius * 0.12;
}

function wrapValue(value, min, max) {
  const span = max - min;
  if (span <= 0) {
    return min;
  }
  if (value < min) {
    const delta = min - value;
    const n = Math.ceil(delta / span);
    return value + span * n;
  }
  if (value > max) {
    const delta = value - max;
    const n = Math.ceil(delta / span);
    return value - span * n;
  }
  return value;
}

function applyBounds(x, y, vx, vy) {
  const { minX, maxX, minY, maxY, mode } = state.bounds;
  const out = BOUNDS_SCRATCH;

  if (mode === 'wrap') {
    out.x = wrapValue(x, minX, maxX);
    out.y = wrapValue(y, minY, maxY);
    out.vx = vx;
    out.vy = vy;
    return out;
  }

  let nextX = x;
  let nextY = y;
  let nextVx = vx;
  let nextVy = vy;

  if (nextX < minX) {
    nextX = minX + (minX - nextX);
    nextVx = Math.abs(nextVx);
  } else if (nextX > maxX) {
    nextX = maxX - (nextX - maxX);
    nextVx = -Math.abs(nextVx);
  }

  if (nextY < minY) {
    nextY = minY + (minY - nextY);
    nextVy = Math.abs(nextVy);
  } else if (nextY > maxY) {
    nextY = maxY - (nextY - maxY);
    nextVy = -Math.abs(nextVy);
  }

  out.x = nextX;
  out.y = nextY;
  out.vx = nextVx;
  out.vy = nextVy;
  return out;
}

function randomUnit() {
  state.rngState = (state.rngState * 1664525 + 1013904223) >>> 0;
  return state.rngState / 0xffffffff;
}

function randomRange(min, max) {
  return lerp(min, max, randomUnit());
}

function allocateIndex() {
  if (state.freeTop <= 0) {
    return -1;
  }
  state.freeTop -= 1;
  const index = state.freeList[state.freeTop];
  return typeof index === 'number' ? index : -1;
}

function addActiveIndex(index) {
  state.liveMap[index] = state.liveCount;
  state.liveList[state.liveCount] = index;
  state.liveCount += 1;
  state.activeCount += 1;
}

function releaseIndex(index) {
  if (!state.alive || !state.freeList || !state.liveMap || !state.liveList) {
    return;
  }

  if (!state.alive[index]) {
    return;
  }

  state.alive[index] = 0;
  state.life[index] = 0;
  state.maxLife[index] = 0;
  state.mass[index] = 0;

  const pos = state.liveMap[index];
  if (pos >= 0) {
    const last = state.liveCount - 1;
    if (last >= 0) {
      const swapIndex = state.liveList[last];
      state.liveList[pos] = swapIndex;
      state.liveMap[swapIndex] = pos;
    }
    state.liveCount = Math.max(0, state.liveCount - 1);
    state.liveMap[index] = -1;
  }

  state.freeList[state.freeTop] = index;
  state.freeTop += 1;
  state.activeCount = Math.max(0, state.activeCount - 1);
}

function spawnParticle() {
  const index = allocateIndex();
  if (index < 0 || !state.posX || !state.posY || !state.velX || !state.velY || !state.mass || !state.seed || !state.alive) {
    return false;
  }

  const radius = Math.min(state.bounds.width, state.bounds.height) * 0.08;
  const angle = randomUnit() * Math.PI * 2;
  const spread = Math.sqrt(randomUnit());
  const offsetX = Math.cos(angle) * radius * spread;
  const offsetY = Math.sin(angle) * radius * spread;

  state.posX[index] = offsetX;
  state.posY[index] = offsetY;
  state.velX[index] = randomRange(-0.35, 0.35);
  state.velY[index] = randomRange(-0.35, 0.35);
  state.life[index] = 0;
  state.maxLife[index] = randomRange(LIFE_MIN, LIFE_MAX);
  state.mass[index] = randomRange(0.75, 1.2);
  state.seed[index] = randomRange(-Math.PI, Math.PI);
  state.alive[index] = 1;

  addActiveIndex(index);
  return true;
}

function spawnParticles(count, params) {
  const limit = Math.max(0, Math.min(count, state.dynamicCap - state.activeCount));
  let spawned = 0;
  for (let i = 0; i < limit; i++) {
    if (spawnParticle(params)) {
      spawned += 1;
    } else {
      break;
    }
  }
  return spawned;
}

function enqueueKill(index) {
  state.killQueue[state.killCount] = index;
  state.killCount += 1;
}

function flushKillQueue() {
  for (let i = 0; i < state.killCount; i++) {
    releaseIndex(state.killQueue[i]);
  }
  state.killCount = 0;
}

function computeCenterOfMass() {
  if (!state.liveList || !state.mass) {
    state.centerX = 0;
    state.centerY = 0;
    return;
  }

  let sumX = 0;
  let sumY = 0;
  let total = 0;

  for (let i = 0; i < state.liveCount; i++) {
    const index = state.liveList[i];
    const m = state.mass[index] || 1;
    sumX += state.posX[index] * m;
    sumY += state.posY[index] * m;
    total += m;
  }

  if (total > EPSILON) {
    state.centerX = sumX / total;
    state.centerY = sumY / total;
  } else {
    state.centerX = 0;
    state.centerY = 0;
  }
}

function sampleFlowAngle(x, y, time, seed) {
  const t = time * FLOW_TIME_SCALE;
  const value =
    Math.sin(x * FLOW_FREQ_X + t + seed) +
    Math.cos(y * FLOW_FREQ_Y - t * 1.37 + seed * 1.3);
  return value * 1.42;
}

function integrateParticles(dt, params) {
  if (!state.liveList) {
    return;
  }

  const dragFactor = Math.exp(-state.drag * dt);
  const repel = state.repelStrength;
  const flowStrength = params.flowStrength;
  const vortexStrength = params.vortexStrength;
  const cohesionForce = params.cohesionForce;

  for (let i = 0; i < state.liveCount; i++) {
    const index = state.liveList[i];
    const mass = state.mass[index] || 1;

    let x = state.posX[index];
    let y = state.posY[index];
    let vx = state.velX[index];
    let vy = state.velY[index];

    state.life[index] += dt;
    if (state.life[index] >= state.maxLife[index]) {
      enqueueKill(index);
      continue;
    }

    let fx = 0;
    let fy = 0;

    // Gravity wells
    for (let w = 0; w < state.wells.length; w++) {
      const well = state.wells[w];
      const dx = well.x - x;
      const dy = well.y - y;
      const distSq = dx * dx + dy * dy + 0.0008;
      const invDist = 1 / Math.sqrt(distSq);
      const strength = well.strength * mass;
      fx += dx * invDist * strength * invDist;
      fy += dy * invDist * strength * invDist;
    }

    // Cohesion spring pulling toward center of mass.
    const dxCenter = state.centerX - x;
    const dyCenter = state.centerY - y;
    fx += dxCenter * cohesionForce;
    fy += dyCenter * cohesionForce;

    // Repeller impulse pushes outward from center.
    if (repel > EPSILON) {
      const rx = x - state.centerX;
      const ry = y - state.centerY;
      const distSq = rx * rx + ry * ry + 0.0004;
      const invDist = 1 / Math.sqrt(distSq);
      const strength = repel * invDist * 1.4;
      fx += rx * invDist * strength;
      fy += ry * invDist * strength;
    }

    // Flow field / noise-driven drift.
    const angle = sampleFlowAngle(x, y, state.time, state.seed[index]);
    fx += Math.cos(angle) * flowStrength;
    fy += Math.sin(angle) * flowStrength;

    // Vortex swirl provides perpendicular acceleration.
    if (vortexStrength > EPSILON) {
      fx += -dyCenter * vortexStrength;
      fy += dxCenter * vortexStrength;
    }

    vx += (fx / mass) * dt;
    vy += (fy / mass) * dt;

    vx *= dragFactor;
    vy *= dragFactor;

    x += vx * dt;
    y += vy * dt;

    const bounded = applyBounds(x, y, vx, vy);
    state.posX[index] = bounded.x;
    state.posY[index] = bounded.y;
    state.velX[index] = bounded.vx;
    state.velY[index] = bounded.vy;
  }
}

function updateDynamicCap(frameTimeInstant, frameTimeAverage) {
  const previousCap = state.dynamicCap;
  const instant = Number.isFinite(frameTimeInstant) && frameTimeInstant > 0 ? frameTimeInstant : Number.NaN;
  const rolling = Number.isFinite(frameTimeAverage) && frameTimeAverage > 0 ? frameTimeAverage : Number.NaN;
  const overloadCheck = Number.isFinite(instant)
    ? (Number.isFinite(rolling) ? Math.max(instant, rolling) : instant)
    : rolling;
  const recoveryCheck = Number.isFinite(rolling)
    ? (Number.isFinite(instant) ? Math.min(instant, rolling) : rolling)
    : instant;

  if (!Number.isFinite(overloadCheck) || overloadCheck <= 0) {
    return;
  }

  const representative = Number.isFinite(rolling) ? rolling : overloadCheck;
  state.metrics.frameTime = representative;
  state.metrics.frameTimeInstant = Number.isFinite(instant) ? instant : representative;
  state.metrics.fps = representative > 0 ? 1000 / representative : 60;
  state.metrics.trimmedLastFrame = false;

  if (overloadCheck > OVERLOAD_THRESHOLD_MS) {
    state.metrics.overloadedFrames = Math.min(120, state.metrics.overloadedFrames + 1);
    const next = Math.max(
      state.minCap,
      Math.floor(state.dynamicCap * (overloadCheck > TARGET_FRAME_MS * 2 ? 0.82 : 0.92)),
    );
    if (next < state.dynamicCap) {
      state.dynamicCap = next;
      state.metrics.trimmedLastFrame = true;
    }
  } else if (Number.isFinite(recoveryCheck) && recoveryCheck < RECOVERY_THRESHOLD_MS) {
    state.metrics.overloadedFrames = Math.max(0, state.metrics.overloadedFrames - 2);
    if (state.dynamicCap < state.baseCap) {
      state.dynamicCap = Math.min(
        state.baseCap,
        Math.ceil(state.dynamicCap + Math.max(16, state.baseCap * 0.02)),
      );
    }
  } else {
    state.metrics.overloadedFrames = Math.max(0, state.metrics.overloadedFrames - 1);
  }

  state.dynamicCap = clamp(Math.floor(state.dynamicCap), state.minCap, state.capacity);

  if (state.dynamicCap !== previousCap) {
    console.info('[physics] dynamicCap', state.dynamicCap);
  }
}

function trimToDynamicCap() {
  const excess = state.activeCount - state.dynamicCap;
  if (excess <= 0) {
    return;
  }

  let trimmed = 0;
  for (let i = state.liveCount - 1; i >= 0 && trimmed < excess; i--) {
    const index = state.liveList[i];
    enqueueKill(index);
    trimmed += 1;
  }
}

function updateRepelStrength(target, dt) {
  const decay = Math.exp(-dt * 6.5);
  state.repelStrength = state.repelStrength * decay + target * (1 - decay);
}

/**
 * Advances the simulation.
 * @param {PhysicsParams} params
 * @param {StepOptions} options
 * @returns {{count: number, dynamicCap: number, spawned: number}}
 */
export function step(params = {}, options = {}) {
  const dt = sanitizeDt(options.dt);
  const frameTimeInstant = Number.isFinite(options.frameTime) ? options.frameTime : dt * 1000;
  const frameTimeAverage = Number.isFinite(options.frameTimeAvg) ? options.frameTimeAvg : frameTimeInstant;
  const sanitized = sanitizeParams(params);

  updateDynamicCap(frameTimeInstant, frameTimeAverage);
  computeCenterOfMass();
  updateRepelStrength(sanitized.repelImpulse, dt);
  updateWells(sanitized, dt);

  state.spawnAccumulator += lerp(SPAWN_RATE_MIN, SPAWN_RATE_MAX, clamp(sanitized.spawnRate, 0, 1)) * dt;
  let requestedSpawns = Math.floor(state.spawnAccumulator);
  state.spawnAccumulator -= requestedSpawns;

  if (requestedSpawns > 0) {
    requestedSpawns = Math.min(
      requestedSpawns,
      state.dynamicCap - state.activeCount,
      state.freeTop,
    );
  }

  const spawned = requestedSpawns > 0 ? spawnParticles(requestedSpawns, sanitized) : 0;

  state.time += dt;

  integrateParticles(dt, sanitized);
  trimToDynamicCap();
  flushKillQueue();

  return {
    count: state.activeCount,
    dynamicCap: state.dynamicCap,
    spawned,
  };
}

/**
 * Resets internal state without altering configuration.
 */
export function reset() {
  resetPools();
  state.spawnAccumulator = 0;
  state.time = 0;
  state.centerX = 0;
  state.centerY = 0;
  state.repelStrength = 0;
  state.metrics.frameTime = TARGET_FRAME_MS;
  state.metrics.frameTimeInstant = TARGET_FRAME_MS;
  state.metrics.fps = 60;
  state.metrics.trimmedLastFrame = false;
  state.metrics.overloadedFrames = 0;
  state.dynamicCap = Math.min(state.baseCap, state.capacity);
}

/**
 * Configures bounds, capacity, and base caps.
 * @param {{capacity?: number, baseCap?: number, minCap?: number, drag?: number, bounds?: {width?: number, height?: number, mode?: 'wrap' | 'reflect'}, defaults?: Partial<PhysicsParams>, seed?: number}} [options]
 */
export function configure(options = {}) {
  if (Number.isInteger(options.capacity) && options.capacity > 0 && options.capacity !== state.capacity) {
    ensurePool(options.capacity);
    reset();
  } else {
    ensurePool(state.capacity || DEFAULT_CAPACITY);
  }

  if (Number.isFinite(options.baseCap) && options.baseCap > 0) {
    state.baseCap = clamp(Math.floor(options.baseCap), MIN_DYNAMIC_CAP, options.capacity || state.capacity);
  }
  if (Number.isFinite(options.minCap) && options.minCap > 0) {
    state.minCap = clamp(Math.floor(options.minCap), 1, state.baseCap);
  }
  state.dynamicCap = clamp(state.dynamicCap, state.minCap, state.baseCap);

  if (Number.isFinite(options.drag) && options.drag >= 0) {
    state.drag = options.drag;
  }

  if (options.bounds) {
    const nextBounds = toBounds(
      Number.isFinite(options.bounds.width) && options.bounds.width > 0
        ? options.bounds.width
        : state.bounds.width,
      Number.isFinite(options.bounds.height) && options.bounds.height > 0
        ? options.bounds.height
        : state.bounds.height,
    );
    state.bounds.minX = nextBounds.minX;
    state.bounds.maxX = nextBounds.maxX;
    state.bounds.minY = nextBounds.minY;
    state.bounds.maxY = nextBounds.maxY;
    state.bounds.width = nextBounds.width;
    state.bounds.height = nextBounds.height;
    if (options.bounds.mode === 'wrap' || options.bounds.mode === 'reflect') {
      state.bounds.mode = options.bounds.mode;
    }
  }

  if (options.defaults) {
    if (Number.isFinite(options.defaults.spawnRate)) {
      state.defaults.spawnRate = clamp(options.defaults.spawnRate, 0, 1.5);
    }
    if (Number.isFinite(options.defaults.fieldStrength)) {
      state.defaults.fieldStrength = clamp(options.defaults.fieldStrength, 0, 1.5);
    }
    if (Number.isFinite(options.defaults.cohesion)) {
      state.defaults.cohesion = clamp(options.defaults.cohesion, 0, 1.2);
    }
    if (Number.isFinite(options.defaults.repelImpulse)) {
      state.defaults.repelImpulse = clamp(options.defaults.repelImpulse, 0, 1);
    }
    if (Number.isFinite(options.defaults.vortexAmount)) {
      state.defaults.vortexAmount = clamp(options.defaults.vortexAmount, 0, 1);
    }
  }

  if (Number.isFinite(options.seed)) {
    state.rngState = options.seed >>> 0;
  }
}

/**
 * Returns read-only views into the particle buffers.
 */
export function getParticles() {
  return {
    count: state.activeCount,
    dynamicCap: state.dynamicCap,
    capacity: state.capacity,
    positions: {
      x: state.posX,
      y: state.posY,
    },
    velocities: {
      x: state.velX,
      y: state.velY,
    },
    life: state.life,
    maxLife: state.maxLife,
    masses: state.mass,
    alive: state.alive,
    indices: state.liveList,
  };
}

/**
 * Returns timing and cap metrics that callers can expose in HUD/diagnostics.
 */
export function getMetrics() {
  return {
    frameTime: state.metrics.frameTime,
    frameTimeInstant: state.metrics.frameTimeInstant,
    fps: state.metrics.fps,
    trimmedLastFrame: state.metrics.trimmedLastFrame,
    overloadedFrames: state.metrics.overloadedFrames,
    dynamicCap: state.dynamicCap,
    count: state.activeCount,
  };
}

// Bootstrap default pool and state.
ensurePool(DEFAULT_CAPACITY);
reset();

export default {
  step,
  reset,
  configure,
  getParticles,
  getMetrics,
};
