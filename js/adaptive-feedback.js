import { PARAM_NAMES, getParamSpec } from './map.js';

const DEFAULT_BUFFER_SIZE = 720; // ~12s at 60 FPS
const DEFAULT_LATENCY_MS = 180;
const MAX_FRAME_AGE_MS = 12_000;
const TARGET_STEP_FRACTION = 0.18;
const OUTPUT_STEP = 0.12;
const OUTPUT_MIN = -1.25;
const OUTPUT_MAX = 1.25;
const BATCH_SIZE = 8;
const BATCH_INTERVAL_MS = 750;

const frameBuffer = new Array(DEFAULT_BUFFER_SIZE);
let writeIndex = 0;
let frameCount = 0;
let defaultLatencyMs = DEFAULT_LATENCY_MS;

const listeners = new Map();
const pendingBatch = [];
let lastBatchEmittedAt = 0;

function clamp(value, min, max) {
  if (value < min) {
    return min;
  }
  if (value > max) {
    return max;
  }
  return value;
}

function toFloat32Array(source) {
  if (source instanceof Float32Array) {
    return new Float32Array(source);
  }
  if (Array.isArray(source)) {
    return Float32Array.from(source);
  }
  if (typeof source === 'number') {
    return Float32Array.of(Number.isFinite(source) ? source : 0);
  }
  return new Float32Array();
}

function copyMappedParams(source) {
  const snapshot = {};
  if (!source || typeof source !== 'object') {
    return snapshot;
  }
  for (const [key, value] of Object.entries(source)) {
    if (typeof value === 'number') {
      snapshot[key] = Number.isFinite(value) ? value : 0;
    }
  }
  return snapshot;
}

function emit(eventName, detail) {
  const handlers = listeners.get(eventName);
  if (!handlers) {
    return;
  }
  for (const handler of handlers) {
    try {
      handler(detail);
    } catch (error) {
      console.warn('[adaptive-feedback] listener error', error);
    }
  }
}

function getNow() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function storeFrame(frame) {
  frameBuffer[writeIndex] = frame;
  writeIndex = (writeIndex + 1) % frameBuffer.length;
  if (frameCount < frameBuffer.length) {
    frameCount += 1;
  }
}

function pruneStaleFrames(cutoff) {
  if (frameCount === 0) {
    return;
  }
  let removed = 0;
  for (let i = 0; i < frameCount; i += 1) {
    const index = (writeIndex - frameCount + i + frameBuffer.length) % frameBuffer.length;
    const frame = frameBuffer[index];
    if (!frame || !Number.isFinite(frame.timestamp)) {
      continue;
    }
    if (frame.timestamp < cutoff) {
      frameBuffer[index] = null;
      removed += 1;
    }
  }
  if (removed === 0) {
    return;
  }
  const compacted = [];
  for (let i = 0; i < frameBuffer.length; i += 1) {
    const frame = frameBuffer[i];
    if (frame) {
      compacted.push(frame);
    }
  }
  compacted.length = Math.min(compacted.length, frameBuffer.length);
  frameBuffer.fill(null);
  writeIndex = 0;
  frameCount = 0;
  for (const frame of compacted) {
    storeFrame(frame);
  }
}

function findFrame(targetTimestamp) {
  if (frameCount === 0) {
    return null;
  }
  let bestFrame = null;
  let bestDistance = Infinity;
  for (let i = 0; i < frameCount; i += 1) {
    const index = (writeIndex - 1 - i + frameBuffer.length) % frameBuffer.length;
    const frame = frameBuffer[index];
    if (!frame) {
      continue;
    }
    const distance = Math.abs(frame.timestamp - targetTimestamp);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestFrame = frame;
      if (distance === 0) {
        break;
      }
    }
    if (frame.timestamp < targetTimestamp && bestFrame) {
      break;
    }
  }
  return bestFrame;
}

function computeTargetParams(mappedParams, direction) {
  const targets = {};
  const polarity = direction >= 0 ? 1 : -1;
  for (const name of PARAM_NAMES) {
    const spec = getParamSpec(name) || {};
    const current = Number.isFinite(mappedParams?.[name]) ? mappedParams[name] : 0;
    const swing = Number.isFinite(spec.safeSwing)
      ? spec.safeSwing
      : Number.isFinite(spec.swing)
        ? spec.swing
        : 0;
    const min = Number.isFinite(spec.min) ? spec.min : Number.isFinite(spec.safeMin) ? spec.safeMin : current - swing;
    const max = Number.isFinite(spec.safeMax)
      ? spec.safeMax
      : Number.isFinite(spec.max)
        ? spec.max
        : current + swing;
    const delta = swing > 0 ? swing * TARGET_STEP_FRACTION * polarity : polarity * 0.1;
    targets[name] = clamp(current + delta, min, max);
  }
  if (mappedParams) {
    if (typeof mappedParams.spawnOffset === 'number') {
      targets.spawnOffset = mappedParams.spawnOffset;
    }
    if (typeof mappedParams.glowOffset === 'number') {
      targets.glowOffset = mappedParams.glowOffset;
    }
    if (typeof mappedParams.sparkleOffset === 'number') {
      targets.sparkleOffset = mappedParams.sparkleOffset;
    }
    if (typeof mappedParams.hueOffset === 'number') {
      targets.hueOffset = mappedParams.hueOffset;
    }
    if (typeof mappedParams.repelImpulse === 'number') {
      targets.repelImpulse = mappedParams.repelImpulse;
    }
  }
  return targets;
}

function computeTargetOutputs(outputs, direction) {
  const polarity = direction >= 0 ? 1 : -1;
  const next = new Float32Array(outputs.length);
  const delta = OUTPUT_STEP * polarity;
  for (let i = 0; i < outputs.length; i += 1) {
    const value = Number.isFinite(outputs[i]) ? outputs[i] : 0;
    next[i] = clamp(value + delta, OUTPUT_MIN, OUTPUT_MAX);
  }
  return next;
}

function enqueueFeedback(sample) {
  pendingBatch.push(sample);
  emit('feedback', sample);
  const now = getNow();
  const shouldFlush =
    pendingBatch.length >= BATCH_SIZE || (now - lastBatchEmittedAt >= BATCH_INTERVAL_MS && pendingBatch.length > 0);
  if (shouldFlush) {
    const batch = pendingBatch.splice(0);
    lastBatchEmittedAt = now;
    emit('batch', {
      timestamp: now,
      samples: batch,
    });
    if (typeof window !== 'undefined' && typeof window.dispatchEvent === 'function') {
      window.dispatchEvent(
        new CustomEvent('ln:feedback-batch', {
          detail: {
            timestamp: now,
            samples: batch.map((entry) => ({ ...entry })),
          },
        }),
      );
    }
  }
}

export function on(eventName, handler) {
  if (typeof handler !== 'function') {
    return () => {};
  }
  if (!listeners.has(eventName)) {
    listeners.set(eventName, new Set());
  }
  const handlers = listeners.get(eventName);
  handlers.add(handler);
  return () => {
    handlers.delete(handler);
    if (handlers.size === 0) {
      listeners.delete(eventName);
    }
  };
}

export function off(eventName, handler) {
  const handlers = listeners.get(eventName);
  if (!handlers) {
    return;
  }
  handlers.delete(handler);
  if (handlers.size === 0) {
    listeners.delete(eventName);
  }
}

export function clear() {
  frameBuffer.fill(null);
  writeIndex = 0;
  frameCount = 0;
  pendingBatch.length = 0;
}

export function setDefaultLatency(latencyMs) {
  if (!Number.isFinite(latencyMs) || latencyMs < 0) {
    return;
  }
  defaultLatencyMs = latencyMs;
}

export function getDefaultLatency() {
  return defaultLatencyMs;
}

export function recordFrame(data) {
  if (!data || typeof data.timestamp !== 'number') {
    return;
  }
  const frame = {
    timestamp: data.timestamp,
    features: toFloat32Array(data.features),
    outputs: toFloat32Array(data.outputs),
    mappedParams: copyMappedParams(data.mappedParams),
  };
  storeFrame(frame);
  pruneStaleFrames(data.timestamp - MAX_FRAME_AGE_MS);
  emit('frame', frame);
}

export function recordFeedback(score, options = {}) {
  if (!Number.isFinite(score) || score === 0) {
    return null;
  }
  const now = getNow();
  const sourceTimestamp = Number.isFinite(options.timestamp) ? options.timestamp : now;
  const latency = Number.isFinite(options.latencyMs) ? options.latencyMs : defaultLatencyMs;
  const targetTimestamp = sourceTimestamp - Math.max(latency, 0);
  const frame = findFrame(targetTimestamp);
  if (!frame) {
    return null;
  }
  const direction = score >= 0 ? 1 : -1;
  const sample = {
    score,
    direction,
    timestamp: sourceTimestamp,
    latencyMs: latency,
    frameTimestamp: frame.timestamp,
    targetTimestamp,
    features: toFloat32Array(frame.features),
    outputs: toFloat32Array(frame.outputs),
    mappedParams: copyMappedParams(frame.mappedParams),
    targetOutputs: computeTargetOutputs(frame.outputs, direction),
    targetParams: computeTargetParams(frame.mappedParams, direction),
    metadata: options.metadata ? { ...options.metadata } : {},
  };
  enqueueFeedback(sample);
  if (typeof window !== 'undefined' && typeof window.dispatchEvent === 'function') {
    window.dispatchEvent(
      new CustomEvent('ln:feedback-event', {
        detail: { ...sample },
      }),
    );
  }
  return sample;
}
