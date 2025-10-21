import * as map from './map.js';
import * as nn from './nn.js';

const diagnosticsState = {
  debugEnabled: false,
  overlay: null,
  sections: {
    metrics: null,
    features: null,
    outputs: null,
    params: null,
    tests: null,
  },
  testsRan: false,
  testResults: [],
};

const OUTPUT_LABELS = map.PARAM_NAMES.slice();

function formatNumber(value, fractionDigits = 3) {
  if (!Number.isFinite(value)) {
    return '—';
  }
  if (Math.abs(value) >= 1000) {
    return value.toFixed(0);
  }
  return value.toFixed(fractionDigits);
}

function formatArray(values, labels = []) {
  if (!values || typeof values.length !== 'number') {
    return 'No data';
  }
  const lines = [];
  const length = values.length;
  for (let i = 0; i < length; i += 1) {
    const label = labels[i] ?? `#${String(i).padStart(2, '0')}`;
    const rawValue = Number(values[i]);
    const formatted = Number.isFinite(rawValue) ? formatNumber(rawValue, 4) : '—';
    lines.push(`${label.padEnd(16)} ${formatted}`);
  }
  return lines.join('\n');
}

function formatParams(params) {
  if (!params || typeof params !== 'object') {
    return 'No parameters';
  }
  const entries = Object.entries(params);
  if (entries.length === 0) {
    return 'No parameters';
  }
  const sorted = entries.sort(([a], [b]) => a.localeCompare(b));
  return sorted
    .map(([name, value]) => `${name.padEnd(16)} ${formatNumber(Number(value), 4)}`)
    .join('\n');
}

function createSection(title) {
  const container = document.createElement('section');
  container.className = 'debug-section';
  const heading = document.createElement('h2');
  heading.textContent = title;
  heading.className = 'debug-heading';
  const pre = document.createElement('pre');
  pre.className = 'debug-content';
  container.append(heading, pre);
  return { container, content: pre };
}

function ensureOverlay() {
  if (!diagnosticsState.debugEnabled || diagnosticsState.overlay || typeof document === 'undefined') {
    return diagnosticsState.overlay;
  }
  const root = document.createElement('div');
  root.id = 'debug-overlay';
  root.className = 'ui';

  const metrics = createSection('Metrics');
  const features = createSection('Features');
  const outputs = createSection('NN Outputs');
  const params = createSection('Parameters');
  const tests = createSection('Startup Tests');

  root.append(metrics.container, features.container, outputs.container, params.container, tests.container);
  document.body.append(root);

  diagnosticsState.sections.metrics = metrics.content;
  diagnosticsState.sections.features = features.content;
  diagnosticsState.sections.outputs = outputs.content;
  diagnosticsState.sections.params = params.content;
  diagnosticsState.sections.tests = tests.content;
  diagnosticsState.overlay = root;

  renderTestResults();
  return root;
}

function renderTestResults() {
  const target = diagnosticsState.sections.tests;
  if (!target) {
    return;
  }
  if (!diagnosticsState.testsRan) {
    target.textContent = 'Pending…';
    return;
  }
  if (diagnosticsState.testResults.length === 0) {
    target.textContent = 'No tests executed.';
    return;
  }
  const lines = diagnosticsState.testResults.map((result) => {
    const status = result.passed ? 'PASS' : 'FAIL';
    const duration = formatNumber(result.durationMs, 2);
    return `${status.padEnd(4)} ${result.name} (${duration} ms)`;
  });
  target.textContent = lines.join('\n');
}

function logTestResult(result) {
  const prefix = '[diag]';
  if (result.passed) {
    console.info(`${prefix} ${result.name} passed in ${formatNumber(result.durationMs, 2)} ms`);
  } else {
    console.error(`${prefix} ${result.name} failed`, result.error);
  }
}

async function runTestCase(testCase, context) {
  const start = typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
  try {
    await testCase.run(context);
    const end = typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
    return {
      name: testCase.name,
      passed: true,
      durationMs: end - start,
    };
  } catch (error) {
    const end = typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
    return {
      name: testCase.name,
      passed: false,
      durationMs: end - start,
      error,
    };
  }
}

function getDummyModel() {
  return {
    input: 3,
    normalization: {
      mean: [0, 0, 0],
      std: [1, 1, 1],
    },
    layers: [
      {
        activation: 'relu',
        weights: [
          0.75, -0.25, 0.5,
          -0.4, 0.9, 0.15,
        ],
        bias: [0.2, -0.1],
      },
      {
        activation: 'tanh',
        weights: [1.6, -0.7],
        bias: [0.05],
      },
    ],
  };
}

const startupTests = [
  {
    name: 'NN forward produces finite output',
    async run() {
      const model = getDummyModel();
      await nn.loadModel(model);
      const features = new Float32Array([0.5, -0.25, 1.2]);
      const normalized = nn.normalize(features);
      const outputs = nn.forward(normalized);
      for (let i = 0; i < outputs.length; i += 1) {
        if (!Number.isFinite(outputs[i])) {
          throw new Error(`Output index ${i} is not finite.`);
        }
      }
    },
  },
  {
    name: 'Map safe mode clamps outputs',
    async run(context) {
      const originalSafe = Boolean(context?.safeMode);
      map.configure({ safeMode: true });
      map.reset();
      const outputs = new Float32Array(OUTPUT_LABELS.length);
      outputs.fill(1);
      let params = {};
      for (let i = 0; i < 16; i += 1) {
        params = map.update(outputs, { dt: 1 / 30, activity: 1 });
      }
      const violations = [];
      for (const name of OUTPUT_LABELS) {
        const spec = map.getParamSpec(name);
        const value = params[name];
        if (!spec) {
          continue;
        }
        if (typeof spec.safeMax === 'number' && value > spec.safeMax + 1e-3) {
          violations.push(`${name}=${formatNumber(value, 4)} > ${spec.safeMax}`);
        }
        if (spec.symmetric && typeof spec.safeMax === 'number' && value < -spec.safeMax - 1e-3) {
          violations.push(`${name}=${formatNumber(value, 4)} < -${spec.safeMax}`);
        }
      }
      map.configure({ safeMode: originalSafe });
      map.reset();
      if (violations.length > 0) {
        throw new Error(`Safe bounds exceeded: ${violations.join(', ')}`);
      }
    },
  },
];

export function initDebugOverlay(options = {}) {
  if (diagnosticsState.debugEnabled) {
    return true;
  }
  if (typeof window === 'undefined') {
    return false;
  }
  const search = typeof options.search === 'string' ? options.search : window.location?.search ?? '';
  const params = new URLSearchParams(search);
  const enabled = options.enabled ?? (params.get('debug') === '1');
  diagnosticsState.debugEnabled = Boolean(enabled);
  if (diagnosticsState.debugEnabled) {
    ensureOverlay();
  }
  return diagnosticsState.debugEnabled;
}

export function updateDebugOverlay(payload = {}) {
  if (!diagnosticsState.debugEnabled) {
    return;
  }
  ensureOverlay();
  const { metrics, features, outputs, params } = diagnosticsState.sections;
  if (metrics) {
    const lines = [];
    if (Number.isFinite(payload.fps)) {
      lines.push(`FPS (inst)      ${formatNumber(payload.fps, 2)}`);
    }
    if (Number.isFinite(payload.fpsAvg)) {
      lines.push(`FPS (avg)       ${formatNumber(payload.fpsAvg, 2)}`);
    }
    if (Number.isFinite(payload.activity)) {
      lines.push(`Activity        ${formatNumber(payload.activity, 4)}`);
    }
    if (payload.modelInfo && typeof payload.modelInfo === 'object') {
      const { inputSize, outputSize, layers } = payload.modelInfo;
      lines.push(`Model           in:${inputSize ?? '—'} out:${outputSize ?? '—'} layers:${layers ?? '—'}`);
    }
    metrics.textContent = lines.length > 0 ? lines.join('\n') : 'No metrics';
  }
  if (features) {
    features.textContent = formatArray(payload.features);
  }
  if (outputs) {
    outputs.textContent = formatArray(payload.outputs, OUTPUT_LABELS);
  }
  if (params) {
    params.textContent = formatParams(payload.params);
  }
}

export async function runStartupDiagnostics(context = {}) {
  if (diagnosticsState.testsRan) {
    return diagnosticsState.testResults;
  }
  const results = [];
  for (const testCase of startupTests) {
    const result = await runTestCase(testCase, context);
    results.push(result);
    logTestResult(result);
  }
  diagnosticsState.testsRan = true;
  diagnosticsState.testResults = results;
  renderTestResults();
  return results;
}
