import { jest } from '@jest/globals';

import { destroy, getModelHudSnapshot, init, renderFrame, setPalette } from '../render.js';

beforeAll(() => {
  Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
    configurable: true,
    writable: true,
    value: function getContextStub() {
      return {
        canvas: this,
        setTransform: jest.fn(),
        fillRect: jest.fn(),
        clearRect: jest.fn(),
        save: jest.fn(),
        restore: jest.fn(),
        beginPath: jest.fn(),
        arc: jest.fn(),
        fill: jest.fn(),
        stroke: jest.fn(),
        moveTo: jest.fn(),
        lineTo: jest.fn(),
        quadraticCurveTo: jest.fn(),
        closePath: jest.fn(),
        drawImage: jest.fn(),
        fillText: jest.fn(),
        globalAlpha: 1,
        globalCompositeOperation: 'source-over',
        filter: 'none',
      };
    },
  });
});

beforeEach(() => {
  document.body.innerHTML = '';
  document.documentElement.style.cssText = '';
  setPalette();
});

afterEach(() => {
  destroy();
  setPalette();
  document.body.innerHTML = '';
  document.documentElement.style.cssText = '';
});

function setupRenderDom() {
  document.body.innerHTML = `
    <canvas id="c"></canvas>
    <div id="hud" class="ui">
      <div id="track-title"></div>
      <div id="track-time"></div>
      <div id="hud-status"></div>
      <div id="hud-fps"></div>
      <div class="hud-volume">
        <label>Volume <span id="volume-display"></span></label>
        <input id="volume" type="range" value="0.7" />
      </div>
    </div>
  `;
}

test('setPalette normalizes palette data and updates CSS variables without init', () => {
  const result = setPalette({
    background: '#123',
    accents: ['#ABCDEF', 'not-a-color', '#00ff99'],
    baseHue: 725,
  });

  expect(result).toEqual({
    background: '#112233',
    accents: ['#abcdef', '#00ff99'],
    baseHue: 5,
  });

  const root = document.documentElement;
  expect(root.style.getPropertyValue('--bg')).toBe('#112233');
  expect(root.style.getPropertyValue('--accent')).toBe('#abcdef');
});

test('setPalette applies gradient background after render init', () => {
  setupRenderDom();
  init();

  const updated = setPalette({
    background: '#0a0b1c',
    accents: ['#ffaa00', '#ffe6a0'],
    baseHue: 180,
  });

  expect(updated.baseHue).toBe(180);
  expect(updated.accents).toEqual(['#ffaa00', '#ffe6a0']);

  const canvas = document.getElementById('c');
  expect(canvas).not.toBeNull();
  if (!canvas) {
    return;
  }
  const backgroundStyle = canvas.style.background.toLowerCase();
  expect(backgroundStyle).toContain('#0a0b1c');

  const root = document.documentElement;
  expect(root.style.getPropertyValue('--accent')).toBe('#ffaa00');
});

test('renderFrame updates model HUD snapshot from audio features and outputs', () => {
  setupRenderDom();
  init();

  const particles = {
    positions: { x: new Float32Array(0), y: new Float32Array(0) },
    life: new Float32Array(0),
    maxLife: new Float32Array(0),
    masses: new Float32Array(0),
    alive: new Uint8Array(0),
    indices: new Uint32Array(0),
    count: 0,
  };

  const metricsWithData = {
    dt: 1 / 60,
    frameTime: 1000 / 60,
    frameTimeAvg: 1000 / 60,
    fps: 60,
    fpsAvg: 60,
    features: new Float32Array([0, 0.5, 1.2, 0.1]),
    hidden: new Float32Array([0.25, -0.75, 0.1]),
    outputs: new Float32Array([0.2, -0.6, 0.9]),
  };

  for (let i = 0; i < 4; i += 1) {
    renderFrame(particles, {}, metricsWithData);
  }

  const snapshot = getModelHudSnapshot();
  expect(snapshot.features).toHaveLength(4);
  expect(snapshot.hidden).toHaveLength(3);
  expect(snapshot.outputs).toHaveLength(3);
  snapshot.features.forEach((value) => {
    expect(value).toBeGreaterThanOrEqual(0);
    expect(value).toBeLessThanOrEqual(1);
  });
  snapshot.hidden.forEach((value) => {
    expect(value).toBeGreaterThanOrEqual(0);
    expect(value).toBeLessThanOrEqual(1);
  });
  const initialSum = snapshot.features.reduce((sum, value) => sum + value, 0);
  expect(snapshot.features[3]).toBeGreaterThan(snapshot.features[0]);
  expect(snapshot.outputs[2]).toBeGreaterThan(snapshot.outputs[0]);

  for (let i = 0; i < 6; i += 1) {
    renderFrame(particles, {}, {
      dt: 1 / 60,
      frameTime: 1000 / 60,
      frameTimeAvg: 1000 / 60,
      fps: 60,
      fpsAvg: 60,
    });
  }

  const decayed = getModelHudSnapshot();
  const decayedSum = decayed.features.reduce((sum, value) => sum + value, 0);
  expect(decayedSum).toBeLessThan(initialSum);
  const hiddenSum = decayed.hidden.reduce((sum, value) => sum + value, 0);
  expect(hiddenSum).toBeLessThan(snapshot.hidden.reduce((sum, value) => sum + value, 0));
});
