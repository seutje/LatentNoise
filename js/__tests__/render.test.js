import { jest } from '@jest/globals';

import { destroy, init, renderFrame, setPalette } from '../render.js';

beforeAll(() => {
  Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
    configurable: true,
    writable: true,
    value: function getContextStub() {
      if (this.__ctx) {
        return this.__ctx;
      }
      const ctx = {
        canvas: this,
        setTransform: jest.fn(),
        fillRect: jest.fn(),
        clearRect: jest.fn(),
        strokeRect: jest.fn(),
        save: jest.fn(),
        restore: jest.fn(),
        beginPath: jest.fn(),
        arc: jest.fn(),
        fill: jest.fn(),
        stroke: jest.fn(),
        moveTo: jest.fn(),
        lineTo: jest.fn(),
        drawImage: jest.fn(),
        fillText: jest.fn(),
        globalAlpha: 1,
        globalCompositeOperation: 'source-over',
        filter: 'none',
      };
      this.__ctx = ctx;
      return ctx;
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

test('renderFrame draws HUD vectors for features and outputs', () => {
  setupRenderDom();
  init();

  const canvas = document.getElementById('c');
  expect(canvas).not.toBeNull();
  if (!canvas) {
    return;
  }
  const ctx = canvas.getContext('2d');
  ctx.fillText.mockClear();

  const particles = {
    positions: { x: new Float32Array(0), y: new Float32Array(0) },
    life: new Float32Array(0),
    maxLife: new Float32Array(0),
    masses: new Float32Array(0),
    alive: new Uint8Array(0),
    indices: new Uint16Array(0),
    count: 0,
  };

  renderFrame(particles, {}, {
    dt: 1 / 60,
    frameTime: 16,
    frameTimeAvg: 16,
    fps: 60,
    fpsAvg: 60,
    features: new Float32Array([0.5, -0.4, 0.9, 0.1]),
    featureLabels: ['sub', 'bass', 'mid', 'high'],
    outputs: new Float32Array([0.25, -0.75, 0.05]),
    outputLabels: ['spawnRate', 'glow', 'sparkleDensity'],
  });

  const texts = ctx.fillText.mock.calls.map((call) => call[0]);
  expect(texts.some((text) => typeof text === 'string' && text.includes('INPUT VECTOR'))).toBe(true);
  expect(texts.some((text) => typeof text === 'string' && text.includes('OUTPUT VECTOR'))).toBe(true);
});
