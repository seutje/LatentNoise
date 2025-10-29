import { jest } from '@jest/globals';

import { destroy, init, setPalette } from '../render.js';

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
        drawImage: jest.fn(),
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
      <div class="hud-feedback">
        <div id="feedback-status" data-state="offline">
          <span id="feedback-status-indicator"></span>
          <span id="feedback-status-text"></span>
        </div>
        <div>
          <button id="feedback-positive" aria-pressed="false"></button>
          <button id="feedback-negative" aria-pressed="false"></button>
        </div>
      </div>
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
  expect(root.style.getPropertyValue('--accent-strong')).toBe('rgba(171, 205, 239, 0.88)');
  expect(root.style.getPropertyValue('--accent-dim')).toBe('rgba(171, 205, 239, 0.52)');
  expect(root.style.getPropertyValue('--accent-soft')).toBe('rgba(171, 205, 239, 0.18)');
  expect(root.style.getPropertyValue('--accent-faint')).toBe('rgba(171, 205, 239, 0.08)');
  expect(root.style.getPropertyValue('--hud-grid')).toMatch(/^rgba\(/);
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
  expect(root.style.getPropertyValue('--accent-strong')).toBe('rgba(255, 170, 0, 0.88)');
});
