import { jest } from '@jest/globals';
import * as videoExport from '../video-export.js';

describe('video-export', () => {
  let canvas;
  let button;
  let link;
  let notify;
  let render;
  let worker;
  let bitmap;

  class MockWorker {
    constructor() {
      this.messages = [];
      this.listeners = new Map();
      this.terminated = false;
    }

    postMessage(data) {
      this.messages.push(data);
    }

    addEventListener(type, handler) {
      this.listeners.set(type, handler);
    }

    terminate() {
      this.terminated = true;
    }

    dispatchMessage(payload) {
      const handler = this.listeners.get('message');
      if (handler) {
        handler({ data: payload });
      }
    }
  }

  function createRenderMock() {
    const listeners = new Map();
    return {
      on(event, handler) {
        listeners.set(event, handler);
        return () => listeners.delete(event);
      },
      emit(event, detail) {
        const handler = listeners.get(event);
        if (handler) {
          handler(detail);
        }
      },
    };
  }

  beforeEach(() => {
    document.body.innerHTML = `
      <canvas id="c"></canvas>
      <button id="export-video"></button>
      <a id="export-video-download"></a>
    `;
    canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('c'));
    button = /** @type {HTMLButtonElement} */ (document.getElementById('export-video'));
    link = /** @type {HTMLAnchorElement} */ (document.getElementById('export-video-download'));
    canvas.width = 640;
    canvas.height = 360;
    render = createRenderMock();
    notify = jest.fn();
    bitmap = { close: jest.fn() };
    global.URL.createObjectURL = jest.fn(() => 'blob:test');
    global.URL.revokeObjectURL = jest.fn();
    global.createImageBitmap = jest.fn().mockResolvedValue(bitmap);
    worker = new MockWorker();
    videoExport.__resetForTests();
    videoExport.configure({ createWorker: () => worker, frameRate: 30, keyframeInterval: 30 });
  });

  afterEach(() => {
    videoExport.__resetForTests();
    delete global.createImageBitmap;
  });

  it('initializes, records frames, and finalizes an export', async () => {
    const initialized = videoExport.init({ canvas, button, downloadLink: link, render, notify });
    expect(initialized).toBe(true);
    expect(button.disabled).toBe(false);

    button.click();
    expect(worker.messages).toContainEqual(expect.objectContaining({ type: 'start', width: 640, height: 360 }));

    worker.dispatchMessage({ type: 'started' });
    render.emit('frame', { timestamp: 0 });
    await Promise.resolve();
    await Promise.resolve();
    expect(global.createImageBitmap).toHaveBeenCalledWith(canvas);
    expect(worker.messages.some((message) => message.type === 'frame')).toBe(true);

    button.click();
    expect(worker.messages.some((message) => message.type === 'stop')).toBe(true);

    const buffer = new ArrayBuffer(8);
    worker.dispatchMessage({ type: 'done', buffer });
    expect(worker.terminated).toBe(true);
    expect(notify).toHaveBeenCalledWith('Video export ready. Download will begin shortly.');
    expect(link.hidden).toBe(false);
    expect(global.URL.createObjectURL).toHaveBeenCalled();
  });

  it('disables export control when capture APIs are unavailable', () => {
    delete global.createImageBitmap;
    const initialized = videoExport.init({ canvas, button, downloadLink: link, render, notify });
    expect(initialized).toBe(false);
    expect(button.disabled).toBe(true);
    expect(button.textContent).toBe('Export Unavailable');
  });
});
