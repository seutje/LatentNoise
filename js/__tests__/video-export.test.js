import { jest } from '@jest/globals';
import { createDownloadFileName, createVideoExporter } from '../video-export.js';

if (typeof Blob !== 'undefined' && typeof Blob.prototype.arrayBuffer !== 'function') {
  Object.defineProperty(Blob.prototype, 'arrayBuffer', {
    configurable: true,
    value: jest.fn(() => Promise.resolve(new Uint8Array([1, 2, 3]).buffer)),
  });
}

function createMockTrack(kind) {
  return {
    kind,
    stop: jest.fn(),
  };
}

class MockStream {
  constructor(tracks = []) {
    this._tracks = [...tracks];
  }

  addTrack(track) {
    this._tracks.push(track);
  }

  getTracks() {
    return [...this._tracks];
  }

  getVideoTracks() {
    return this._tracks.filter((track) => track.kind === 'video');
  }

  getAudioTracks() {
    return this._tracks.filter((track) => track.kind === 'audio');
  }
}

function createCanvasElement() {
  const canvas = document.createElement('canvas');
  canvas.captureStream = jest.fn(() => new MockStream([createMockTrack('video')]));
  return canvas;
}

function createAudioElement(hasCapture = true, paused = false) {
  const audio = document.createElement('audio');
  Object.defineProperty(audio, 'paused', {
    configurable: true,
    get: () => paused,
  });
  audio.play = jest.fn(() => Promise.resolve());
  if (hasCapture) {
    audio.captureStream = jest.fn(() => new MockStream([createMockTrack('audio')]));
  }
  return audio;
}

function createMockRecorderFactory({ blobType, onCreate }) {
  return jest.fn((stream, options = {}) => {
    if (typeof onCreate === 'function') {
      onCreate({ stream, options });
    }
    const events = {};
    return {
      events,
      state: 'inactive',
      start: jest.fn(function start() {
        this.state = 'recording';
      }),
      stop: jest.fn(function stop() {
        this.state = 'inactive';
        if (events.dataavailable) {
          const blob = new Blob(['chunk'], { type: blobType });
          Object.defineProperty(blob, 'arrayBuffer', {
            configurable: true,
            value: jest.fn(() => Promise.resolve(new Uint8Array([1, 2, 3]).buffer)),
          });
          events.dataavailable({ data: blob });
        }
        if (events.stop) {
          events.stop({});
        }
      }),
      addEventListener: jest.fn((type, handler) => {
        events[type] = handler;
      }),
      removeEventListener: jest.fn((type, handler) => {
        if (events[type] === handler) {
          delete events[type];
        }
      }),
    };
  });
}

describe('video-export utilities', () => {
  test('createDownloadFileName sanitizes titles and formats timestamp', () => {
    const name = createDownloadFileName('My Test Track!', new Date('2025-01-02T03:04:05Z'));
    expect(name).toBe('my-test-track-2025-01-02T03-04-05-000Z.mp4');
  });

  test('init disables button when support is unavailable', () => {
    const exporter = createVideoExporter({ createMediaRecorder: null, createWorker: null, createStream: null });
    const button = document.createElement('button');
    const result = exporter.init({
      canvas: document.createElement('canvas'),
      audio: document.createElement('audio'),
      button,
    });
    expect(result.isSupported).toBe(false);
    expect(button.disabled).toBe(true);
    expect(button.getAttribute('aria-disabled')).toBe('true');
  });

  test('records and downloads MP4 directly when supported', () => {
    const canvas = createCanvasElement();
    const audio = createAudioElement(true, true);
    const button = document.createElement('button');
    const downloadBlob = jest.fn();
    const recorderFactory = createMockRecorderFactory({ blobType: 'video/mp4' });
    const exporter = createVideoExporter({
      MediaRecorderClass: class {
        static isTypeSupported(type) {
          return type.includes('video/mp4');
        }
      },
      createMediaRecorder: recorderFactory,
      createStream: () => new MockStream(),
      createWorker: () => ({
        listeners: {},
        addEventListener: jest.fn(),
        postMessage: jest.fn(),
      }),
      downloadBlob,
    });

    exporter.init({
      canvas,
      audio,
      button,
      notify: jest.fn(),
      getFileName: () => 'Test Clip',
    });

    expect(exporter.start()).toBe(true);
    expect(recorderFactory).toHaveBeenCalledTimes(1);
    const [, recorderOptions] = recorderFactory.mock.calls[0];
    expect(recorderOptions).toMatchObject({
      mimeType: expect.stringContaining('video/mp4'),
    });
    expect(recorderOptions.videoBitsPerSecond).toBeGreaterThanOrEqual(6_000_000);
    expect(recorderOptions.audioBitsPerSecond).toBe(192_000);
    expect(recorderOptions.bitsPerSecond).toBe(recorderOptions.videoBitsPerSecond + recorderOptions.audioBitsPerSecond);
    expect(button.textContent).toBe('Stop Export');
    expect(exporter.getState().status).toBe('recording');

    expect(exporter.stop()).toBe(true);
    expect(downloadBlob).toHaveBeenCalledTimes(1);
    const [blob, filename] = downloadBlob.mock.calls[0];
    expect(blob).toBeInstanceOf(Blob);
    expect(blob.type).toContain('video/mp4');
    expect(filename).toMatch(/^test-clip-/);
    expect(exporter.getState().status).toBe('idle');
  });

  test('uses worker conversion when MP4 is not supported by MediaRecorder', async () => {
    const canvas = createCanvasElement();
    const audio = createAudioElement(true, true);
    const button = document.createElement('button');
    const downloadBlob = jest.fn();
    const recorderFactory = createMockRecorderFactory({ blobType: 'video/webm' });

    const worker = {
      listeners: {},
      addEventListener: jest.fn(function add(type, handler) {
        this.listeners[type] = handler;
      }),
      postMessage: jest.fn(),
    };

    const exporter = createVideoExporter({
      MediaRecorderClass: class {
        static isTypeSupported(type) {
          return type.includes('webm');
        }
      },
      createMediaRecorder: recorderFactory,
      createStream: () => new MockStream(),
      createWorker: () => worker,
      downloadBlob,
    });

    exporter.init({
      canvas,
      audio,
      button,
      notify: jest.fn(),
      getFileName: () => 'Clip',
    });

    exporter.start();
    expect(recorderFactory).toHaveBeenCalledTimes(1);
    const [, recorderOptions] = recorderFactory.mock.calls[0];
    expect(recorderOptions).toMatchObject({
      mimeType: expect.stringContaining('webm'),
    });
    expect(recorderOptions.videoBitsPerSecond).toBeGreaterThanOrEqual(6_000_000);
    expect(recorderOptions.audioBitsPerSecond).toBe(192_000);
    expect(recorderOptions.bitsPerSecond).toBe(recorderOptions.videoBitsPerSecond + recorderOptions.audioBitsPerSecond);
    exporter.stop();

    await Promise.resolve();
    await Promise.resolve();
    const convertCall = worker.postMessage.mock.calls.find(([message]) => message.type === 'convert');
    expect(convertCall).toBeDefined();
    const [{ jobId }] = convertCall;
    expect(exporter.getState().status).toBe('processing');

    const resultBuffer = new Uint8Array([1, 2, 3]).buffer;
    worker.listeners.message?.({ data: { type: 'result', jobId, buffer: resultBuffer } });

    expect(downloadBlob).toHaveBeenCalledTimes(1);
    const [blob, filename] = downloadBlob.mock.calls[0];
    expect(blob).toBeInstanceOf(Blob);
    expect(blob.type).toBe('video/mp4');
    expect(filename).toMatch(/^clip-/);
    expect(exporter.getState().status).toBe('idle');
  });

  test('falls back to WebM when MP4 encoder errors at runtime', async () => {
    const canvas = createCanvasElement();
    const audio = createAudioElement(true, true);
    const button = document.createElement('button');
    const downloadBlob = jest.fn();
    const notify = jest.fn();

    const worker = {
      listeners: {},
      addEventListener: jest.fn(function add(type, handler) {
        this.listeners[type] = handler;
      }),
      postMessage: jest.fn(),
    };

    const createEncodingErrorRecorder = () => {
      const events = {};
      return {
        events,
        state: 'inactive',
        start: jest.fn(function start() {
          this.state = 'recording';
          setTimeout(() => {
            events.error?.({
              error: { name: 'EncodingError', message: 'Encoder initialization failed.' },
            });
          }, 0);
        }),
        stop: jest.fn(),
        addEventListener: jest.fn((type, handler) => {
          events[type] = handler;
        }),
        removeEventListener: jest.fn((type, handler) => {
          if (events[type] === handler) {
            delete events[type];
          }
        }),
      };
    };

    const createSuccessfulRecorder = () => {
      const events = {};
      return {
        events,
        state: 'inactive',
        start: jest.fn(function start() {
          this.state = 'recording';
        }),
        stop: jest.fn(function stop() {
          this.state = 'inactive';
          if (events.dataavailable) {
            const blob = new Blob(['chunk'], { type: 'video/webm' });
            Object.defineProperty(blob, 'arrayBuffer', {
              configurable: true,
              value: jest.fn(() => Promise.resolve(new Uint8Array([4, 5, 6]).buffer)),
            });
            events.dataavailable({ data: blob });
          }
          events.stop?.({});
        }),
        addEventListener: jest.fn((type, handler) => {
          events[type] = handler;
        }),
        removeEventListener: jest.fn((type, handler) => {
          if (events[type] === handler) {
            delete events[type];
          }
        }),
      };
    };

    const createMediaRecorder = jest.fn((_, options = {}) => {
      if (options.mimeType && options.mimeType.includes('mp4')) {
        return createEncodingErrorRecorder();
      }
      return createSuccessfulRecorder();
    });

    const exporter = createVideoExporter({
      MediaRecorderClass: class {
        static isTypeSupported(type) {
          return type.includes('mp4') || type.includes('webm');
        }
      },
      createMediaRecorder,
      createStream: () => new MockStream(),
      createWorker: () => worker,
      downloadBlob,
    });

    exporter.init({
      canvas,
      audio,
      button,
      notify,
      getFileName: () => 'Clip',
    });

    expect(exporter.start()).toBe(true);

    const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
    await tick();
    await tick();

    expect(createMediaRecorder).toHaveBeenCalledWith(expect.anything(), expect.objectContaining({ mimeType: expect.stringContaining('mp4') }));
    expect(createMediaRecorder).toHaveBeenCalledWith(expect.anything(), expect.objectContaining({ mimeType: expect.stringContaining('webm') }));
    expect(notify).toHaveBeenCalledWith('Primary encoder failed; retrying with WebM fallback...', expect.objectContaining({ tone: 'warning' }));

    expect(exporter.getState().status).toBe('recording');

    exporter.stop();

    await Promise.resolve();
    await Promise.resolve();

    const convertCall = worker.postMessage.mock.calls.find(([message]) => message.type === 'convert');
    expect(convertCall).toBeDefined();
    const [{ jobId }] = convertCall;

    const resultBuffer = new Uint8Array([7, 8, 9]).buffer;
    worker.listeners.message?.({ data: { type: 'result', jobId, buffer: resultBuffer } });

    await Promise.resolve();

    expect(downloadBlob).toHaveBeenCalledTimes(1);
    const [blob, filename] = downloadBlob.mock.calls[0];
    expect(blob).toBeInstanceOf(Blob);
    expect(blob.type).toBe('video/mp4');
    expect(filename).toMatch(/^clip-/);
    expect(exporter.getState().status).toBe('idle');
  });
});
