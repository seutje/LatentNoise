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
  let pausedState = paused;
  let currentTime = 0;
  const audio = document.createElement('audio');
  audio.src = 'https://example.com/audio.mp3';
  Object.defineProperty(audio, 'paused', {
    configurable: true,
    get: () => pausedState,
  });
  Object.defineProperty(audio, 'currentTime', {
    configurable: true,
    get: () => currentTime,
    set: (value) => {
      currentTime = value;
    },
  });
  Object.defineProperty(audio, 'currentSrc', {
    configurable: true,
    get: () => audio.src,
  });
  audio.play = jest.fn(() => {
    pausedState = false;
    return Promise.resolve();
  });
  audio.pause = jest.fn(() => {
    pausedState = true;
  });
  if (hasCapture) {
    audio.captureStream = jest.fn(() => new MockStream([createMockTrack('audio')]));
  }
  return audio;
}

function createMockRecorderFactory({ blobType, onCreate, createChunk }) {
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
          const blob = typeof createChunk === 'function'
            ? createChunk({ stream, options })
            : new Blob(['chunk'], { type: blobType });
          if (typeof blob.arrayBuffer !== 'function') {
            Object.defineProperty(blob, 'arrayBuffer', {
              configurable: true,
              value: jest.fn(() => Promise.resolve(new Uint8Array([1, 2, 3]).buffer)),
            });
          }
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
  beforeEach(() => {
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        headers: { get: () => 'audio/mpeg' },
        arrayBuffer: () => Promise.resolve(new Uint8Array([9, 9, 9]).buffer),
      }),
    );
  });

  afterEach(() => {
    delete global.fetch;
  });

  test('createDownloadFileName sanitizes titles and formats timestamp', () => {
    const name = createDownloadFileName('My Test Track!', new Date('2025-01-02T03:04:05Z'));
    expect(name).toBe('my-test-track-2025-01-02T03-04-05-000Z.mp4');
    const webmName = createDownloadFileName('My Test Track!', new Date('2025-01-02T03:04:05Z'), 'webm');
    expect(webmName).toBe('my-test-track-2025-01-02T03-04-05-000Z.webm');
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

  test('start applies custom export options for bitrate and frame rate', () => {
    const canvas = createCanvasElement();
    const audio = createAudioElement(true, true);
    const button = document.createElement('button');
    const recorderFactory = createMockRecorderFactory({ blobType: 'video/mp4' });
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
          return type.includes('mp4');
        }
      },
      createMediaRecorder: recorderFactory,
      createStream: () => new MockStream(),
      createWorker: () => worker,
      downloadBlob: jest.fn(),
    });

    exporter.init({
      canvas,
      audio,
      button,
      notify: jest.fn(),
      getFileName: () => 'Custom Clip',
    });

    expect(
      exporter.start({
        width: 2560,
        height: 1440,
        frameRate: 48,
        videoBitsPerSecond: 25_000_000,
      }),
    ).toBe(true);

    expect(canvas.captureStream).toHaveBeenCalledWith(48);
    expect(recorderFactory).toHaveBeenCalledTimes(1);
    const [, recorderOptions] = recorderFactory.mock.calls[0];
    expect(recorderOptions.mimeType).toContain('mp4');
    expect(recorderOptions.videoBitsPerSecond).toBe(25_000_000);
    expect(recorderOptions.bitsPerSecond).toBeGreaterThan(recorderOptions.videoBitsPerSecond);

    exporter.stop();
  });

  test('records and downloads MP4 directly when supported', async () => {
    const canvas = createCanvasElement();
    const audio = createAudioElement(true, true);
    const button = document.createElement('button');
    const downloadBlob = jest.fn();
    const recorderFactory = createMockRecorderFactory({ blobType: 'video/mp4' });
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
          return type.includes('video/mp4');
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

    await new Promise((resolve) => setTimeout(resolve, 0));
    await Promise.resolve();

    const convertCall = worker.postMessage.mock.calls.find(([message]) => message.type === 'convert');
    expect(convertCall).toBeDefined();
    const [convertMessage] = convertCall;
    expect(convertMessage.preferCopyVideo).toBe(true);
    expect(convertMessage.externalAudioType).toBe('audio/mpeg');
    expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining('audio.mp3'));

    const jobId = convertMessage.jobId;
    const resultBuffer = new Uint8Array([1, 2, 3]).buffer;
    worker.listeners.message?.({ data: { type: 'result', jobId, buffer: resultBuffer } });

    expect(downloadBlob).toHaveBeenCalledTimes(1);
    const [blob, filename] = downloadBlob.mock.calls[0];
    expect(blob).toBeInstanceOf(Blob);
    expect(blob.type).toContain('video/mp4');
    expect(filename).toMatch(/^test-clip-/);
    expect(exporter.getState().status).toBe('idle');
  });

  test('exports WebM directly when format is requested', async () => {
    const canvas = createCanvasElement();
    const audio = createAudioElement(true, true);
    const button = document.createElement('button');
    const downloadBlob = jest.fn();
    const notify = jest.fn();
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
      notify,
      getFileName: () => 'WebM Clip',
    });

    expect(
      exporter.start({
        format: 'webm',
        width: 1920,
        height: 1080,
        frameRate: 50,
        videoBitsPerSecond: 8_000_000,
      }),
    ).toBe(true);

    const [, recorderOptions] = recorderFactory.mock.calls[0];
    expect(recorderOptions.mimeType).toContain('webm');
    expect(recorderOptions.videoBitsPerSecond).toBe(8_000_000);

    exporter.stop();

    await new Promise((resolve) => setTimeout(resolve, 0));
    await Promise.resolve();

    const convertCall = worker.postMessage.mock.calls.find(([message]) => message?.type === 'convert');
    expect(convertCall).toBeUndefined();
    expect(downloadBlob).toHaveBeenCalledTimes(1);
    const [blob, filename] = downloadBlob.mock.calls[0];
    expect(blob).toBeInstanceOf(Blob);
    expect(blob.type).toContain('webm');
    expect(filename).toMatch(/\.webm$/);
    expect(notify).toHaveBeenCalledWith('Video export ready. Downloading WebM.', expect.any(Object));
    expect(exporter.getState().status).toBe('idle');
  });

  test('recovers when recorded blob arrayBuffer throws NotReadableError', async () => {
    const canvas = createCanvasElement();
    const audio = createAudioElement(true, true);
    const button = document.createElement('button');
    const downloadBlob = jest.fn();
    const notReadableError = Object.assign(new Error('The requested file could not be read.'), {
      name: 'NotReadableError',
    });

    const recorderFactory = createMockRecorderFactory({
      blobType: 'video/mp4',
      createChunk: () => {
        const blob = new Blob(['chunk'], { type: 'video/mp4' });
        Object.defineProperty(blob, 'arrayBuffer', {
          configurable: true,
          value: jest.fn(() => Promise.reject(notReadableError)),
        });
        return blob;
      },
    });

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
          return type.includes('video/mp4');
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
      getFileName: () => 'Test Clip',
    });

    expect(exporter.start()).toBe(true);
    expect(exporter.stop()).toBe(true);

    await new Promise((resolve) => setTimeout(resolve, 0));
    await Promise.resolve();

    const convertCall = worker.postMessage.mock.calls.find(([message]) => message.type === 'convert');
    expect(convertCall).toBeDefined();
    const [convertMessage] = convertCall;
    expect(convertMessage.hasAudio).toBe(true);
    expect(downloadBlob).not.toHaveBeenCalled();
  });

  test('recovers when recorded blob arrayBuffer throws TypeError failed to fetch', async () => {
    const canvas = createCanvasElement();
    const audio = createAudioElement(true, true);
    const button = document.createElement('button');
    const downloadBlob = jest.fn();
    const failedFetchError = Object.assign(new TypeError('Failed to fetch'), {
      name: 'TypeError',
    });

    const recorderFactory = createMockRecorderFactory({
      blobType: 'video/mp4',
      createChunk: () => {
        const blob = new Blob(['chunk'], { type: 'video/mp4' });
        Object.defineProperty(blob, 'arrayBuffer', {
          configurable: true,
          value: jest.fn(() => Promise.reject(failedFetchError)),
        });
        return blob;
      },
    });

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
          return type.includes('video/mp4');
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
      getFileName: () => 'Test Clip',
    });

    expect(exporter.start()).toBe(true);
    expect(exporter.stop()).toBe(true);

    await new Promise((resolve) => setTimeout(resolve, 0));
    await Promise.resolve();

    const convertCall = worker.postMessage.mock.calls.find(([message]) => message.type === 'convert');
    expect(convertCall).toBeDefined();
    const [convertMessage] = convertCall;
    expect(convertMessage.hasAudio).toBe(true);
    expect(downloadBlob).not.toHaveBeenCalled();
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

    await new Promise((resolve) => setTimeout(resolve, 0));
    await Promise.resolve();
    const convertCall = worker.postMessage.mock.calls.find(([message]) => message.type === 'convert');
    expect(convertCall).toBeDefined();
    const [convertMessage] = convertCall;
    expect(convertMessage.externalAudioType).toBe('audio/mpeg');
    expect(convertMessage.preferCopyVideo).toBe(false);
    expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining('audio.mp3'));
    const { jobId } = convertMessage;
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

    await new Promise((resolve) => setTimeout(resolve, 0));
    await Promise.resolve();

    const convertCall = worker.postMessage.mock.calls.find(([message]) => message.type === 'convert');
    expect(convertCall).toBeDefined();
    const [convertMessage] = convertCall;
    expect(convertMessage.externalAudioType).toBe('audio/mpeg');
    const { jobId } = convertMessage;

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
