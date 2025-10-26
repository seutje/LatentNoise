import { jest } from '@jest/globals';
import { createVideoExporter, pickSupportedMimeType, MIME_TYPE_CANDIDATES } from '../video-export.js';

describe('video-export', () => {
  let OriginalMediaRecorder;
  let OriginalMediaStream;

  class FakeWorker extends EventTarget {
    constructor() {
      super();
      this.messages = [];
      this.terminated = false;
    }

    postMessage(message) {
      this.messages.push(message);
    }

    terminate() {
      this.terminated = true;
    }
  }

  beforeAll(() => {
    OriginalMediaRecorder = global.MediaRecorder;
    OriginalMediaStream = global.MediaStream;
  });

  afterAll(() => {
    global.MediaRecorder = OriginalMediaRecorder;
    global.MediaStream = OriginalMediaStream;
  });

  beforeEach(() => {
    class FakeMediaStream {
      constructor(tracks = []) {
        this._tracks = tracks.slice();
      }

      addTrack(track) {
        this._tracks.push(track);
      }

      getTracks() {
        return this._tracks.slice();
      }

      getVideoTracks() {
        return this._tracks.filter((track) => track.kind === 'video');
      }

      getAudioTracks() {
        return this._tracks.filter((track) => track.kind === 'audio');
      }
    }

    class FakeMediaRecorder extends EventTarget {
      constructor(stream, options) {
        super();
        this.stream = stream;
        this.options = options;
        this.state = 'inactive';
      }

      start() {
        this.state = 'recording';
      }

      stop() {
        if (this.state === 'inactive') {
          return;
        }
        this.state = 'inactive';
        this.dispatchEvent(new Event('stop'));
      }
    }

    FakeMediaRecorder.isTypeSupported = jest.fn((type) => String(type).startsWith('video/mp4'));

    global.MediaStream = FakeMediaStream;
    global.MediaRecorder = FakeMediaRecorder;
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('selects an mp4 mime type when supported', () => {
    const type = pickSupportedMimeType(MIME_TYPE_CANDIDATES, global.MediaRecorder);
    expect(type).toMatch(/^video\/mp4/);
    expect(global.MediaRecorder.isTypeSupported).toHaveBeenCalled();
  });

  it('records video and resolves with a blob on stop', async () => {
    const worker = new FakeWorker();
    const videoTrack = { kind: 'video', stop: jest.fn() };
    const canvas = {
      captureStream: jest.fn(() => new MediaStream([videoTrack])),
    };
    const audioTrack = { kind: 'audio', stop: jest.fn() };
    const release = jest.fn();
    let recorder;

    const exporter = createVideoExporter({
      canvas,
      requestAudioStream: async () => ({
        stream: new MediaStream([audioTrack]),
        release,
      }),
      createWorker: () => worker,
      createRecorder: (stream, options) => {
        recorder = new MediaRecorder(stream, options);
        return recorder;
      },
    });

    await exporter.start({ trackTitle: 'Test Track' });
    const chunkData = {
      size: 4,
      arrayBuffer: () => Promise.resolve(new Uint8Array([1, 2, 3, 4]).buffer),
    };
    const dataEvent = new Event('dataavailable');
    Object.defineProperty(dataEvent, 'data', { value: chunkData });
    recorder.dispatchEvent(dataEvent);
    await Promise.resolve();
    await Promise.resolve();

    const stopPromise = exporter.stop();
    expect(worker.messages.some((message) => message.type === 'start')).toBe(true);
    expect(worker.messages.some((message) => message.type === 'stop')).toBe(true);

    const completeBlob = new Blob(['final'], { type: 'video/mp4' });
    worker.dispatchEvent(new MessageEvent('message', { data: { type: 'complete', blob: completeBlob } }));
    const result = await stopPromise;

    expect(result).toEqual({ blob: completeBlob, mimeType: expect.stringMatching(/^video\/mp4/), metadata: { trackTitle: 'Test Track' }, cancelled: false });
    expect(worker.terminated).toBe(true);
    expect(videoTrack.stop).toHaveBeenCalled();
    expect(audioTrack.stop).toHaveBeenCalled();
    expect(release).toHaveBeenCalled();
  });

  it('cancels recording without producing a blob', async () => {
    const worker = new FakeWorker();
    const canvas = {
      captureStream: jest.fn(() => new MediaStream([{ kind: 'video', stop: jest.fn() }])),
    };
    const exporter = createVideoExporter({
      canvas,
      requestAudioStream: async () => new MediaStream([{ kind: 'audio', stop: jest.fn() }]),
      createWorker: () => worker,
    });

    await exporter.start();
    const cancelPromise = exporter.cancel();
    expect(worker.messages.some((message) => message.type === 'cancel')).toBe(true);
    worker.dispatchEvent(new MessageEvent('message', { data: { type: 'cancelled' } }));
    const result = await cancelPromise;

    expect(result).toEqual({ blob: null, mimeType: expect.stringMatching(/^video\/mp4/), metadata: null, cancelled: true });
    expect(worker.terminated).toBe(true);
  });
});
