const DEFAULT_FRAME_RATE = 60;
const DEFAULT_TIMESLICE_MS = 1000;
const DEFAULT_VIDEO_BITRATE = 8_000_000;
const DEFAULT_AUDIO_BITRATE = 192_000;
const MIME_CANDIDATES = Object.freeze([
  'video/mp4;codecs="avc1.64001E,mp4a.40.2"',
  'video/mp4;codecs="avc1.4D401E,mp4a.40.2"',
  'video/mp4;codecs="avc1.42E01E,mp4a.40.2"',
  'video/mp4',
]);

function defaultLogger() {
  return typeof console !== 'undefined' ? console : { log() {}, warn() {}, error() {} };
}

function defaultCreateWorker() {
  const url = new URL('./workers/video-export-worker.js', import.meta.url);
  return new Worker(url, { type: 'module' });
}

function stopTracks(stream) {
  if (!stream || typeof stream.getTracks !== 'function') {
    return;
  }
  stream.getTracks().forEach((track) => {
    try {
      track.stop();
    } catch {
      // Ignore track stop failures (already stopped, etc.).
    }
  });
}

function normalizeAudioRequestResult(result) {
  if (!result) {
    return { stream: null, cleanup: null };
  }
  if (result instanceof MediaStream) {
    return { stream: result, cleanup: null };
  }
  const stream = result.stream instanceof MediaStream ? result.stream : null;
  const cleanup = typeof result.release === 'function' ? result.release : result.disconnect;
  return { stream, cleanup: typeof cleanup === 'function' ? cleanup : null };
}

export function pickSupportedMimeType(candidates = MIME_CANDIDATES, recorderClass = typeof MediaRecorder !== 'undefined' ? MediaRecorder : null) {
  if (!recorderClass || typeof recorderClass.isTypeSupported !== 'function') {
    return null;
  }
  for (const candidate of candidates) {
    if (candidate && recorderClass.isTypeSupported(candidate)) {
      return candidate;
    }
  }
  return null;
}

export function createVideoExporter(options = {}) {
  const {
    canvas,
    frameRate = DEFAULT_FRAME_RATE,
    timeslice = DEFAULT_TIMESLICE_MS,
    requestAudioStream = null,
    onStart = () => {},
    onComplete = () => {},
    onCancel = () => {},
    onError = () => {},
    createWorker = defaultCreateWorker,
    createRecorder = (stream, recorderOptions) => new MediaRecorder(stream, recorderOptions),
    logger = defaultLogger(),
    mimeTypes = MIME_CANDIDATES,
    videoBitsPerSecond = DEFAULT_VIDEO_BITRATE,
    audioBitsPerSecond = DEFAULT_AUDIO_BITRATE,
  } = options;

  if (!canvas) {
    throw new Error('createVideoExporter requires a canvas reference.');
  }

  const recorderClass = typeof MediaRecorder !== 'undefined' ? MediaRecorder : null;
  const preferredMimeType = pickSupportedMimeType(mimeTypes, recorderClass);

  const state = {
    recording: false,
    worker: null,
    recorder: null,
    stream: null,
    videoStream: null,
    audioStream: null,
    audioCleanup: null,
    mimeType: preferredMimeType,
    pendingStop: null,
    resolveStop: null,
    rejectStop: null,
    discard: false,
    metadata: null,
    handlers: {
      data: null,
      stop: null,
      error: null,
      worker: null,
    },
  };

  function cleanup() {
    if (state.recorder && state.handlers.data) {
      state.recorder.removeEventListener('dataavailable', state.handlers.data);
    }
    if (state.recorder && state.handlers.stop) {
      state.recorder.removeEventListener('stop', state.handlers.stop);
    }
    if (state.recorder && state.handlers.error) {
      state.recorder.removeEventListener('error', state.handlers.error);
    }
    if (state.worker && state.handlers.worker) {
      state.worker.removeEventListener('message', state.handlers.worker);
    }
    try {
      state.worker?.terminate?.();
    } catch {
      // Ignore termination failures.
    }
    if (state.audioCleanup) {
      try {
        state.audioCleanup();
      } catch {
        // Ignore cleanup failures.
      }
    }
    stopTracks(state.stream);
    stopTracks(state.videoStream);
    stopTracks(state.audioStream);
    state.recording = false;
    state.worker = null;
    state.recorder = null;
    state.stream = null;
    state.videoStream = null;
    state.audioStream = null;
    state.audioCleanup = null;
    state.pendingStop = null;
    state.resolveStop = null;
    state.rejectStop = null;
    state.discard = false;
    state.metadata = null;
    state.handlers.data = null;
    state.handlers.stop = null;
    state.handlers.error = null;
    state.handlers.worker = null;
  }

  function finalize(result) {
    const resolve = state.resolveStop;
    cleanup();
    if (resolve) {
      resolve(result);
    }
  }

  function fail(error) {
    const reject = state.rejectStop;
    cleanup();
    if (reject) {
      reject(error);
    }
    try {
      onError(error);
    } catch (notifyError) {
      logger?.error?.('[video-export] onError handler failed', notifyError);
    }
  }

  async function handleDataAvailable(event) {
    if (!event || !event.data || event.data.size === 0 || !state.worker || state.discard) {
      return;
    }
    try {
      const buffer = await event.data.arrayBuffer();
      if (buffer && buffer.byteLength > 0) {
        state.worker.postMessage({ type: 'chunk', chunk: buffer }, [buffer]);
      }
    } catch (error) {
      logger?.warn?.('[video-export] Failed to transfer recording chunk', error);
    }
  }

  function handleRecorderStop() {
    if (!state.worker) {
      return;
    }
    const messageType = state.discard ? 'cancel' : 'stop';
    try {
      state.worker.postMessage({ type: messageType });
    } catch (error) {
      fail(error);
    }
  }

  function handleRecorderError(event) {
    const error = event?.error ?? new Error('MediaRecorder error.');
    fail(error);
  }

  function handleWorkerMessage(event) {
    const data = event.data;
    if (!data || typeof data.type !== 'string') {
      return;
    }
    switch (data.type) {
      case 'complete': {
        try {
          onComplete({ blob: data.blob, mimeType: state.mimeType, metadata: state.metadata, cancelled: false });
        } catch (callbackError) {
          logger?.error?.('[video-export] onComplete handler failed', callbackError);
        }
        finalize({ blob: data.blob, mimeType: state.mimeType, metadata: state.metadata, cancelled: false });
        break;
      }
      case 'cancelled': {
        try {
          onCancel({ mimeType: state.mimeType, metadata: state.metadata });
        } catch (callbackError) {
          logger?.error?.('[video-export] onCancel handler failed', callbackError);
        }
        finalize({ blob: null, mimeType: state.mimeType, metadata: state.metadata, cancelled: true });
        break;
      }
      case 'error': {
        const error = new Error(data.message ?? 'Video export worker error.');
        fail(error);
        break;
      }
      default:
        break;
    }
  }

  async function start(metadata = null) {
    if (state.recording) {
      throw new Error('Video export already in progress.');
    }
    const recorderCtor = typeof MediaRecorder !== 'undefined' ? MediaRecorder : null;
    const mimeType = state.mimeType ?? pickSupportedMimeType(mimeTypes, recorderCtor);
    if (!recorderCtor || typeof canvas.captureStream !== 'function') {
      throw new Error('Video export requires MediaRecorder and canvas captureStream support.');
    }
    if (!mimeType) {
      throw new Error('MP4 recording is not supported in this browser.');
    }

    let videoStream;
    try {
      videoStream = canvas.captureStream(frameRate);
    } catch (error) {
      throw new Error(`Failed to capture canvas stream: ${error?.message ?? error}`);
    }
    if (!(videoStream instanceof MediaStream)) {
      throw new Error('Canvas captureStream did not return a MediaStream.');
    }

    let audioStream = null;
    let audioCleanup = null;
    if (typeof requestAudioStream === 'function') {
      const audioResult = await requestAudioStream();
      const normalized = normalizeAudioRequestResult(audioResult);
      audioStream = normalized.stream;
      audioCleanup = normalized.cleanup;
    }

    const combinedStream = new MediaStream();
    videoStream.getVideoTracks().forEach((track) => combinedStream.addTrack(track));
    if (audioStream) {
      audioStream.getAudioTracks().forEach((track) => combinedStream.addTrack(track));
    }

    let worker;
    try {
      worker = createWorker();
    } catch (error) {
      stopTracks(combinedStream);
      stopTracks(videoStream);
      stopTracks(audioStream);
      if (audioCleanup) {
        try {
          audioCleanup();
        } catch {
          // Ignore cleanup errors.
        }
      }
      throw error;
    }

    let recorder;
    try {
      recorder = createRecorder(combinedStream, {
        mimeType,
        videoBitsPerSecond,
        audioBitsPerSecond,
      });
    } catch (error) {
      worker.terminate?.();
      stopTracks(combinedStream);
      stopTracks(videoStream);
      stopTracks(audioStream);
      if (audioCleanup) {
        try {
          audioCleanup();
        } catch {
          // Ignore cleanup errors.
        }
      }
      throw error;
    }

    state.recording = true;
    state.worker = worker;
    state.recorder = recorder;
    state.stream = combinedStream;
    state.videoStream = videoStream;
    state.audioStream = audioStream;
    state.audioCleanup = audioCleanup;
    state.mimeType = mimeType;
    state.metadata = metadata;
    state.discard = false;
    state.pendingStop = new Promise((resolve, reject) => {
      state.resolveStop = resolve;
      state.rejectStop = reject;
    });

    state.handlers.data = (event) => {
      void handleDataAvailable(event);
    };
    state.handlers.stop = () => {
      handleRecorderStop();
    };
    state.handlers.error = (event) => {
      handleRecorderError(event);
    };
    state.handlers.worker = (event) => {
      handleWorkerMessage(event);
    };

    recorder.addEventListener('dataavailable', state.handlers.data);
    recorder.addEventListener('stop', state.handlers.stop);
    recorder.addEventListener('error', state.handlers.error);
    worker.addEventListener('message', state.handlers.worker);

    try {
      worker.postMessage({ type: 'start', mimeType });
      recorder.start(timeslice);
    } catch (error) {
      fail(error);
      throw error;
    }

    try {
      onStart({ mimeType, metadata });
    } catch (callbackError) {
      logger?.error?.('[video-export] onStart handler failed', callbackError);
    }
  }

  function ensureRecorderStopped() {
    if (!state.recorder) {
      return;
    }
    if (state.recorder.state && state.recorder.state === 'inactive') {
      return;
    }
    try {
      state.recorder.stop();
    } catch {
      // Ignore stop errors (already stopping, etc.).
    }
  }

  function stop() {
    if (!state.recording) {
      return Promise.resolve(null);
    }
    state.discard = false;
    ensureRecorderStopped();
    return state.pendingStop ?? Promise.resolve(null);
  }

  function cancel() {
    if (!state.recording) {
      return Promise.resolve(null);
    }
    state.discard = true;
    ensureRecorderStopped();
    return state.pendingStop ?? Promise.resolve(null);
  }

  function isRecording() {
    return state.recording;
  }

  function canRecord() {
    if (!state.mimeType) {
      const recorderCtor = typeof MediaRecorder !== 'undefined' ? MediaRecorder : null;
      return Boolean(pickSupportedMimeType(mimeTypes, recorderCtor));
    }
    return true;
  }

  return {
    start,
    stop,
    cancel,
    isRecording,
    canRecord,
  };
}

export const MIME_TYPE_CANDIDATES = MIME_CANDIDATES;
