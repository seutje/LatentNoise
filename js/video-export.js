const DEFAULT_TIMESLICE_MS = 1000;
const MIN_TIMESLICE_MS = 250;
const MAX_TIMESLICE_MS = 10000;
const DEFAULT_FPS = 60;
const MIN_FPS = 1;
const MAX_FPS = 120;
const DEFAULT_VIDEO_BITRATE = 6_000_000;
const DEFAULT_AUDIO_BITRATE = 192_000;
const DEFAULT_BASENAME = 'latent-noise';

export const MP4_MIME_CANDIDATES = Object.freeze([
  'video/mp4;codecs=avc1.42E01E,mp4a.40.2',
  'video/mp4;codecs=h264,aac',
  'video/mp4',
]);

function clamp(value, min, max) {
  if (!Number.isFinite(value)) {
    return min;
  }
  if (value < min) {
    return min;
  }
  if (value > max) {
    return max;
  }
  return value;
}

function defaultIsTypeSupported(type) {
  if (!type || typeof type !== 'string') {
    return false;
  }
  if (typeof MediaRecorder === 'undefined') {
    return false;
  }
  if (typeof MediaRecorder.isTypeSupported === 'function') {
    try {
      return MediaRecorder.isTypeSupported(type);
    } catch {
      return false;
    }
  }
  return false;
}

export function selectMp4MimeType(isTypeSupported = defaultIsTypeSupported) {
  if (typeof isTypeSupported !== 'function') {
    return null;
  }
  for (const candidate of MP4_MIME_CANDIDATES) {
    try {
      if (isTypeSupported(candidate)) {
        return candidate;
      }
    } catch {
      // Ignore detection failure for this candidate and continue scanning.
    }
  }
  return null;
}

export function buildExportFileName(title, extension = 'mp4') {
  const normalized = typeof title === 'string' ? title.trim().toLowerCase() : '';
  const slug = normalized
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/-+/g, '-');
  const safeBase = slug.length > 0 ? slug : DEFAULT_BASENAME;
  const safeExtension = typeof extension === 'string' && extension.trim().length > 0 ? extension.trim() : 'mp4';
  return `${safeBase}.${safeExtension}`;
}

function sanitizeTimeslice(value) {
  return clamp(Math.round(Number(value) || DEFAULT_TIMESLICE_MS), MIN_TIMESLICE_MS, MAX_TIMESLICE_MS);
}

function getAnimationFrameHandles() {
  if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
    return {
      request: (cb) => window.requestAnimationFrame(cb),
      cancel: (id) => window.cancelAnimationFrame(id),
    };
  }
  return {
    request: (cb) => setTimeout(() => cb(Date.now()), 1000 / 60),
    cancel: (id) => clearTimeout(id),
  };
}

function captureAudioPlaybackState(element) {
  return {
    currentTime: Number(element?.currentTime) || 0,
    wasPaused: !element || element.paused,
    playbackRate: Number.isFinite(element?.playbackRate) ? element.playbackRate : 1,
    loop: Boolean(element?.loop),
  };
}

async function restoreAudioPlaybackState(element, state) {
  if (!element || !state) {
    return;
  }
  try {
    element.pause();
  } catch {
    // Ignore pause failures.
  }
  element.loop = state.loop;
  try {
    element.playbackRate = state.playbackRate;
  } catch {
    // Ignore playbackRate assignment failures.
  }
  try {
    element.currentTime = state.currentTime;
  } catch {
    // Ignore seek failures.
  }
  if (!state.wasPaused) {
    try {
      await element.play();
    } catch {
      // Swallow autoplay rejections; the caller initiated playback earlier.
    }
  }
}

function emitState(handler, state, detail = {}) {
  if (typeof handler === 'function') {
    handler({ state, ...detail });
  }
}

export async function exportVideo(options = {}) {
  const {
    canvas,
    audioElement,
    requestAudioStream,
    onStateChange,
    onProgress,
    fps = DEFAULT_FPS,
    timeslice = DEFAULT_TIMESLICE_MS,
    videoBitsPerSecond = DEFAULT_VIDEO_BITRATE,
    audioBitsPerSecond = DEFAULT_AUDIO_BITRATE,
  } = options;

  emitState(onStateChange, 'preparing');

  if (typeof MediaRecorder === 'undefined') {
    const error = new Error('MediaRecorder API is not available in this browser.');
    emitState(onStateChange, 'error', { error });
    throw error;
  }

  const mimeType = selectMp4MimeType();
  if (!mimeType) {
    const error = new Error('MP4 recording is not supported in this browser.');
    emitState(onStateChange, 'error', { error });
    throw error;
  }

  if (!(canvas instanceof HTMLCanvasElement) || typeof canvas.captureStream !== 'function') {
    const error = new Error('Canvas captureStream() is not supported.');
    emitState(onStateChange, 'error', { error });
    throw error;
  }

  if (!(audioElement instanceof HTMLMediaElement)) {
    const error = new Error('An HTMLMediaElement is required for audio capture.');
    emitState(onStateChange, 'error', { error });
    throw error;
  }

  if (typeof requestAudioStream !== 'function') {
    const error = new Error('An audio stream provider is required for export.');
    emitState(onStateChange, 'error', { error });
    throw error;
  }

  const duration = Number(audioElement.duration);
  if (!Number.isFinite(duration) || duration <= 0) {
    const error = new Error('Load the track before exporting video.');
    emitState(onStateChange, 'error', { error });
    throw error;
  }

  const playbackState = captureAudioPlaybackState(audioElement);

  let audioStream;
  try {
    audioStream = await requestAudioStream();
  } catch (error) {
    emitState(onStateChange, 'error', { error });
    await restoreAudioPlaybackState(audioElement, playbackState);
    throw error;
  }

  if (!audioStream || typeof audioStream.getAudioTracks !== 'function') {
    const error = new Error('Unable to access audio for export.');
    emitState(onStateChange, 'error', { error });
    await restoreAudioPlaybackState(audioElement, playbackState);
    throw error;
  }

  const audioTracks = audioStream.getAudioTracks().filter((track) => track.readyState !== 'ended');
  if (audioTracks.length === 0) {
    const error = new Error('The audio capture stream has no active tracks.');
    emitState(onStateChange, 'error', { error });
    await restoreAudioPlaybackState(audioElement, playbackState);
    throw error;
  }

  const frameRate = clamp(Number(fps) || DEFAULT_FPS, MIN_FPS, MAX_FPS);
  let canvasStream;
  try {
    canvasStream = canvas.captureStream(frameRate);
  } catch (error) {
    emitState(onStateChange, 'error', { error });
    await restoreAudioPlaybackState(audioElement, playbackState);
    throw error;
  }

  if (!canvasStream || typeof canvasStream.getVideoTracks !== 'function') {
    const error = new Error('Unable to capture the canvas stream.');
    emitState(onStateChange, 'error', { error });
    await restoreAudioPlaybackState(audioElement, playbackState);
    throw error;
  }

  const videoTracks = canvasStream.getVideoTracks();
  if (!videoTracks || videoTracks.length === 0) {
    const error = new Error('The canvas stream does not contain a video track.');
    emitState(onStateChange, 'error', { error });
    await restoreAudioPlaybackState(audioElement, playbackState);
    throw error;
  }

  const combinedStream = new MediaStream();
  videoTracks.forEach((track) => combinedStream.addTrack(track));
  audioTracks.forEach((track) => combinedStream.addTrack(track));

  if (combinedStream.getAudioTracks().length === 0) {
    const error = new Error('Unable to attach audio track to the recorder.');
    emitState(onStateChange, 'error', { error });
    await restoreAudioPlaybackState(audioElement, playbackState);
    throw error;
  }

  let recorder;
  try {
    recorder = new MediaRecorder(combinedStream, {
      mimeType,
      videoBitsPerSecond: Number.isFinite(videoBitsPerSecond) && videoBitsPerSecond > 0
        ? videoBitsPerSecond
        : DEFAULT_VIDEO_BITRATE,
      audioBitsPerSecond: Number.isFinite(audioBitsPerSecond) && audioBitsPerSecond > 0
        ? audioBitsPerSecond
        : DEFAULT_AUDIO_BITRATE,
    });
  } catch (error) {
    emitState(onStateChange, 'error', { error });
    await restoreAudioPlaybackState(audioElement, playbackState);
    throw error;
  }

  const timesliceMs = sanitizeTimeslice(timeslice);
  const { request: requestFrame, cancel: cancelFrame } = getAnimationFrameHandles();

  return new Promise((resolve, reject) => {
    const chunks = [];
    let rafId = 0;
    let endedListener = null;
    let stopIssued = false;

    const stopProgressLoop = () => {
      if (rafId) {
        cancelFrame(rafId);
        rafId = 0;
      }
    };

    const cleanup = () => {
      stopProgressLoop();
      if (endedListener) {
        audioElement.removeEventListener('ended', endedListener);
        endedListener = null;
      }
      videoTracks.forEach((track) => {
        if (typeof track.stop === 'function') {
          try {
            track.stop();
          } catch {
            // Ignore stop failures.
          }
        }
      });
    };

    const finalize = async () => {
      cleanup();
      const blob = new Blob(chunks, { type: mimeType });
      const url = URL.createObjectURL(blob);
      emitState(onStateChange, 'complete', { blob, url, mimeType });
      if (typeof onProgress === 'function') {
        onProgress({ currentTime: duration, duration });
      }
      await restoreAudioPlaybackState(audioElement, playbackState);
      resolve({ blob, url, mimeType });
    };

    const fail = async (error) => {
      cleanup();
      emitState(onStateChange, 'error', { error });
      await restoreAudioPlaybackState(audioElement, playbackState);
      reject(error);
    };

    recorder.addEventListener('dataavailable', (event) => {
      if (event.data && event.data.size > 0) {
        chunks.push(event.data);
      }
    });

    recorder.addEventListener('error', (event) => {
      const error = event?.error instanceof Error ? event.error : new Error('Recording failed.');
      fail(error);
    });

    recorder.addEventListener('stop', () => {
      finalize().catch((error) => {
        // If finalize fails, propagate via rejection.
        fail(error);
      });
    });

    const startProgressLoop = () => {
      if (typeof onProgress !== 'function') {
        return;
      }
      const tick = () => {
        if (recorder.state !== 'recording') {
          stopProgressLoop();
          return;
        }
        onProgress({
          currentTime: Number(audioElement.currentTime) || 0,
          duration,
        });
        rafId = requestFrame(tick);
      };
      rafId = requestFrame(tick);
    };

    const beginRecording = async () => {
      try {
        audioElement.pause();
      } catch {
        // Ignore pause errors.
      }
      audioElement.loop = false;
      try {
        audioElement.currentTime = 0;
      } catch {
        // Ignore seek failures.
      }

      try {
        recorder.start(timesliceMs);
      } catch (error) {
        await fail(error);
        return;
      }

      emitState(onStateChange, 'recording', { mimeType });
      startProgressLoop();

      endedListener = () => {
        if (stopIssued) {
          return;
        }
        stopIssued = true;
        emitState(onStateChange, 'finalizing');
        try {
          if (recorder.state !== 'inactive') {
            recorder.stop();
          }
        } catch (error) {
          fail(error);
        }
      };

      audioElement.addEventListener('ended', endedListener, { once: true });

      try {
        await audioElement.play();
      } catch (error) {
        await fail(error);
      }
    };

    beginRecording().catch((error) => {
      fail(error);
    });
  });
}
