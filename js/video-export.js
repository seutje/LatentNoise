const STATE_IDLE = 'idle';
const STATE_RECORDING = 'recording';
const STATE_PROCESSING = 'processing';

const PREFERRED_MP4_MIME_TYPES = Object.freeze([
  'video/mp4;codecs=avc1.42E01E,mp4a.40.2',
  'video/mp4;codecs=avc1.4D401E,mp4a.40.2',
  'video/mp4;codecs=avc1.64001F,mp4a.40.2',
  'video/mp4',
]);

const PREFERRED_MP4_VIDEO_ONLY = Object.freeze([
  'video/mp4;codecs=avc1.42E01E',
  'video/mp4;codecs=avc1.4D401E',
  'video/mp4;codecs=avc1.64001F',
  'video/mp4',
]);

const FALLBACK_WEBM_MIME_TYPES = Object.freeze([
  'video/webm;codecs=vp9,opus',
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8,opus',
  'video/webm;codecs=vp8',
  'video/webm',
]);

const BASE_EXPORT_RESOLUTION_PIXELS = 1920 * 1080;
const BASE_EXPORT_VIDEO_BITRATE = 12_000_000;
const MIN_EXPORT_VIDEO_BITRATE = 6_000_000;
const MAX_EXPORT_VIDEO_BITRATE = 48_000_000;
const EXPORT_AUDIO_BITRATE = 192_000;

function defaultNotify() {}

function defaultDownloadBlob(blob, filename) {
  if (!(blob instanceof Blob)) {
    return;
  }
  const url = URL.createObjectURL(blob);
  try {
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename || 'latent-noise-export.mp4';
    anchor.rel = 'noopener';
    anchor.style.display = 'none';
    document.body?.append(anchor);
    anchor.click();
    anchor.remove();
  } finally {
    URL.revokeObjectURL(url);
  }
}

function defaultCreateWorker() {
  if (typeof Worker === 'undefined') {
    return null;
  }
  try {
    return new Worker(new URL('./workers/video-export-worker.js', import.meta.url), { type: 'module' });
  } catch (error) {
    console.error('[video-export] Failed to create worker', error);
    return null;
  }
}

function getDefaultMediaRecorderClass() {
  if (typeof MediaRecorder !== 'undefined') {
    return MediaRecorder;
  }
  return null;
}

function getDefaultMediaStreamFactory() {
  if (typeof MediaStream !== 'undefined') {
    return () => new MediaStream();
  }
  return null;
}

function clamp01(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return 0;
  }
  if (numeric <= 0) {
    return 0;
  }
  if (numeric >= 1) {
    return 1;
  }
  return numeric;
}

function clampNumber(value, min, max) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return min;
  }
  if (numeric < min) {
    return min;
  }
  if (numeric > max) {
    return max;
  }
  return numeric;
}

function sanitizeTitleForFile(title) {
  if (typeof title !== 'string') {
    return 'latent-noise';
  }
  const trimmed = title.trim();
  if (!trimmed) {
    return 'latent-noise';
  }
  const normalized = trimmed
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+/, '')
    .replace(/-+$/, '');
  if (!normalized) {
    return 'latent-noise';
  }
  return normalized.slice(0, 64);
}

function formatTimestamp(date = new Date()) {
  if (!(date instanceof Date) || Number.isNaN(date.getTime())) {
    return formatTimestamp(new Date());
  }
  return date.toISOString().replace(/[:.]/g, '-');
}

export function createDownloadFileName(title, date = new Date()) {
  const safeTitle = sanitizeTitleForFile(title);
  const timestamp = formatTimestamp(date);
  return `${safeTitle}-${timestamp}.mp4`;
}

function calculateVideoBitrate(width, height, frameRate = 60) {
  const w = Number(width);
  const h = Number(height);
  if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) {
    return 0;
  }
  const pixelCount = w * h;
  const pixelScale = clampNumber(pixelCount / BASE_EXPORT_RESOLUTION_PIXELS, 0.35, 4);
  const fps = Number(frameRate);
  const fpsScale = Number.isFinite(fps) && fps > 0 ? clampNumber(fps / 60, 0.5, 1.2) : 1;
  const target = BASE_EXPORT_VIDEO_BITRATE * pixelScale * fpsScale;
  return Math.round(clampNumber(target, MIN_EXPORT_VIDEO_BITRATE, MAX_EXPORT_VIDEO_BITRATE));
}

function createRecorderQualityOptions(canvas, hasAudio, frameRate) {
  if (!canvas) {
    return null;
  }
  const videoBitsPerSecond = calculateVideoBitrate(canvas.width, canvas.height, frameRate);
  const audioBitsPerSecond = hasAudio ? EXPORT_AUDIO_BITRATE : 0;
  if (videoBitsPerSecond <= 0 && audioBitsPerSecond <= 0) {
    return null;
  }
  const options = {};
  if (videoBitsPerSecond > 0) {
    options.videoBitsPerSecond = videoBitsPerSecond;
  }
  if (audioBitsPerSecond > 0) {
    options.audioBitsPerSecond = audioBitsPerSecond;
  }
  const total = (videoBitsPerSecond > 0 ? videoBitsPerSecond : 0) + (audioBitsPerSecond > 0 ? audioBitsPerSecond : 0);
  if (total > 0) {
    options.bitsPerSecond = total;
  }
  return options;
}

function buildRecorderOptionAttempts(mimeType, qualityOptions) {
  const attempts = [];
  const videoBits = Math.round(Number(qualityOptions?.videoBitsPerSecond) || 0);
  const audioBits = Math.round(Number(qualityOptions?.audioBitsPerSecond) || 0);
  if (videoBits > 0 || audioBits > 0) {
    const qualityAttempt = { mimeType };
    if (videoBits > 0) {
      qualityAttempt.videoBitsPerSecond = videoBits;
    }
    if (audioBits > 0) {
      qualityAttempt.audioBitsPerSecond = audioBits;
    }
    const total = videoBits + audioBits;
    if (total > 0) {
      qualityAttempt.bitsPerSecond = total;
    }
    attempts.push(qualityAttempt);
  }
  attempts.push({ mimeType });
  return attempts;
}

function resolveVideoFrameRate(tracks, fallback = 60) {
  if (!Array.isArray(tracks)) {
    return fallback;
  }
  for (const track of tracks) {
    if (track && typeof track.getSettings === 'function') {
      const settings = track.getSettings();
      const candidate = Number(settings?.frameRate);
      if (Number.isFinite(candidate) && candidate > 0) {
        return candidate;
      }
    }
  }
  return fallback;
}

function resolveAudioCaptureStream(audioElement) {
  if (!audioElement) {
    return null;
  }
  if (typeof audioElement.captureStream === 'function') {
    try {
      return audioElement.captureStream();
    } catch (error) {
      console.warn('[video-export] captureStream failed', error);
    }
  }
  if (typeof audioElement.mozCaptureStream === 'function') {
    try {
      return audioElement.mozCaptureStream();
    } catch (error) {
      console.warn('[video-export] mozCaptureStream failed', error);
    }
  }
  return null;
}

function buildMimeCandidateList(MediaRecorderClass, hasAudioTrack) {
  const results = [];
  const seen = new Set();

  const appendCandidates = (list, isMp4) => {
    for (const candidate of list) {
      if (!candidate || seen.has(candidate)) {
        continue;
      }
      let supported = true;
      if (MediaRecorderClass && typeof MediaRecorderClass.isTypeSupported === 'function') {
        try {
          supported = MediaRecorderClass.isTypeSupported(candidate);
        } catch {
          supported = false;
        }
      }
      if (supported) {
        seen.add(candidate);
        results.push({ mimeType: candidate, isMp4 });
      }
    }
  };

  const preferredMp4 = hasAudioTrack ? PREFERRED_MP4_MIME_TYPES : PREFERRED_MP4_VIDEO_ONLY;
  appendCandidates(preferredMp4, true);

  const fallbackWebm = hasAudioTrack ? FALLBACK_WEBM_MIME_TYPES : FALLBACK_WEBM_MIME_TYPES.filter((mime) => !/opus/.test(mime));
  appendCandidates(fallbackWebm, false);

  if (results.length === 0) {
    const defaultFallback = hasAudioTrack ? 'video/webm;codecs=vp8,opus' : 'video/webm;codecs=vp8';
    results.push({ mimeType: defaultFallback, isMp4: false });
    if (!seen.has('video/webm')) {
      results.push({ mimeType: 'video/webm', isMp4: false });
    }
  }

  return results;
}

function ensureArrayBuffer(data) {
  if (data instanceof ArrayBuffer) {
    return data;
  }
  if (data instanceof Uint8Array) {
    return data.buffer;
  }
  if (ArrayBuffer.isView(data)) {
    return data.buffer;
  }
  return null;
}

export function createVideoExporter(dependencies = {}) {
  const MediaRecorderClass = typeof dependencies.MediaRecorderClass === 'function' ? dependencies.MediaRecorderClass : getDefaultMediaRecorderClass();
  const createMediaRecorder =
    typeof dependencies.createMediaRecorder === 'function'
      ? dependencies.createMediaRecorder
      : MediaRecorderClass
        ? (stream, options) => new MediaRecorderClass(stream, options)
        : null;
  const createStream =
    typeof dependencies.createStream === 'function' ? dependencies.createStream : getDefaultMediaStreamFactory();
  const createWorker = typeof dependencies.createWorker === 'function' ? dependencies.createWorker : defaultCreateWorker;
  const downloadBlob = typeof dependencies.downloadBlob === 'function' ? dependencies.downloadBlob : defaultDownloadBlob;
  const notifyFallback = typeof dependencies.notify === 'function' ? dependencies.notify : defaultNotify;

  const state = {
    status: STATE_IDLE,
    support: {
      canvasCapture: false,
      audioCapture: false,
      mediaRecorder: Boolean(createMediaRecorder),
      worker: false,
      stream: Boolean(createStream),
      isSupported: false,
    },
    canvas: null,
    audio: null,
    button: null,
    notify: notifyFallback,
    getFileName: null,
    recorder: null,
    chunks: [],
    stream: null,
    canvasStream: null,
    audioStream: null,
    recordingMimeType: '',
    recordingStartedAt: null,
    recordingProducesMp4: false,
    hasAudioTrack: false,
    pendingFileName: '',
    cancelRecording: false,
    recorderQuality: null,
    worker: null,
    workerReady: false,
    pendingJobId: '',
    processingProgress: 0,
    pendingMimeCandidates: [],
  };

  function resetRecordingState() {
    state.chunks = [];
    state.recordingMimeType = '';
    state.recordingStartedAt = null;
    state.recordingProducesMp4 = false;
    state.hasAudioTrack = false;
    state.pendingFileName = '';
    state.pendingJobId = '';
    state.processingProgress = 0;
    state.cancelRecording = false;
    state.recorderQuality = null;
    state.pendingMimeCandidates = [];
  }

  function cleanupStreams() {
    const streams = [state.stream, state.canvasStream, state.audioStream];
    streams.forEach((entry) => {
      if (!entry || typeof entry.getTracks !== 'function') {
        return;
      }
      entry.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch {
          // Ignore errors stopping tracks.
        }
      });
    });
    state.stream = null;
    state.canvasStream = null;
    state.audioStream = null;
  }

  function updateButtonUi() {
    if (!state.button) {
      return;
    }
    state.button.dataset.state = state.status;
    if (!state.support.isSupported) {
      state.button.disabled = true;
      state.button.setAttribute('aria-disabled', 'true');
      state.button.textContent = 'Export Video';
      state.button.title = 'Video export is not supported on this browser.';
      return;
    }
    if (state.status === STATE_IDLE) {
      state.button.disabled = false;
      state.button.removeAttribute('aria-disabled');
      state.button.textContent = 'Export Video';
      state.button.title = 'Export the current animation as an MP4 download.';
      return;
    }
    if (state.status === STATE_RECORDING) {
      state.button.disabled = false;
      state.button.removeAttribute('aria-disabled');
      state.button.textContent = 'Stop Export';
      state.button.title = 'Stop recording and encode the video.';
      return;
    }
    state.button.disabled = true;
    state.button.setAttribute('aria-disabled', 'true');
    const progressLabel = Number.isFinite(state.processingProgress) && state.processingProgress > 0
      ? `Processing ${Math.round(state.processingProgress * 100)}%`
      : 'Processing…';
    state.button.textContent = progressLabel;
    state.button.title = 'Encoding MP4 in the background…';
  }

  function setStatus(nextStatus) {
    state.status = nextStatus;
    updateButtonUi();
  }

  function handleWorkerMessage(event) {
    const data = event?.data ?? {};
    if (data.type === 'ready') {
      state.workerReady = true;
      return;
    }
    if (data.type === 'progress') {
      const ratioCandidate = typeof data.data?.ratio === 'number' ? data.data.ratio : typeof data.data?.progress === 'number' ? data.data.progress : typeof data.ratio === 'number' ? data.ratio : data.progress;
      if (Number.isFinite(ratioCandidate)) {
        state.processingProgress = clamp01(ratioCandidate);
        updateButtonUi();
      }
      return;
    }
    if (data.type === 'error') {
      console.error('[video-export] Worker error', data.error ?? data.message ?? data);
      state.notify?.('Video export failed during encoding.', { tone: 'error', duration: 6000 });
      cleanupStreams();
      resetRecordingState();
      setStatus(STATE_IDLE);
      return;
    }
    if (data.type === 'result') {
      if (state.status !== STATE_PROCESSING) {
        return;
      }
      if (state.pendingJobId && data.jobId && data.jobId !== state.pendingJobId) {
        return;
      }
      const buffer = ensureArrayBuffer(data.buffer ?? data.data ?? null);
      if (!buffer) {
        console.error('[video-export] Worker produced invalid buffer');
        state.notify?.('Video export failed: received invalid data.', { tone: 'error', duration: 6000 });
        cleanupStreams();
        resetRecordingState();
        setStatus(STATE_IDLE);
        return;
      }
      const mp4Blob = new Blob([buffer], { type: 'video/mp4' });
      downloadBlob(mp4Blob, state.pendingFileName || createDownloadFileName('latent-noise'));
      state.notify?.('Video export ready. Downloading MP4.', { tone: 'success', duration: 6000 });
      cleanupStreams();
      resetRecordingState();
      setStatus(STATE_IDLE);
    }
  }

  function handleWorkerError(event) {
    console.error('[video-export] Worker crashed', event?.message ?? event);
    state.notify?.('Video export worker crashed. Reload and try again.', { tone: 'error', duration: 6000 });
    if (state.worker) {
      try {
        state.worker.terminate?.();
      } catch {
        // Ignore terminate errors.
      }
      state.worker = null;
    }
    cleanupStreams();
    resetRecordingState();
    state.support.worker = false;
    state.support.isSupported = false;
    setStatus(STATE_IDLE);
  }

  function ensureWorker() {
    if (state.worker || !state.support.worker) {
      return state.worker;
    }
    const created = createWorker?.();
    if (!created) {
      state.support.worker = false;
      state.support.isSupported = false;
      updateButtonUi();
      return null;
    }
    created.addEventListener('message', handleWorkerMessage);
    created.addEventListener('error', handleWorkerError);
    try {
      created.postMessage({ type: 'warmup' });
    } catch (error) {
      console.error('[video-export] Failed to warm up worker', error);
    }
    state.worker = created;
    state.support.worker = true;
    return state.worker;
  }

  function handleRecorderData(event) {
    if (!event) {
      return;
    }
    const blob = event.data instanceof Blob ? event.data : null;
    if (blob && blob.size > 0) {
      state.chunks.push(blob);
    }
  }

  function handleRecorderStop() {
    if (!state.recorder) {
      cleanupStreams();
      resetRecordingState();
      setStatus(STATE_IDLE);
      return;
    }
    const recorder = state.recorder;
    recorder.removeEventListener?.('dataavailable', handleRecorderData);
    recorder.removeEventListener?.('stop', handleRecorderStop);
    recorder.removeEventListener?.('error', handleRecorderError);
    state.recorder = null;

    cleanupStreams();

    if (state.cancelRecording) {
      state.notify?.('Video export cancelled.', { tone: 'info', duration: 4000 });
      resetRecordingState();
      setStatus(STATE_IDLE);
      return;
    }

    if (state.chunks.length === 0) {
      state.notify?.('Nothing was recorded. Start playback before exporting.', { tone: 'warning', duration: 5000 });
      resetRecordingState();
      setStatus(STATE_IDLE);
      return;
    }

    const recordedBlob = new Blob(state.chunks, { type: state.recordingMimeType || state.chunks[0]?.type || 'video/webm' });
    if (state.recordingProducesMp4) {
      downloadBlob(recordedBlob, state.pendingFileName || createDownloadFileName('latent-noise'));
      state.notify?.('Video export ready. Downloading MP4.', { tone: 'success', duration: 6000 });
      resetRecordingState();
      setStatus(STATE_IDLE);
      return;
    }

    const worker = ensureWorker();
    if (!worker) {
      state.notify?.('MP4 conversion is not available in this browser.', { tone: 'error', duration: 6000 });
      resetRecordingState();
      setStatus(STATE_IDLE);
      return;
    }

    setStatus(STATE_PROCESSING);
    state.processingProgress = 0;

    recordedBlob
      .arrayBuffer()
      .then((buffer) => {
        const jobId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        state.pendingJobId = jobId;
        try {
          worker.postMessage(
            {
              type: 'convert',
              jobId,
              buffer,
              sourceType: recordedBlob.type || state.recordingMimeType || 'video/webm',
              hasAudio: state.hasAudioTrack,
            },
            [buffer],
          );
        } catch (error) {
          console.error('[video-export] Failed to post convert job', error);
          state.notify?.('Video export failed during conversion.', { tone: 'error', duration: 6000 });
          resetRecordingState();
          setStatus(STATE_IDLE);
        }
      })
      .catch((error) => {
        console.error('[video-export] Failed to read recording buffer', error);
        state.notify?.('Video export failed while reading recording data.', { tone: 'error', duration: 6000 });
        resetRecordingState();
        setStatus(STATE_IDLE);
      });
  }

  function handleRecorderError(event) {
    console.error('[video-export] Recorder error', event);
    const fallbackCandidates = Array.isArray(state.pendingMimeCandidates)
      ? state.pendingMimeCandidates.filter((candidate) => candidate && !candidate.isMp4)
      : [];

    if (state.recorder) {
      state.recorder.removeEventListener?.('dataavailable', handleRecorderData);
      state.recorder.removeEventListener?.('stop', handleRecorderStop);
      state.recorder.removeEventListener?.('error', handleRecorderError);
    }
    state.recorder = null;

    cleanupStreams();
    resetRecordingState();
    setStatus(STATE_IDLE);

    if (fallbackCandidates.length > 0) {
      state.notify?.('Primary encoder failed; retrying with WebM fallback...', { tone: 'warning', duration: 5000 });
      state.pendingMimeCandidates = [];
      setTimeout(() => {
        startRecording({ mimeCandidates: fallbackCandidates });
      }, 0);
      return;
    }

    state.notify?.('Video export failed. Check console for details.', { tone: 'error', duration: 6000 });
  }

  function ensurePlayback() {
    if (!state.audio || !state.audio.paused) {
      return;
    }
    try {
      const playResult = state.audio.play();
      if (playResult && typeof playResult.catch === 'function') {
        playResult.catch(() => {
          // Ignore autoplay rejection.
        });
      }
    } catch {
      // Ignore playback errors.
    }
  }

  function tryStartRecorder(combinedStream, candidates, baseTitle) {
    for (let index = 0; index < candidates.length; index += 1) {
      const candidate = candidates[index];
      if (!candidate || !candidate.mimeType) {
        continue;
      }

      const optionAttempts = buildRecorderOptionAttempts(candidate.mimeType, state.recorderQuality);
      let recorder = null;
      let usedOptions = null;

      for (let attemptIndex = 0; attemptIndex < optionAttempts.length; attemptIndex += 1) {
        const attemptOptions = optionAttempts[attemptIndex];
        try {
          recorder = createMediaRecorder(combinedStream, attemptOptions);
          usedOptions = attemptOptions;
          break;
        } catch (error) {
          console.warn(
            `[video-export] Failed to create MediaRecorder for ${candidate.mimeType} (attempt ${attemptIndex + 1})`,
            error,
          );
          recorder = null;
        }
      }

      if (!recorder) {
        continue;
      }

      const removeListeners = () => {
        recorder?.removeEventListener?.('dataavailable', handleRecorderData);
        recorder?.removeEventListener?.('stop', handleRecorderStop);
        recorder?.removeEventListener?.('error', handleRecorderError);
      };

      recorder?.addEventListener?.('dataavailable', handleRecorderData);
      recorder?.addEventListener?.('stop', handleRecorderStop);
      recorder?.addEventListener?.('error', handleRecorderError);

      state.recorder = recorder;
      state.recordingMimeType = candidate.mimeType;
      state.recordingProducesMp4 = Boolean(candidate.isMp4);
      state.pendingMimeCandidates = candidates.slice(index + 1);
      state.chunks = [];
      state.cancelRecording = false;
      state.recorderQuality = usedOptions || null;

      try {
        recorder.start();
      } catch (error) {
        console.warn(`[video-export] Failed to start MediaRecorder with ${candidate.mimeType}`, error);
        removeListeners();
        state.recorder = null;
        state.recordingMimeType = '';
        state.recordingProducesMp4 = false;
        continue;
      }

      state.recordingStartedAt = new Date();
      state.pendingFileName = createDownloadFileName(baseTitle, state.recordingStartedAt);
      return true;
    }

    state.recorder = null;
    state.recordingMimeType = '';
    state.recordingProducesMp4 = false;
    state.pendingMimeCandidates = [];
    state.recorderQuality = null;
    return false;
  }

  function startRecording(options = {}) {
    if (!state.support.isSupported || !createMediaRecorder || !createStream || !state.canvas) {
      state.notify?.('Video export is not supported on this browser.', { tone: 'warning', duration: 5000 });
      return false;
    }
    if (state.status !== STATE_IDLE) {
      return false;
    }

    const canvasStream = typeof state.canvas.captureStream === 'function' ? state.canvas.captureStream(60) : null;
    if (!canvasStream) {
      state.notify?.('Canvas capture is not supported.', { tone: 'error', duration: 6000 });
      return false;
    }
    state.canvasStream = canvasStream;

    const audioStream = resolveAudioCaptureStream(state.audio);
    if (audioStream) {
      state.audioStream = audioStream;
    } else if (state.support.audioCapture) {
      state.notify?.('Audio stream capture failed; export will be silent.', { tone: 'warning', duration: 5000 });
    }

    const combinedStream = createStream();
    if (!combinedStream || typeof combinedStream.addTrack !== 'function') {
      state.notify?.('Unable to create combined media stream for export.', { tone: 'error', duration: 6000 });
      cleanupStreams();
      return false;
    }

    const videoTracks = typeof canvasStream.getVideoTracks === 'function' ? canvasStream.getVideoTracks() : canvasStream.getTracks?.() ?? [];
    videoTracks.forEach((track) => {
      try {
        combinedStream.addTrack(track);
      } catch (error) {
        console.warn('[video-export] Failed to add video track', error);
      }
    });

    const audioTracks = audioStream && typeof audioStream.getAudioTracks === 'function' ? audioStream.getAudioTracks() : [];
    state.hasAudioTrack = audioTracks.length > 0;
    audioTracks.forEach((track) => {
      try {
        combinedStream.addTrack(track);
      } catch (error) {
        console.warn('[video-export] Failed to add audio track', error);
      }
    });

    state.stream = combinedStream;

    const hasAudio = audioTracks.length > 0;
    const candidates = Array.isArray(options.mimeCandidates) && options.mimeCandidates.length > 0
      ? options.mimeCandidates
      : buildMimeCandidateList(MediaRecorderClass, hasAudio);

    if (!candidates || candidates.length === 0) {
      state.notify?.('Video export could not find a supported recording format.', { tone: 'error', duration: 6000 });
      cleanupStreams();
      resetRecordingState();
      return false;
    }

    const baseTitle = typeof state.getFileName === 'function' ? state.getFileName() : 'latent-noise';
    const frameRate = resolveVideoFrameRate(videoTracks, 60);
    state.recorderQuality = createRecorderQualityOptions(state.canvas, hasAudio, frameRate);
    const started = tryStartRecorder(combinedStream, candidates, baseTitle);

    if (!started) {
      cleanupStreams();
      resetRecordingState();
      state.notify?.('Video export could not start recording with available encoders.', { tone: 'error', duration: 6000 });
      return false;
    }

    ensurePlayback();
    state.notify?.('Recording started. Click again to finish export.', { tone: 'info', duration: 5000 });
    setStatus(STATE_RECORDING);
    return true;
  }

  function stopRecording() {
    if (state.status !== STATE_RECORDING || !state.recorder) {
      return false;
    }
    state.cancelRecording = false;
    state.notify?.('Finalizing video export…', { tone: 'info', duration: 5000 });
    try {
      state.recorder.stop();
    } catch (error) {
      console.error('[video-export] Failed to stop recorder', error);
      state.notify?.('Video export could not stop recording.', { tone: 'error', duration: 6000 });
      cleanupStreams();
      resetRecordingState();
      setStatus(STATE_IDLE);
      return false;
    }
    return true;
  }

  function cancelRecording(options = {}) {
    if (state.status !== STATE_RECORDING || !state.recorder) {
      return false;
    }
    state.cancelRecording = true;
    if (!options.silent) {
      state.notify?.('Cancelling video export…', { tone: 'warning', duration: 4000 });
    }
    try {
      state.recorder.stop();
    } catch (error) {
      console.error('[video-export] Failed to stop recorder during cancel', error);
      cleanupStreams();
      resetRecordingState();
      setStatus(STATE_IDLE);
    }
    return true;
  }

  function handleButtonClick() {
    if (state.status === STATE_IDLE) {
      startRecording();
      return;
    }
    if (state.status === STATE_RECORDING) {
      stopRecording();
    }
  }

  function handleAudioEnded() {
    if (state.status === STATE_RECORDING) {
      stopRecording();
    }
  }

  function evaluateSupport() {
    const canvasCapture = Boolean(state.canvas && typeof state.canvas.captureStream === 'function');
    const audioCapture = Boolean(state.audio && (typeof state.audio.captureStream === 'function' || typeof state.audio.mozCaptureStream === 'function'));
    const workerSupported = Boolean(createWorker);
    const streamSupported = Boolean(createStream);
    const recorderSupported = Boolean(createMediaRecorder);

    state.support = {
      canvasCapture,
      audioCapture,
      mediaRecorder: recorderSupported,
      worker: workerSupported,
      stream: streamSupported,
      isSupported: canvasCapture && recorderSupported && workerSupported && streamSupported,
    };
    return state.support;
  }

  function init(options = {}) {
    state.canvas = options.canvas ?? state.canvas;
    state.audio = options.audio ?? state.audio;
    state.button = options.button ?? state.button;
    state.notify = typeof options.notify === 'function' ? options.notify : notifyFallback;
    state.getFileName = typeof options.getFileName === 'function' ? options.getFileName : state.getFileName;

    evaluateSupport();

    if (state.support.isSupported) {
      ensureWorker();
    } else {
      updateButtonUi();
    }

    if (state.button && !state.button.dataset.videoExportBound) {
      state.button.addEventListener('click', handleButtonClick);
      state.button.dataset.videoExportBound = 'true';
    }

    if (state.audio && !state.audio.dataset.videoExportBound) {
      state.audio.addEventListener('ended', handleAudioEnded);
      state.audio.dataset.videoExportBound = 'true';
    }

    updateButtonUi();
    return { ...state.support };
  }

  function getState() {
    return {
      status: state.status,
      support: { ...state.support },
      recordingMimeType: state.recordingMimeType,
      recordingProducesMp4: state.recordingProducesMp4,
      recordingQuality: state.recorderQuality ? { ...state.recorderQuality } : null,
    };
  }

  return {
    init,
    start: startRecording,
    stop: stopRecording,
    cancel: cancelRecording,
    getState,
    isSupported: () => state.support.isSupported,
  };
}

const defaultExporter = createVideoExporter();

export const init = defaultExporter.init;
export const start = defaultExporter.start;
export const stop = defaultExporter.stop;
export const cancel = defaultExporter.cancel;
export const getState = defaultExporter.getState;
export const isSupported = defaultExporter.isSupported;
