import { notify as notifyDefault } from './notifications.js';

const DEFAULT_FRAME_RATE = 60;
const DEFAULT_BITRATE = 8_000_000;
const KEYFRAME_INTERVAL = 120;

const state = {
  initialized: false,
  canvas: /** @type {HTMLCanvasElement|null} */ (null),
  button: /** @type {HTMLButtonElement|null} */ (null),
  downloadLink: /** @type {HTMLAnchorElement|null} */ (null),
  render: null,
  unsubscribeFrame: /** @type {(() => void)|null} */ (null),
  unsubscribeResolution: /** @type {(() => void)|null} */ (null),
  notify: /** @type {(message: string, options?: { tone?: string, duration?: number }) => void} */ (notifyDefault),
  workerFactory: null,
  worker: /** @type {Worker|null} */ (null),
  recording: false,
  finalizing: false,
  ready: false,
  capturing: false,
  frameRate: DEFAULT_FRAME_RATE,
  bitrate: DEFAULT_BITRATE,
  keyframeInterval: KEYFRAME_INTERVAL,
  frameIndex: 0,
  nextCaptureAt: 0,
  captureWidth: 0,
  captureHeight: 0,
  downloadUrl: '',
};

function defaultWorkerFactory() {
  if (typeof Worker === 'undefined') {
    return null;
  }
  try {
    return new Worker(new URL('./workers/video-export-worker.js', import.meta.url), { type: 'module' });
  } catch (error) {
    console.warn('[video-export] failed to create worker', error);
    return null;
  }
}

function assertCanvas(element) {
  if (!element || !(element instanceof HTMLCanvasElement)) {
    return null;
  }
  return element;
}

function assertButton(element) {
  if (!element || !(element instanceof HTMLButtonElement)) {
    return null;
  }
  return element;
}

function assertAnchor(element) {
  if (!element || !(element instanceof HTMLAnchorElement)) {
    return null;
  }
  return element;
}

function cleanupWorker() {
  if (state.worker) {
    try {
      state.worker.terminate();
    } catch (error) {
      console.warn('[video-export] failed to terminate worker', error);
    }
  }
  state.worker = null;
  state.ready = false;
}

function revokeDownloadUrl() {
  if (state.downloadUrl) {
    try {
      URL.revokeObjectURL(state.downloadUrl);
    } catch (error) {
      console.warn('[video-export] failed to revoke object URL', error);
    }
    state.downloadUrl = '';
  }
}

function updateDownloadLink(url, filename = 'latent-noise-export.mp4') {
  revokeDownloadUrl();
  if (!state.downloadLink) {
    state.downloadUrl = url || '';
    return;
  }
  if (url) {
    state.downloadUrl = url;
    state.downloadLink.href = url;
    state.downloadLink.download = filename;
    state.downloadLink.textContent = 'Download MP4';
    state.downloadLink.hidden = false;
    state.downloadLink.setAttribute('aria-hidden', 'false');
  } else {
    state.downloadLink.removeAttribute('href');
    state.downloadLink.hidden = true;
    state.downloadLink.setAttribute('aria-hidden', 'true');
  }
}

function updateButtonUi() {
  if (!state.button) {
    return;
  }
  if (state.finalizing) {
    state.button.textContent = 'Finalizing…';
    state.button.disabled = true;
    state.button.setAttribute('aria-pressed', 'true');
    state.button.dataset.state = 'finalizing';
    return;
  }
  if (state.recording) {
    state.button.textContent = 'Stop Export';
    state.button.disabled = false;
    state.button.setAttribute('aria-pressed', 'true');
    state.button.dataset.state = 'recording';
    return;
  }
  state.button.textContent = 'Export MP4';
  state.button.disabled = false;
  state.button.setAttribute('aria-pressed', 'false');
  state.button.dataset.state = 'idle';
}

function handleWorkerMessage(event) {
  const { data } = event;
  if (!data || typeof data.type !== 'string') {
    return;
  }
  if (data.type === 'started') {
    state.ready = true;
    return;
  }
  if (data.type === 'error') {
    abortRecording(typeof data.message === 'string' ? data.message : 'Video export failed.');
    return;
  }
  if (data.type === 'aborted') {
    state.recording = false;
    state.finalizing = false;
    cleanupWorker();
    updateButtonUi();
    return;
  }
  if (data.type === 'done') {
    const buffer = data.buffer instanceof ArrayBuffer ? data.buffer : null;
    let url = '';
    if (buffer) {
      try {
        const blob = new Blob([buffer], { type: 'video/mp4' });
        url = URL.createObjectURL(blob);
      } catch (error) {
        console.error('[video-export] failed to build MP4 blob', error);
      }
    }
    cleanupWorker();
    state.recording = false;
    state.finalizing = false;
    updateButtonUi();
    if (url) {
      updateDownloadLink(url);
      if (state.notify) {
        state.notify('Video export ready. Download will begin shortly.');
      }
      triggerDownload(url);
    } else {
      updateDownloadLink(null);
      if (state.notify) {
        state.notify('Video export completed, but no data was produced.', { tone: 'error' });
      }
    }
  }
}

function triggerDownload(url) {
  if (!url) {
    return;
  }
  if (state.downloadLink && !state.downloadLink.hidden) {
    try {
      state.downloadLink.click();
      return;
    } catch (error) {
      console.warn('[video-export] automatic download failed', error);
    }
  }
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'latent-noise-export.mp4';
  anchor.style.display = 'none';
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
}

function handleWorkerError(event) {
  const message = event?.message || 'Video export worker error.';
  abortRecording(message);
}

function startWorker(width, height) {
  if (!state.workerFactory) {
    state.workerFactory = defaultWorkerFactory;
  }
  const factory = state.workerFactory;
  if (typeof factory !== 'function') {
    return null;
  }
  const worker = factory();
  if (!worker) {
    return null;
  }
  if (typeof worker.addEventListener === 'function') {
    worker.addEventListener('message', handleWorkerMessage);
    worker.addEventListener('error', handleWorkerError);
  } else {
    worker.onmessage = handleWorkerMessage;
    worker.onerror = handleWorkerError;
  }
  worker.postMessage({
    type: 'start',
    width,
    height,
    frameRate: state.frameRate,
    bitrate: state.bitrate,
    keyInterval: state.keyframeInterval,
  });
  return worker;
}

function abortRecording(message) {
  const wasActive = state.recording || state.finalizing;
  const errorMessage = message || 'Video export failed.';
  if (state.worker) {
    try {
      state.worker.postMessage({ type: 'abort' });
    } catch (error) {
      console.warn('[video-export] failed to notify worker about abort', error);
    }
  }
  cleanupWorker();
  state.recording = false;
  state.finalizing = false;
  state.capturing = false;
  updateDownloadLink(null);
  updateButtonUi();
  if (wasActive && state.notify) {
    state.notify(errorMessage, { tone: 'error' });
  }
}

async function captureFrame() {
  if (!state.canvas || !state.worker || !state.recording || state.finalizing) {
    return;
  }
  if (state.capturing) {
    return;
  }
  if (state.captureWidth !== state.canvas.width || state.captureHeight !== state.canvas.height) {
    abortRecording('Canvas resolution changed during export. Recording stopped.');
    return;
  }
  state.capturing = true;
  const frameNumber = state.frameIndex;
  state.frameIndex += 1;
  const captureTimestampUs = Math.round((frameNumber * 1e6) / state.frameRate);
  try {
    const bitmap = await createImageBitmap(state.canvas);
    if (!state.recording || !state.worker) {
      bitmap.close();
      state.capturing = false;
      return;
    }
    const keyFrame = frameNumber % state.keyframeInterval === 0;
    state.worker.postMessage(
      {
        type: 'frame',
        bitmap,
        timestamp: captureTimestampUs,
        frameIndex: frameNumber,
        keyFrame,
      },
      [bitmap],
    );
  } catch (error) {
    console.error('[video-export] failed to capture frame', error);
    abortRecording('Unable to capture video frame. Export cancelled.');
  } finally {
    state.capturing = false;
  }
}

function handleFrameEvent(detail) {
  if (!state.recording || state.finalizing) {
    return;
  }
  if (!state.ready) {
    return;
  }
  if (!detail || typeof detail.timestamp !== 'number') {
    return;
  }
  if (detail.timestamp < state.nextCaptureAt) {
    return;
  }
  state.nextCaptureAt = detail.timestamp + 1000 / state.frameRate;
  captureFrame();
}

function handleResolutionChange() {
  if (!state.recording) {
    return;
  }
  abortRecording('Rendering resolution changed. Video export stopped.');
}

function handleToggle(event) {
  if (event && typeof event.preventDefault === 'function') {
    event.preventDefault();
  }
  if (!state.recording && !state.finalizing) {
    startRecording();
  } else if (state.recording) {
    stopRecording();
  }
}

function ensureSupport() {
  if (typeof createImageBitmap !== 'function') {
    return false;
  }
  if (typeof URL === 'undefined' || typeof URL.createObjectURL !== 'function') {
    return false;
  }
  if (typeof document === 'undefined') {
    return false;
  }
  return true;
}

function startRecording() {
  if (state.recording || state.finalizing) {
    return;
  }
  if (!state.canvas || !state.button) {
    return;
  }
  if (!ensureSupport()) {
    if (state.notify) {
      state.notify('Video export is not supported in this browser.', { tone: 'error' });
    }
    return;
  }
  const width = state.canvas.width;
  const height = state.canvas.height;
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    if (state.notify) {
      state.notify('Unable to start video export due to invalid canvas size.', { tone: 'error' });
    }
    return;
  }
  const worker = startWorker(width, height);
  if (!worker) {
    if (state.notify) {
      state.notify('Video export worker is unavailable.', { tone: 'error' });
    }
    return;
  }
  updateDownloadLink(null);
  state.worker = worker;
  state.recording = true;
  state.finalizing = false;
  state.capturing = false;
  state.ready = false;
  state.frameIndex = 0;
  state.nextCaptureAt = 0;
  state.captureWidth = width;
  state.captureHeight = height;
  updateButtonUi();
  if (state.notify) {
    state.notify('Video export started. Recording animation frames…', { tone: 'info', duration: 6000 });
  }
}

function stopRecording() {
  if (!state.recording || state.finalizing) {
    return;
  }
  state.recording = false;
  state.finalizing = true;
  state.capturing = false;
  updateButtonUi();
  if (state.worker) {
    try {
      state.worker.postMessage({ type: 'stop' });
    } catch (error) {
      console.warn('[video-export] failed to request finalization', error);
    }
  }
  if (state.notify) {
    state.notify('Finalizing video export…', { tone: 'info', duration: 4000 });
  }
}

export function configure(options = {}) {
  if (typeof options.frameRate === 'number' && options.frameRate > 5) {
    state.frameRate = options.frameRate;
  }
  if (typeof options.bitrate === 'number' && options.bitrate > 0) {
    state.bitrate = options.bitrate;
  }
  if (typeof options.keyframeInterval === 'number' && options.keyframeInterval > 0) {
    state.keyframeInterval = Math.max(1, Math.floor(options.keyframeInterval));
  }
  if (typeof options.createWorker === 'function') {
    state.workerFactory = options.createWorker;
  }
}

export function init(options = {}) {
  if (state.initialized) {
    return true;
  }
  const canvas = assertCanvas(options.canvas || document.getElementById('c'));
  const button = assertButton(options.button || document.getElementById('export-video'));
  const downloadLink = assertAnchor(options.downloadLink || document.getElementById('export-video-download'));
  const renderSource = options.render || null;
  const notifier = typeof options.notify === 'function' ? options.notify : notifyDefault;

  if (!canvas || !button || !renderSource || typeof renderSource.on !== 'function') {
    console.warn('[video-export] initialization failed: missing dependencies');
    if (button) {
      button.disabled = true;
      button.textContent = 'Export Unavailable';
    }
    return false;
  }

  if (!ensureSupport()) {
    button.disabled = true;
    button.textContent = 'Export Unavailable';
    return false;
  }

  state.canvas = canvas;
  state.button = button;
  state.downloadLink = downloadLink;
  state.render = renderSource;
  state.notify = notifier;
  state.workerFactory = state.workerFactory || defaultWorkerFactory;

  button.addEventListener('click', handleToggle);

  if (typeof renderSource.on === 'function') {
    state.unsubscribeFrame = renderSource.on('frame', handleFrameEvent);
    state.unsubscribeResolution = renderSource.on('resolutionChange', handleResolutionChange);
  }

  updateDownloadLink(null);
  updateButtonUi();
  state.initialized = true;
  return true;
}

export function teardown() {
  if (!state.initialized) {
    return;
  }
  if (state.unsubscribeFrame) {
    state.unsubscribeFrame();
    state.unsubscribeFrame = null;
  }
  if (state.unsubscribeResolution) {
    state.unsubscribeResolution();
    state.unsubscribeResolution = null;
  }
  if (state.button) {
    state.button.removeEventListener('click', handleToggle);
  }
  if (state.recording || state.finalizing) {
    abortRecording('Video export stopped.');
  } else {
    cleanupWorker();
    revokeDownloadUrl();
    updateButtonUi();
  }
  state.canvas = null;
  state.button = null;
  state.downloadLink = null;
  state.render = null;
  state.initialized = false;
}

export function isRecording() {
  return state.recording;
}

export function isFinalizing() {
  return state.finalizing;
}

export function __resetForTests() {
  if (state.unsubscribeFrame) {
    state.unsubscribeFrame();
  }
  if (state.unsubscribeResolution) {
    state.unsubscribeResolution();
  }
  revokeDownloadUrl();
  cleanupWorker();
  Object.assign(state, {
    initialized: false,
    canvas: null,
    button: null,
    downloadLink: null,
    render: null,
    unsubscribeFrame: null,
    unsubscribeResolution: null,
    notify: notifyDefault,
    workerFactory: null,
    worker: null,
    recording: false,
    finalizing: false,
    ready: false,
    capturing: false,
    frameRate: DEFAULT_FRAME_RATE,
    bitrate: DEFAULT_BITRATE,
    keyframeInterval: KEYFRAME_INTERVAL,
    frameIndex: 0,
    nextCaptureAt: 0,
    captureWidth: 0,
    captureHeight: 0,
    downloadUrl: '',
  });
}

export default {
  init,
  configure,
  teardown,
  isRecording,
  isFinalizing,
};
