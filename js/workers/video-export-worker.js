// Video export worker encodes canvas frames into MP4 using WebCodecs and a vendored MP4 muxer.
// Relies on mp4-muxer (MIT, see js/vendor/mp4-muxer.js).

import { ArrayBufferTarget, Muxer } from '../vendor/mp4-muxer.js';
import { selectCodec } from '../video-export-codec.js';

const DEFAULT_BITRATE = 8_000_000;
const DEFAULT_AUDIO_CODEC = 'mp4a.40.2';
const DEFAULT_AUDIO_BITRATE = 192_000;

let muxer = null;
let target = null;
let encoder = null;
let configured = false;
let frameRate = 60;
let keyInterval = 60;
let audioEncoder = null;
let audioConfigured = false;

function resetEncoder() {
  if (encoder) {
    try {
      encoder.close();
    } catch (error) {
      console.warn('[video-export-worker] encoder close failed', error);
    }
  }
  if (audioEncoder) {
    try {
      audioEncoder.close();
    } catch (error) {
      console.warn('[video-export-worker] audio encoder close failed', error);
    }
  }
  muxer = null;
  target = null;
  encoder = null;
  audioEncoder = null;
  configured = false;
  audioConfigured = false;
}

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

function handleChunk(chunk, meta) {
  if (!muxer) {
    return;
  }
  try {
    muxer.addVideoChunk(chunk, meta || {});
  } catch (error) {
    post('error', { message: error?.message || 'Failed to mux video chunk.' });
  }
}

function handleError(error) {
  const message = error?.message || error?.name || 'Video encoder error.';
  post('error', { message });
}

function handleAudioChunk(chunk) {
  if (!muxer) {
    return;
  }
  try {
    muxer.addAudioChunk(chunk);
  } catch (error) {
    post('error', { message: error?.message || 'Failed to mux audio chunk.' });
  }
}

function handleAudioError(error) {
  const message = error?.message || error?.name || 'Audio encoder error.';
  post('error', { message });
}

function start(options) {
  if (configured) {
    resetEncoder();
  }
  const { width, height } = options;
  frameRate = Number.isFinite(options.frameRate) && options.frameRate > 0 ? options.frameRate : 60;
  keyInterval = Number.isFinite(options.keyInterval) && options.keyInterval > 0 ? Math.floor(options.keyInterval) : 60;
  const bitrate = Number.isFinite(options.bitrate) && options.bitrate > 0 ? options.bitrate : DEFAULT_BITRATE;
  const codec = selectCodec(width, height, options.codec);
  audioEncoder = null;
  audioConfigured = false;
  let audioTrackConfig = null;
  const audioOptions = options.audio && typeof options.audio === 'object' ? options.audio : null;

  if (typeof VideoEncoder === 'undefined') {
    post('error', { message: 'VideoEncoder API is not available.' });
    return;
  }
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    post('error', { message: 'Invalid canvas dimensions for video export.' });
    return;
  }

  if (audioOptions && typeof AudioEncoder !== 'undefined') {
    const sampleRate = Number.isFinite(audioOptions.sampleRate) && audioOptions.sampleRate > 0
      ? Math.floor(audioOptions.sampleRate)
      : 0;
    const numberOfChannels = Number.isFinite(audioOptions.numberOfChannels) && audioOptions.numberOfChannels > 0
      ? Math.floor(audioOptions.numberOfChannels)
      : 0;
    if (sampleRate > 0 && numberOfChannels > 0) {
      const audioCodec = typeof audioOptions.codec === 'string' && audioOptions.codec.trim().length > 0
        ? audioOptions.codec
        : DEFAULT_AUDIO_CODEC;
      const audioBitrate = Number.isFinite(audioOptions.bitrate) && audioOptions.bitrate > 0
        ? audioOptions.bitrate
        : DEFAULT_AUDIO_BITRATE;
      try {
        audioEncoder = new AudioEncoder({ output: handleAudioChunk, error: handleAudioError });
        audioEncoder.configure({
          codec: audioCodec,
          sampleRate,
          numberOfChannels,
          bitrate: audioBitrate,
        });
        audioConfigured = true;
        audioTrackConfig = {
          codec: 'aac',
          numberOfChannels,
          sampleRate,
        };
      } catch (error) {
        console.warn('[video-export-worker] failed to configure audio encoder', error);
        if (audioEncoder) {
          try {
            audioEncoder.close();
          } catch (closeError) {
            console.warn('[video-export-worker] audio encoder cleanup failed', closeError);
          }
        }
        audioEncoder = null;
        audioConfigured = false;
      }
    }
  }

  target = new ArrayBufferTarget();
  muxer = new Muxer({
    target,
    video: {
      codec: 'avc',
      width: Math.floor(width),
      height: Math.floor(height),
      frameRate,
    },
    audio: audioTrackConfig || undefined,
    fastStart: 'in-memory',
    firstTimestampBehavior: 'offset',
  });

  encoder = new VideoEncoder({ output: handleChunk, error: handleError });
  try {
    encoder.configure({
      codec,
      width: Math.floor(width),
      height: Math.floor(height),
      bitrate,
      framerate: frameRate,
      hardwareAcceleration: 'prefer-hardware',
      latencyMode: 'realtime',
      avc: { format: 'annexb' },
    });
  } catch (error) {
    post('error', { message: error?.message || 'Failed to configure video encoder.' });
    resetEncoder();
    return;
  }

  configured = true;
  post('started');
}

function encodeFrame(data) {
  if (!configured || !encoder) {
    if (data.bitmap) {
      try {
        data.bitmap.close();
      } catch (error) {
        console.warn('[video-export-worker] failed to close bitmap', error);
      }
    }
    return;
  }
  const frameIndex = Number.isFinite(data.frameIndex) ? data.frameIndex : 0;
  const timestamp = Number.isFinite(data.timestamp) ? data.timestamp : Math.round((frameIndex * 1e6) / frameRate);
  let videoFrame = null;
  try {
    videoFrame = new VideoFrame(data.bitmap, { timestamp });
    const keyFrame = frameIndex === 0 || (keyInterval > 0 && frameIndex % keyInterval === 0);
    encoder.encode(videoFrame, { keyFrame: keyFrame || data.keyFrame === true });
  } catch (error) {
    post('error', { message: error?.message || 'Failed to encode frame.' });
  } finally {
    if (videoFrame) {
      try {
        videoFrame.close();
      } catch (error) {
        console.warn('[video-export-worker] failed to close video frame', error);
      }
    }
    if (data.bitmap) {
      try {
        data.bitmap.close();
      } catch (error) {
        console.warn('[video-export-worker] failed to release bitmap', error);
      }
    }
  }
}

function encodeAudio(data) {
  if (!audioConfigured || !audioEncoder) {
    if (data.audioData && typeof data.audioData.close === 'function') {
      try {
        data.audioData.close();
      } catch (error) {
        console.warn('[video-export-worker] failed to close audio data', error);
      }
    }
    return;
  }
  const audioData = data.audioData;
  if (!audioData) {
    return;
  }
  try {
    audioEncoder.encode(audioData);
  } catch (error) {
    post('error', { message: error?.message || 'Failed to encode audio frame.' });
  } finally {
    if (typeof audioData.close === 'function') {
      try {
        audioData.close();
      } catch (error) {
        console.warn('[video-export-worker] failed to release audio frame', error);
      }
    }
  }
}

async function finalize() {
  if (!configured || !encoder || !muxer || !target) {
    resetEncoder();
    post('done', { buffer: new ArrayBuffer(0) });
    return;
  }
  if (audioEncoder) {
    try {
      await audioEncoder.flush();
    } catch (error) {
      post('error', { message: error?.message || 'Failed to flush audio encoder.' });
      resetEncoder();
      return;
    }
  }
  try {
    await encoder.flush();
  } catch (error) {
    post('error', { message: error?.message || 'Failed to flush encoder.' });
    resetEncoder();
    return;
  }
  try {
    muxer.finalize();
  } catch (error) {
    post('error', { message: error?.message || 'Failed to finalize MP4.' });
    resetEncoder();
    return;
  }
  const buffer = target?.buffer;
  resetEncoder();
  if (buffer instanceof ArrayBuffer) {
    post('done', { buffer }, [buffer]);
  } else {
    post('done', { buffer: new ArrayBuffer(0) });
  }
}

function abort() {
  resetEncoder();
  post('aborted');
}

self.addEventListener('message', (event) => {
  const { data } = event;
  if (!data || typeof data.type !== 'string') {
    return;
  }
  switch (data.type) {
    case 'start':
      start(data);
      break;
    case 'frame':
      encodeFrame(data);
      break;
    case 'audio':
      encodeAudio(data);
      break;
    case 'stop':
      finalize();
      break;
    case 'abort':
      abort();
      break;
    default:
      break;
  }
});
