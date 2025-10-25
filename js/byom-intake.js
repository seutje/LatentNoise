import { createFeatureExtractor, FEATURE_COUNT, mixToMono } from './audio-features.js';
import { createModel, infer, loadModelDefinition } from './nn.js';
import { FRESH_MODEL_ID } from './byom-constants.js';
import { PARAM_NAMES as OUTPUT_PARAM_NAMES } from './map.js';

const FRAME_SIZE = 2048;
const TARGET_FPS = 60;
const MIN_DURATION_SECONDS = 30;
const MAX_FILE_BYTES = 45 * 1024 * 1024;
const PROGRESS_IMPORT = 0.05;
const PROGRESS_DECODE = 0.15;
const PROGRESS_FEATURES = 0.8;

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return '0:00';
  }
  const minutes = Math.floor(seconds / 60);
  const remaining = Math.round(seconds - minutes * 60);
  return `${minutes}:${String(remaining).padStart(2, '0')}`;
}

function throwIfAborted(signal) {
  if (signal?.aborted) {
    throw new DOMException('Analysis cancelled', 'AbortError');
  }
}

function bitReverse(index, bits) {
  let reversed = 0;
  for (let i = 0; i < bits; i += 1) {
    reversed = (reversed << 1) | (index & 1);
    index >>= 1;
  }
  return reversed;
}

function fft(real, imag) {
  const n = real.length;
  const bits = Math.log2(n);
  if (!Number.isInteger(bits)) {
    throw new Error('FFT input length must be a power of two.');
  }

  for (let i = 0; i < n; i += 1) {
    const j = bitReverse(i, bits);
    if (j > i) {
      const tmpR = real[i];
      real[i] = real[j];
      real[j] = tmpR;
      const tmpI = imag[i];
      imag[i] = imag[j];
      imag[j] = tmpI;
    }
  }

  for (let size = 2; size <= n; size <<= 1) {
    const halfSize = size >> 1;
    const angleStep = (-2 * Math.PI) / size;
    for (let start = 0; start < n; start += size) {
      for (let offset = 0; offset < halfSize; offset += 1) {
        const index = start + offset;
        const match = index + halfSize;
        const angle = angleStep * offset;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const tre = cos * real[match] - sin * imag[match];
        const tim = sin * real[match] + cos * imag[match];
        real[match] = real[index] - tre;
        imag[match] = imag[index] - tim;
        real[index] += tre;
        imag[index] += tim;
      }
    }
  }
}

function computeSpectrum(frameBuffer, scratchReal, scratchImag, spectrum) {
  scratchReal.set(frameBuffer);
  scratchImag.fill(0);
  fft(scratchReal, scratchImag);
  const n = scratchReal.length;
  const half = spectrum.length;
  const scale = 2 / n;
  spectrum[0] = Math.abs(scratchReal[0]) / n;
  for (let i = 1; i < half; i += 1) {
    const magnitude = Math.hypot(scratchReal[i], scratchImag[i]) * scale;
    spectrum[i] = Number.isFinite(magnitude) ? magnitude : 0;
  }
}

async function maybeYield(iteration) {
  if (iteration % 64 === 0) {
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
  }
}

function buildFrameStarts(totalSamples, hopSamples, frameSize) {
  if (totalSamples <= 0) {
    return [0];
  }
  const starts = [];
  const lastStart = Math.max(0, totalSamples - frameSize);
  for (let start = 0; start < lastStart; start += hopSamples) {
    starts.push(start);
  }
  if (starts.length === 0 || starts[starts.length - 1] !== lastStart) {
    starts.push(lastStart);
  }
  return starts;
}

function makeWarnings({ duration, sizeBytes }) {
  const warnings = [];
  if (Number.isFinite(duration) && duration > 0 && duration < MIN_DURATION_SECONDS) {
    warnings.push(`Audio duration ${formatDuration(duration)} is below the recommended ${MIN_DURATION_SECONDS}s.`);
  }
  if (Number.isFinite(sizeBytes) && sizeBytes > MAX_FILE_BYTES) {
    const mb = (sizeBytes / (1024 * 1024)).toFixed(1);
    const limitMb = (MAX_FILE_BYTES / (1024 * 1024)).toFixed(0);
    warnings.push(`File size ${mb}MB exceeds the ${limitMb}MB threshold and may impact performance.`);
  }
  return warnings;
}

function createDatasetSummary({
  file,
  duration,
  sampleRate,
  channels,
  frameCount,
  hopSamples,
  trainFrames,
  valFrames,
  warnings,
}) {
  return {
    fileName: file?.name ?? 'unknown',
    fileSizeBytes: file?.size ?? 0,
    durationSeconds: duration,
    durationFormatted: formatDuration(duration),
    sampleRate,
    channels,
    frameCount,
    hopSamples,
    hopMs: hopSamples > 0 && sampleRate > 0 ? (hopSamples / sampleRate) * 1000 : 0,
    frameMs: sampleRate > 0 ? (FRAME_SIZE / sampleRate) * 1000 : 0,
    trainFrames,
    validationFrames: valFrames,
    warnings,
  };
}

export async function analyzeFile({
  file,
  presetId,
  modelUrl,
  onProgress,
  signal,
}) {
  if (!(file instanceof File)) {
    throw new TypeError('analyzeFile requires a File instance.');
  }

  throwIfAborted(signal);
  onProgress?.({ stage: 'import', value: PROGRESS_IMPORT });
  const arrayBuffer = await file.arrayBuffer();
  throwIfAborted(signal);

  const audioContext = new AudioContext();
  try {
    onProgress?.({ stage: 'decode', value: PROGRESS_IMPORT + PROGRESS_DECODE * 0.5 });
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    throwIfAborted(signal);

    const channels = audioBuffer.numberOfChannels;
    const channelData = [];
    for (let ch = 0; ch < channels; ch += 1) {
      channelData.push(audioBuffer.getChannelData(ch));
    }
    const mono = mixToMono(channelData);
    const sampleRate = audioBuffer.sampleRate;
    const duration = audioBuffer.duration;
    const hopSamples = Math.max(1, Math.round(sampleRate / TARGET_FPS));
    const frameStarts = buildFrameStarts(mono.length, hopSamples, FRAME_SIZE);
    const frameCount = frameStarts.length;

    const isFreshModel = modelUrl === FRESH_MODEL_ID;
    onProgress?.({ stage: 'model', value: PROGRESS_IMPORT + PROGRESS_DECODE });
    let model = null;
    let outputSize = OUTPUT_PARAM_NAMES.length;
    if (!isFreshModel) {
      const rawModel = await loadModelDefinition(modelUrl);
      model = createModel(rawModel);
      outputSize = model.outputSize;
    }

    const extractor = createFeatureExtractor({ sampleRate, fftSize: FRAME_SIZE });
    const spectrum = new Float32Array(FRAME_SIZE / 2);
    const scratchReal = new Float32Array(FRAME_SIZE);
    const scratchImag = new Float32Array(FRAME_SIZE);
    const frameBuffer = new Float32Array(FRAME_SIZE);
    const featureValues = new Float32Array(frameCount * FEATURE_COUNT);
    const targetValues = new Float32Array(frameCount * outputSize);
    const outputScratch = isFreshModel ? null : new Float32Array(outputSize);

    let previousStart = frameStarts[0];

    for (let index = 0; index < frameCount; index += 1) {
      throwIfAborted(signal);
      const start = frameStarts[index];
      const end = Math.min(mono.length, start + FRAME_SIZE);
      frameBuffer.fill(0);
      frameBuffer.set(mono.subarray(start, end));
      computeSpectrum(frameBuffer, scratchReal, scratchImag, spectrum);

      const deltaSamples = index === 0 ? 0 : Math.max(1, start - previousStart);
      const deltaMs = (deltaSamples / sampleRate) * 1000;
      previousStart = start;
      const trackRatio = duration > 0 ? Math.min(Math.max(start / (sampleRate * duration), 0), 1) : -1;
      const trackPosition = trackRatio >= 0 ? trackRatio * 2 - 1 : -1;

      const features = extractor.process({
        magnitudes: spectrum,
        waveform: frameBuffer,
        deltaMs,
        trackPosition,
        sampleRateOverride: sampleRate,
        fftSizeOverride: FRAME_SIZE,
      });

      featureValues.set(features, index * FEATURE_COUNT);
      if (model) {
        const outputs = infer(model, features, outputScratch);
        targetValues.set(outputs, index * outputSize);
      }

      if ((index & 0x3f) === 0) {
        const progressBase = PROGRESS_IMPORT + PROGRESS_DECODE;
        const fraction = frameCount > 0 ? index / frameCount : 0;
        onProgress?.({ stage: 'features', value: progressBase + PROGRESS_FEATURES * fraction });
        await maybeYield(index);
      }
    }

    onProgress?.({ stage: 'complete', value: 1 });

    let trainFrames = Math.max(1, Math.floor(frameCount * 0.8));
    let valFrames = frameCount - trainFrames;
    if (frameCount > 1 && valFrames === 0) {
      trainFrames -= 1;
      valFrames = 1;
    }
    if (frameCount === 1) {
      trainFrames = 1;
      valFrames = 0;
    }

    const warnings = makeWarnings({ duration, sizeBytes: file.size });

    const summary = createDatasetSummary({
      file,
      duration,
      sampleRate,
      channels,
      frameCount,
      hopSamples,
      trainFrames,
      valFrames,
      warnings,
    });

    return {
      dataset: {
        features: featureValues,
        targets: targetValues,
        frameCount,
        featureSize: FEATURE_COUNT,
        targetSize: outputSize,
        hopSamples,
        sampleRate,
        frameSize: FRAME_SIZE,
        splits: {
          train: { start: 0, count: trainFrames },
          validation: { start: trainFrames, count: valFrames },
        },
        metadata: {
          duration,
          presetId,
          modelUrl,
          sampleRate,
          channels,
          frameStarts,
        },
      },
      summary,
    };
  } finally {
    try {
      await audioContext.close();
    } catch {
      // Ignore errors on close; context may already be closed.
    }
  }
}
