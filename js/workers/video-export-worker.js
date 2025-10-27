import { FFmpeg } from '../vendor/ffmpeg/classes.js';

const ffmpeg = new FFmpeg();
let loadPromise = null;
let progressHandler = null;

function inferVideoExtension(mimeType) {
  if (typeof mimeType !== 'string') {
    return 'webm';
  }
  const lower = mimeType.toLowerCase();
  if (lower.includes('mp4')) {
    return 'mp4';
  }
  if (lower.includes('webm')) {
    return 'webm';
  }
  if (lower.includes('ogg')) {
    return 'ogg';
  }
  return 'webm';
}

function inferAudioExtension(mimeType) {
  if (typeof mimeType !== 'string') {
    return 'mp3';
  }
  const lower = mimeType.toLowerCase();
  if (lower.includes('mp4') || lower.includes('m4a')) {
    return 'm4a';
  }
  if (lower.includes('aac')) {
    return 'aac';
  }
  if (lower.includes('wav')) {
    return 'wav';
  }
  if (lower.includes('ogg')) {
    return 'ogg';
  }
  if (lower.includes('flac')) {
    return 'flac';
  }
  if (lower.includes('mp3') || lower.includes('mpeg')) {
    return 'mp3';
  }
  return 'mp3';
}

function resolveAsset(path) {
  return new URL(path, import.meta.url).href;
}

function ensureLoaded() {
  if (ffmpeg.loaded) {
    return Promise.resolve();
  }
  if (!loadPromise) {
    const coreURL = resolveAsset('../../assets/ffmpeg/ffmpeg-core.js');
    const wasmURL = resolveAsset('../../assets/ffmpeg/ffmpeg-core.wasm');
    loadPromise = ffmpeg
      .load({ coreURL, wasmURL })
      .then(() => {
        if (!progressHandler) {
          progressHandler = (data) => {
            self.postMessage({ type: 'progress', data });
          };
          ffmpeg.on('progress', progressHandler);
        }
      })
      .catch((error) => {
        loadPromise = null;
        throw error;
      });
  }
  return loadPromise;
}

async function convertToMp4({
  jobId,
  buffer,
  sourceType,
  hasAudio,
  externalAudioBuffer,
  externalAudioType,
  preferCopyVideo,
}) {
  await ensureLoaded();
  const videoExt = inferVideoExtension(sourceType);
  const inputName = jobId ? `input-${jobId}.${videoExt}` : `input.${videoExt}`;
  const outputName = jobId ? `output-${jobId}.mp4` : 'output.mp4';
  const inputData = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  await ffmpeg.writeFile(inputName, inputData);

  let audioName = '';
  if (externalAudioBuffer) {
    const audioExt = inferAudioExtension(externalAudioType);
    audioName = jobId ? `audio-${jobId}.${audioExt}` : `audio.${audioExt}`;
    const audioData = externalAudioBuffer instanceof Uint8Array ? externalAudioBuffer : new Uint8Array(externalAudioBuffer);
    await ffmpeg.writeFile(audioName, audioData);
  }

  const args = ['-i', inputName];
  if (audioName) {
    args.push('-i', audioName);
  }

  const isMp4Input = typeof sourceType === 'string' && sourceType.toLowerCase().includes('mp4');
  const useCopyVideo = Boolean(preferCopyVideo) && isMp4Input;

  args.push('-map', '0:v:0');
  if (audioName) {
    args.push('-map', '1:a:0');
  } else if (hasAudio) {
    args.push('-map', '0:a:0');
  }

  if (useCopyVideo) {
    args.push('-c:v', 'copy');
  } else {
    args.push('-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p');
  }

  if (audioName) {
    args.push('-c:a', 'aac', '-b:a', '192k', '-shortest');
  } else if (hasAudio) {
    if (useCopyVideo) {
      args.push('-c:a', 'copy');
    } else {
      args.push('-c:a', 'aac', '-b:a', '192k');
    }
  } else {
    args.push('-an');
  }

  args.push('-movflags', '+faststart', outputName);

  const code = await ffmpeg.exec(args);
  if (typeof code === 'number' && code !== 0) {
    throw new Error(`ffmpeg exited with code ${code}`);
  }

  const outputData = await ffmpeg.readFile(outputName);
  try {
    await ffmpeg.deleteFile(inputName);
  } catch {
    // Ignore cleanup errors.
  }
  try {
    await ffmpeg.deleteFile(outputName);
  } catch {
    // Ignore cleanup errors.
  }
  if (audioName) {
    try {
      await ffmpeg.deleteFile(audioName);
    } catch {
      // Ignore cleanup errors.
    }
  }
  return outputData;
}

self.onmessage = async (event) => {
  const data = event?.data ?? {};
  if (data.type === 'warmup') {
    try {
      await ensureLoaded();
      self.postMessage({ type: 'ready' });
    } catch (error) {
      self.postMessage({ type: 'error', error: { message: error?.message ?? String(error) } });
    }
    return;
  }
  if (data.type === 'convert') {
    try {
      const result = await convertToMp4({
        jobId: data.jobId,
        buffer: data.buffer,
        sourceType: data.sourceType,
        hasAudio: Boolean(data.hasAudio),
        externalAudioBuffer: data.externalAudioBuffer,
        externalAudioType: data.externalAudioType,
        preferCopyVideo: Boolean(data.preferCopyVideo),
      });
      self.postMessage({ type: 'result', jobId: data.jobId, buffer: result.buffer }, [result.buffer]);
    } catch (error) {
      self.postMessage({ type: 'error', error: { message: error?.message ?? String(error) } });
    }
  }
};
