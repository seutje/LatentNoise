import { FFmpeg } from '../vendor/ffmpeg/classes.js';

const ffmpeg = new FFmpeg();
let loadPromise = null;
let progressHandler = null;

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

async function convertToMp4({ jobId, buffer, hasAudio }) {
  await ensureLoaded();
  const inputName = jobId ? `input-${jobId}.webm` : 'input.webm';
  const outputName = jobId ? `output-${jobId}.mp4` : 'output.mp4';
  const inputData = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  await ffmpeg.writeFile(inputName, inputData);

  const args = ['-i', inputName, '-c:v', 'libx264', '-preset', 'medium', '-pix_fmt', 'yuv420p'];
  if (hasAudio) {
    args.push('-c:a', 'aac', '-b:a', '192k');
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
      const result = await convertToMp4({ jobId: data.jobId, buffer: data.buffer, hasAudio: Boolean(data.hasAudio) });
      self.postMessage({ type: 'result', jobId: data.jobId, buffer: result.buffer }, [result.buffer]);
    } catch (error) {
      self.postMessage({ type: 'error', error: { message: error?.message ?? String(error) } });
    }
  }
};

