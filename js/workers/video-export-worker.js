const chunks = [];
let mimeType = 'video/mp4';
let active = false;

function reset() {
  chunks.length = 0;
  active = false;
}

self.addEventListener('message', (event) => {
  const data = event.data;
  if (!data || typeof data.type !== 'string') {
    return;
  }
  switch (data.type) {
    case 'start': {
      chunks.length = 0;
      mimeType = typeof data.mimeType === 'string' && data.mimeType.length > 0 ? data.mimeType : 'video/mp4';
      active = true;
      break;
    }
    case 'chunk': {
      if (!active) {
        return;
      }
      const payload = data.chunk;
      if (payload instanceof ArrayBuffer) {
        chunks.push(payload);
      } else if (ArrayBuffer.isView(payload)) {
        chunks.push(payload.buffer.slice(payload.byteOffset, payload.byteOffset + payload.byteLength));
      }
      break;
    }
    case 'stop': {
      if (!active) {
        reset();
        return;
      }
      active = false;
      try {
        const blob = new Blob(chunks.slice(), { type: mimeType });
        self.postMessage({ type: 'complete', blob });
      } catch (error) {
        self.postMessage({ type: 'error', message: error?.message ?? String(error) });
      }
      chunks.length = 0;
      break;
    }
    case 'cancel': {
      if (active) {
        reset();
      } else {
        chunks.length = 0;
      }
      self.postMessage({ type: 'cancelled' });
      break;
    }
    default:
      break;
  }
});
