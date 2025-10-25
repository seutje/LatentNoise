const DB_NAME = 'ln.byom';
const STORE_NAME = 'models';
const DB_VERSION = 1;
const MEMORY_STORE = new Map();
const STORAGE_PATH = `${DB_NAME}.${STORE_NAME}`;

let dbPromise = null;
let dbDisabled = false;

function hasIndexedDb() {
  return typeof window !== 'undefined' && 'indexedDB' in window;
}

function createObjectStore(event) {
  const db = event.target.result;
  if (!db.objectStoreNames.contains(STORE_NAME)) {
    const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
    store.createIndex('byCreated', 'createdAt', { unique: false });
    store.createIndex('byUpdated', 'updatedAt', { unique: false });
  }
}

function openDatabase() {
  if (!hasIndexedDb() || dbDisabled) {
    return Promise.resolve(null);
  }
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const request = window.indexedDB.open(DB_NAME, DB_VERSION);
      request.onupgradeneeded = createObjectStore;
      request.onsuccess = () => {
        const db = request.result;
        db.onversionchange = () => {
          db.close();
        };
        resolve(db);
      };
      request.onerror = () => {
        dbDisabled = true;
        reject(request.error || new Error('IndexedDB open failed.'));
      };
      request.onblocked = () => {
        console.warn('[byom-storage] IndexedDB open blocked.');
      };
    }).catch((error) => {
      console.warn('[byom-storage] Disabling IndexedDB due to error', error);
      dbDisabled = true;
      return null;
    });
  }
  return dbPromise;
}

function fallbackList() {
  return Array.from(MEMORY_STORE.values()).sort((a, b) => (a.createdAt || 0) - (b.createdAt || 0));
}

function fallbackGet(id) {
  return MEMORY_STORE.get(id) || null;
}

function fallbackPut(entry) {
  MEMORY_STORE.set(entry.id, entry);
  return entry;
}

function fallbackDelete(id) {
  MEMORY_STORE.delete(id);
  return true;
}

function sanitizeEntry(entry) {
  if (!entry || typeof entry !== 'object') {
    throw new TypeError('Persisted entry must be an object.');
  }
  if (!entry.id) {
    throw new Error('Persisted entry requires an id.');
  }
  const now = Date.now();
  const normalized = {
    version: entry.version || 1,
    id: entry.id,
    name: entry.name?.trim() || 'Untitled Model',
    createdAt: Number.isFinite(entry.createdAt) ? entry.createdAt : now,
    updatedAt: now,
    file: entry.file ? { ...entry.file } : null,
    baseline: entry.baseline ? { ...entry.baseline } : null,
    summary: entry.summary ? { ...entry.summary } : null,
    model: entry.model ? { ...entry.model } : null,
    stats: entry.stats ? { ...entry.stats } : null,
  };
  return normalized;
}

function computeInvStd(stdArray) {
  if (!Array.isArray(stdArray)) {
    return [];
  }
  return stdArray.map((value) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric <= 0) {
      return 1;
    }
    return 1 / numeric;
  });
}

function normalizeLayer(layer) {
  const act = typeof layer.activation === 'string' ? layer.activation : 'linear';
  const cloneArray = (value) => {
    if (Array.isArray(value)) {
      return value.slice();
    }
    if (value instanceof Float32Array || value instanceof Float64Array) {
      return Array.from(value);
    }
    if (value instanceof Int32Array || value instanceof Int16Array || value instanceof Int8Array) {
      return Array.from(value);
    }
    if (value instanceof Uint32Array || value instanceof Uint16Array || value instanceof Uint8Array) {
      return Array.from(value);
    }
    return [];
  };
  const bias = layer.bias !== undefined ? cloneArray(layer.bias) : cloneArray(layer.biases);
  const weights = cloneArray(layer.weights);
  return {
    activation: act,
    act,
    weights,
    bias,
  };
}

function normalizeModel(model, meta = {}) {
  if (!model || typeof model !== 'object') {
    throw new TypeError('Model definition must be an object.');
  }
  const layers = Array.isArray(model.layers) ? model.layers.map(normalizeLayer) : [];
  const input = Number(model.input);
  if (!Number.isFinite(input) || input <= 0) {
    throw new Error('Model definition missing valid input size.');
  }
  const normMean = Array.isArray(model.normalization?.mean)
    ? model.normalization.mean.slice()
    : model.normalization?.mean instanceof Float32Array
      ? Array.from(model.normalization.mean)
      : Array.isArray(model.norm?.mean)
        ? model.norm.mean.slice()
        : model.norm?.mean instanceof Float32Array
          ? Array.from(model.norm.mean)
          : [];
  const normStd = Array.isArray(model.normalization?.std)
    ? model.normalization.std.slice()
    : model.normalization?.std instanceof Float32Array
      ? Array.from(model.normalization.std)
      : Array.isArray(model.norm?.std)
        ? model.norm.std.slice()
        : model.norm?.std instanceof Float32Array
          ? Array.from(model.norm.std)
          : [];
  const metaObject = {
    ...(typeof model.meta === 'object' ? model.meta : {}),
    ...meta,
  };
  return {
    input,
    normalization: {
      mean: normMean,
      std: normStd.length === normMean.length ? normStd : normMean.map(() => 1),
    },
    norm: {
      mean: normMean,
      invStd: computeInvStd(normStd.length === normMean.length ? normStd : normMean.map(() => 1)),
    },
    layers,
    meta: metaObject,
  };
}

export function isSupported() {
  return hasIndexedDb() && !dbDisabled;
}

export async function listEntries() {
  const db = await openDatabase();
  if (!db) {
    return fallbackList();
  }
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const request = store.getAll();
    request.onsuccess = () => {
      const rows = Array.isArray(request.result) ? request.result : [];
      const sorted = rows.sort((a, b) => (a.createdAt || 0) - (b.createdAt || 0));
      resolve(sorted);
    };
    request.onerror = () => {
      console.warn('[byom-storage] getAll failed, switching to fallback', request.error);
      resolve(fallbackList());
    };
    tx.onabort = () => {
      reject(tx.error || new Error('IndexedDB transaction aborted.'));
    };
  });
}

export async function getEntry(id) {
  if (!id) {
    return null;
  }
  const db = await openDatabase();
  if (!db) {
    return fallbackGet(id);
  }
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const request = store.get(id);
    request.onsuccess = () => {
      resolve(request.result || null);
    };
    request.onerror = () => {
      console.warn('[byom-storage] get failed, falling back', request.error);
      resolve(fallbackGet(id));
    };
    tx.onabort = () => {
      reject(tx.error || new Error('IndexedDB transaction aborted.'));
    };
  });
}

export async function putEntry(entry, meta = {}) {
  const normalized = sanitizeEntry(entry);
  if (normalized.model) {
    normalized.model = normalizeModel(normalized.model, meta);
  }
  const db = await openDatabase();
  if (!db) {
    fallbackPut(normalized);
    return normalized;
  }
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const request = store.put(normalized);
    request.onsuccess = () => {
      resolve(normalized);
    };
    request.onerror = () => {
      console.warn('[byom-storage] put failed, using fallback', request.error);
      fallbackPut(normalized);
      resolve(normalized);
    };
    tx.onabort = () => {
      reject(tx.error || new Error('IndexedDB transaction aborted.'));
    };
  });
}

export async function deleteEntry(id) {
  const db = await openDatabase();
  if (!db) {
    return fallbackDelete(id);
  }
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const request = store.delete(id);
    request.onsuccess = () => {
      fallbackDelete(id);
      resolve(true);
    };
    request.onerror = () => {
      console.warn('[byom-storage] delete failed, removing from fallback', request.error);
      resolve(fallbackDelete(id));
    };
    tx.onabort = () => {
      reject(tx.error || new Error('IndexedDB transaction aborted.'));
    };
  });
}

export async function updateEntry(id, updates = {}) {
  if (!id) {
    throw new Error('updateEntry requires an id.');
  }
  const existing = await getEntry(id);
  if (!existing) {
    throw new Error(`No BYOM entry found for id ${id}`);
  }
  const merged = sanitizeEntry({ ...existing, ...updates, id, createdAt: existing.createdAt });
  if (updates.model) {
    merged.model = normalizeModel(updates.model, merged.model?.meta);
  }
  const db = await openDatabase();
  if (!db) {
    fallbackPut(merged);
    return merged;
  }
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const request = store.put(merged);
    request.onsuccess = () => {
      fallbackPut(merged);
      resolve(merged);
    };
    request.onerror = () => {
      console.warn('[byom-storage] update failed, using fallback', request.error);
      fallbackPut(merged);
      resolve(merged);
    };
    tx.onabort = () => {
      reject(tx.error || new Error('IndexedDB transaction aborted.'));
    };
  });
}

export async function renameEntry(id, name) {
  const normalized = typeof name === 'string' ? name.trim() : '';
  if (!normalized) {
    throw new Error('Name must be a non-empty string.');
  }
  return updateEntry(id, { name: normalized });
}

export function createEntryPayload({
  id,
  name,
  file,
  baseline,
  summary,
  model,
  stats,
}) {
  const entryId = id || (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `byom-${Date.now()}-${Math.floor(Math.random() * 1e6)}`);
  return sanitizeEntry({
    id: entryId,
    name,
    file,
    baseline,
    summary,
    model: model ? normalizeModel(model, { name: entryId }) : null,
    stats,
  });
}

export { STORAGE_PATH };
