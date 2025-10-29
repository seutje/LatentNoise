const DB_NAME = 'ln.byom';
const STORE_NAME = 'models';
const DB_VERSION = 2;
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

function augmentReinforcement(entry) {
  if (!entry || typeof entry !== 'object') {
    return entry;
  }
  if (!entry.reinforcement) {
    return { ...entry, reinforcement: normalizeReinforcement(null, entry.id) };
  }
  const normalized = normalizeReinforcement(entry.reinforcement, entry.id);
  if (normalized === entry.reinforcement) {
    return entry;
  }
  return { ...entry, reinforcement: normalized };
}

function fallbackList() {
  return Array.from(MEMORY_STORE.values())
    .map((entry) => augmentReinforcement(entry))
    .sort((a, b) => (a.createdAt || 0) - (b.createdAt || 0));
}

function fallbackGet(id) {
  const entry = MEMORY_STORE.get(id) || null;
  return augmentReinforcement(entry);
}

function fallbackPut(entry) {
  MEMORY_STORE.set(entry.id, augmentReinforcement(entry));
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
    reinforcement: normalizeReinforcement(entry.reinforcement || entry.adaptive || null, entry.id),
  };
  return normalized;
}

function normalizeReinforcementSession(session, entryId) {
  if (!session || typeof session !== 'object') {
    return null;
  }
  const sessionId = typeof session.id === 'string' && session.id ? session.id : `${entryId || 'session'}-${Date.now()}`;
  const startedAt = Number.isFinite(session.startedAt) ? session.startedAt : 0;
  const endedAt = Number.isFinite(session.endedAt) ? session.endedAt : 0;
  const lastRewardAt = Number.isFinite(session.lastRewardAt) ? session.lastRewardAt : 0;
  const batches = Number.isFinite(session.batches) ? Math.max(0, session.batches) : 0;
  const positive = Number.isFinite(session.positive) ? Math.max(0, session.positive) : 0;
  const negative = Number.isFinite(session.negative) ? Math.max(0, session.negative) : 0;
  const total = Number.isFinite(session.total)
    ? Math.max(0, session.total)
    : Math.max(0, positive + negative);
  const version = Number.isFinite(session.modelVersion) ? Math.max(0, session.modelVersion) : 0;
  return {
    id: sessionId,
    startedAt,
    endedAt,
    lastRewardAt,
    batches,
    positive,
    negative,
    total,
    modelVersion: version,
    note: typeof session.note === 'string' ? session.note : '',
  };
}

function normalizeReinforcementVersion(version, entryId) {
  if (!version || typeof version !== 'object') {
    return null;
  }
  const versionNumber = Number.isFinite(version.version) ? Math.max(1, version.version) : 1;
  const updatedAt = Number.isFinite(version.updatedAt) ? version.updatedAt : Date.now();
  let model = null;
  if (version.model) {
    try {
      model = normalizeModel(version.model, {
        ...(typeof version.model?.meta === 'object' ? version.model.meta : {}),
        sessionId: typeof version.sessionId === 'string' ? version.sessionId : undefined,
        entryId,
      });
    } catch (error) {
      console.warn('[byom-storage] Failed to normalize reinforcement model version', error);
      model = null;
    }
  }
  return {
    version: versionNumber,
    updatedAt,
    sessionId: typeof version.sessionId === 'string' ? version.sessionId : '',
    model,
  };
}

function normalizeReinforcement(reinforcement, entryId) {
  const source = reinforcement && typeof reinforcement === 'object' ? reinforcement : {};
  const sessions = Array.isArray(source.sessions)
    ? source.sessions
        .map((session) => normalizeReinforcementSession(session, entryId))
        .filter(Boolean)
    : [];
  const versions = Array.isArray(source.versions)
    ? source.versions
        .map((version) => normalizeReinforcementVersion(version, entryId))
        .filter(Boolean)
    : [];
  const latestSession = sessions.length > 0 ? sessions[sessions.length - 1] : null;
  const latestVersion = versions.length > 0 ? versions[versions.length - 1] : null;
  return {
    sessions,
    versions,
    lastUpdatedAt: Number.isFinite(source.lastUpdatedAt)
      ? source.lastUpdatedAt
      : latestVersion?.updatedAt || latestSession?.endedAt || 0,
    latestVersion: latestVersion?.version || source.latestVersion || 0,
  };
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
      const sorted = rows
        .map((entry) => augmentReinforcement(entry))
        .sort((a, b) => (a.createdAt || 0) - (b.createdAt || 0));
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
      const entry = request.result || null;
      resolve(augmentReinforcement(entry));
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
  reinforcement,
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
    reinforcement,
  });
}

export { STORAGE_PATH };
