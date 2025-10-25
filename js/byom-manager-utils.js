function isPlainObject(value) {
  return value !== null && typeof value === 'object';
}

export function extractImportEntries(payload) {
  const results = [];
  const queue = [];
  const seenIds = new Set();

  if (Array.isArray(payload)) {
    queue.push(...payload);
  } else if (isPlainObject(payload)) {
    queue.push(payload);
  }

  while (queue.length > 0) {
    const item = queue.shift();
    if (Array.isArray(item)) {
      queue.push(...item);
      continue;
    }
    if (!isPlainObject(item)) {
      continue;
    }

    if (Array.isArray(item.entries)) {
      queue.push(...item.entries);
    }
    if (isPlainObject(item.entriesById)) {
      queue.push(...Object.values(item.entriesById));
    }
    if (isPlainObject(item.entry)) {
      queue.push(item.entry);
    }
    if (Array.isArray(item.models)) {
      queue.push(...item.models);
    }

    const id = typeof item.id === 'string' ? item.id.trim() : '';
    if (!id || seenIds.has(id)) {
      continue;
    }
    const model = isPlainObject(item.model) ? item.model : null;
    if (!model) {
      continue;
    }
    seenIds.add(id);
    results.push(item);
  }

  return results;
}

function slugify(value) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/gi, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/-+/g, '-');
}

export function createExportFileName(entry) {
  const baseName =
    (typeof entry?.name === 'string' && entry.name.trim())
      ? entry.name.trim()
      : typeof entry?.file?.name === 'string' && entry.file.name.trim()
        ? entry.file.name.trim()
        : 'byom-model';
  const slug = slugify(baseName) || 'byom-model';
  const idFragment = typeof entry?.id === 'string'
    ? entry.id.replace(/[^a-z0-9]/gi, '').slice(0, 8).toLowerCase()
    : '';
  const suffix = idFragment ? `-${idFragment}` : '';
  return `${slug}${suffix}.json`;
}

