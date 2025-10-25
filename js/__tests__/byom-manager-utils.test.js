import { extractImportEntries, createExportFileName } from '../byom-manager-utils.js';

describe('extractImportEntries', () => {
  it('returns single entry objects', () => {
    const entry = { id: 'alpha', model: { layers: [] } };
    const result = extractImportEntries(entry);
    expect(result).toHaveLength(1);
    expect(result[0]).toBe(entry);
  });

  it('extracts entries from wrapper structures', () => {
    const entry = { id: 'wrap', model: { layers: [] } };
    const payload = { storage: 'ln.byom.models', entry };
    const result = extractImportEntries(payload);
    expect(result).toHaveLength(1);
    expect(result[0]).toBe(entry);
  });

  it('flattens nested arrays and maps while deduplicating', () => {
    const first = { id: 'one', model: { layers: [] } };
    const duplicate = { id: 'one', model: { layers: [] } };
    const second = { id: 'two', model: { layers: [] } };
    const payload = {
      entries: [first, { entriesById: { b: duplicate } }, [second]],
    };
    const result = extractImportEntries(payload);
    expect(result).toHaveLength(2);
    const ids = result.map((entry) => entry.id).sort();
    expect(ids).toEqual(['one', 'two']);
  });

  it('ignores items without id or model definitions', () => {
    const payload = [{ id: 'no-model' }, { model: {} }, 'string'];
    const result = extractImportEntries(payload);
    expect(result).toHaveLength(0);
  });
});

describe('createExportFileName', () => {
  it('slugifies the provided name and appends id fragment', () => {
    const filename = createExportFileName({ id: 'ABC123XYZ', name: 'My Cool Model' });
    expect(filename).toBe('my-cool-model-abc123xy.json');
  });

  it('falls back to file name and default slug when necessary', () => {
    const filename = createExportFileName({ id: 'id', file: { name: 'Demo Track.mp3' } });
    expect(filename).toBe('demo-track-mp3-id.json');
  });

  it('provides a stable default when no metadata is present', () => {
    expect(createExportFileName({})).toBe('byom-model.json');
  });
});
