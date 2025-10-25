import { count, getList, resolveUrl } from '../playlist.js';

describe('playlist', () => {
  test('getList returns a defensive copy of the tracks', () => {
    const first = getList();
    const second = getList();

    expect(first).toHaveLength(11);
    expect(second).toHaveLength(11);
    expect(first).not.toBe(second);

    first.pop();
    expect(getList()).toHaveLength(11);
  });

  test('resolveUrl encodes filenames and rejects invalid indexes', () => {
    expect(resolveUrl(0)).toBe('assets/audio/Meditation.mp3');
    expect(resolveUrl(10)).toBe('assets/audio/Epoch%20%E2%88%9E.mp3');
    expect(() => resolveUrl(11)).toThrow(RangeError);
    expect(() => resolveUrl(-1)).toThrow(RangeError);
  });

  test('count reports the playlist size', () => {
    expect(count()).toBe(11);
  });
});
