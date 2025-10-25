import { FRESH_MODEL_ID, FRESH_MODEL_LABEL, isFreshModelId } from '../byom-constants.js';

describe('byom-constants', () => {
  test('constants expose the fresh model metadata', () => {
    expect(FRESH_MODEL_ID).toBe('byom:fresh');
    expect(FRESH_MODEL_LABEL).toBe('Fresh');
  });

  test('isFreshModelId identifies the sentinel id', () => {
    expect(isFreshModelId('byom:fresh')).toBe(true);
    expect(isFreshModelId('other')).toBe(false);
  });
});
