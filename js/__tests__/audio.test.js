import { getActivityLevel } from '../audio.js';

describe('audio activity level', () => {
  test('clamps invalid values to zero', () => {
    expect(getActivityLevel()).toBe(0);
    expect(getActivityLevel(-1)).toBe(0);
    expect(getActivityLevel(Number.NaN)).toBe(0);
  });

  test('maps unity RMS to full activity', () => {
    expect(getActivityLevel(1)).toBe(1);
    expect(getActivityLevel(2)).toBe(1);
  });

  test('applies perceptual scaling', () => {
    const rmsMinus25Db = 10 ** (-25 / 20);
    expect(getActivityLevel(rmsMinus25Db)).toBeCloseTo(30 / 55, 5);

    const rmsMinus55Db = 10 ** (-55 / 20);
    expect(getActivityLevel(rmsMinus55Db)).toBe(0);
  });
});
