import { projectFeatureValue, formatCorrelation } from '../correlation-math.js';

describe('projectFeatureValue', () => {
  it('projects positive direct features into the expected range', () => {
    expect(projectFeatureValue(1, 'positive', 1)).toBe(1);
    expect(projectFeatureValue(0.5, 'positive', 1)).toBeCloseTo(0, 5);
    expect(projectFeatureValue(0, 'positive', 1)).toBe(-1);
  });

  it('projects positive inverse features by flipping the sign', () => {
    expect(projectFeatureValue(1, 'positive', -1)).toBe(-1);
    expect(projectFeatureValue(0, 'positive', -1)).toBe(1);
  });

  it('projects signed features without remapping the magnitude', () => {
    expect(projectFeatureValue(0.25, 'signed', 1)).toBeCloseTo(0.25, 5);
    expect(projectFeatureValue(-0.75, 'signed', 1)).toBeCloseTo(-0.75, 5);
    expect(projectFeatureValue(0.8, 'signed', -1)).toBeCloseTo(-0.8, 5);
  });
});

describe('formatCorrelation', () => {
  it('returns em dash for non-finite values', () => {
    expect(formatCorrelation(Number.NaN)).toBe('—');
  });

  it('formats mid-range values to three decimals', () => {
    expect(formatCorrelation(0.45678)).toBe('0.457');
  });

  it('retains precision for values near one', () => {
    expect(formatCorrelation(0.99983)).toBe('0.9998');
  });
});
