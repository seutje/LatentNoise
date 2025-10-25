import {
  applyPreset,
  describePreset,
  getDefaultPreset,
  getPreset,
  listPresets,
  resolvePreset,
} from '../presets.js';

describe('presets', () => {
  test('listPresets returns a fresh array with cloned top-level objects', () => {
    const presets = listPresets();
    const second = listPresets();

    expect(Array.isArray(presets)).toBe(true);
    expect(presets).not.toBe(second);

    const first = presets[0];
    const baseline = getDefaultPreset();
    expect(first).not.toBe(baseline);

    first.id = 'mutated-id';
    expect(getDefaultPreset().id).toBe('meditation');
    expect(first.palette).toBe(baseline.palette);
  });

  test('getPreset resolves by index, slug, and default fallback', () => {
    const presetByIndex = getPreset(1);
    expect(presetByIndex).not.toBeNull();

    const presetBySlug = getPreset('Built on the Steppers');
    expect(presetBySlug?.id).toBe('built-on-the-steppers');

    expect(getPreset()).toBe(getDefaultPreset());
    expect(getPreset('missing')).toBeNull();
  });

  test('resolvePreset returns fallback when lookup fails', () => {
    const fallback = { id: 'fallback' };
    expect(resolvePreset('unknown', fallback)).toBe(fallback);
  });

  test('applyPreset scales and clamps grouped parameters', () => {
    const preset = getPreset('meditation');
    const params = {
      sim: {
        spawnRate: 1,
        fieldStrength: 1,
        cohesion: 1,
        repelImpulse: 1,
        vortexAmount: 1,
      },
      render: {
        trailFade: 0.9,
        glow: 0.8,
        sizeJitter: 0.4,
        hueShift: 0,
        sparkleDensity: 0.5,
        zoom: 1,
      },
    };

    const adjusted = applyPreset(preset, params);

    expect(adjusted.sim.spawnRate).toBeCloseTo(0.85, 5);
    expect(adjusted.sim.cohesion).toBeCloseTo(1.15, 5);
    expect(adjusted.render.trailFade).toBeCloseTo(0.98, 5);
    expect(adjusted.render.glow).toBeCloseTo(1, 5);
  });

  test('applyPreset works on flat parameter objects', () => {
    const preset = getPreset('traffic-jam');
    const adjusted = applyPreset(preset, {
      spawnRate: 0.4,
      fieldStrength: 0.5,
      cohesion: 0.6,
      repelImpulse: 0.7,
      vortexAmount: 0.8,
      trailFade: 0.9,
      glow: 0.5,
      sizeJitter: 0.3,
      hueShift: 0,
      sparkleDensity: 0.4,
      zoom: 1,
    });

    expect(adjusted.spawnRate).toBeCloseTo(0.1, 5);
    expect(adjusted.trailFade).toBeCloseTo(0.81, 5);
    expect(adjusted.sparkleDensity).toBeCloseTo(0.42, 5);
  });

  test('describePreset returns presentation metadata', () => {
    const summary = describePreset('Clouds');
    expect(summary).toMatchObject({ id: 'clouds', title: 'Clouds' });
    expect(summary.palette).not.toBe(getPreset('Clouds').palette);
    expect(describePreset('missing')).toBeNull();
  });
});
