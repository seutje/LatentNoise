const DEFAULT_BASELINE_PROFILE_HEX = '42';
const MAIN_PROFILE_HEX = '4D';
const HIGH_PROFILE_HEX = '64';

const AVC_LEVEL_CONFIGS = [
  { maxMacroblocks: 1620, levelHex: '1E', profileHex: DEFAULT_BASELINE_PROFILE_HEX },
  { maxMacroblocks: 3600, levelHex: '1F', profileHex: DEFAULT_BASELINE_PROFILE_HEX },
  { maxMacroblocks: 5120, levelHex: '20', profileHex: MAIN_PROFILE_HEX },
  { maxMacroblocks: 8192, levelHex: '28', profileHex: MAIN_PROFILE_HEX },
  { maxMacroblocks: 8704, levelHex: '2A', profileHex: MAIN_PROFILE_HEX },
  { maxMacroblocks: 22080, levelHex: '32', profileHex: HIGH_PROFILE_HEX },
  { maxMacroblocks: 36864, levelHex: '33', profileHex: HIGH_PROFILE_HEX },
  { maxMacroblocks: 139264, levelHex: '34', profileHex: HIGH_PROFILE_HEX },
  { maxMacroblocks: Number.POSITIVE_INFINITY, levelHex: '34', profileHex: HIGH_PROFILE_HEX },
];

function clampDimension(value) {
  if (!Number.isFinite(value) || value <= 0) {
    return 1;
  }
  return Math.floor(value);
}

function calculateMacroblocks(width, height) {
  const safeWidth = clampDimension(width);
  const safeHeight = clampDimension(height);
  const macroblockWidth = Math.ceil(safeWidth / 16);
  const macroblockHeight = Math.ceil(safeHeight / 16);
  return macroblockWidth * macroblockHeight;
}

export function selectCodec(width, height, preferredCodec) {
  if (typeof preferredCodec === 'string' && preferredCodec.trim().length > 0) {
    return preferredCodec;
  }
  const macroblocks = calculateMacroblocks(width, height);
  const match = AVC_LEVEL_CONFIGS.find(({ maxMacroblocks }) => macroblocks <= maxMacroblocks);
  const { levelHex, profileHex } = match || AVC_LEVEL_CONFIGS[AVC_LEVEL_CONFIGS.length - 1];
  return `avc1.${profileHex}00${levelHex}`;
}

export const __test = {
  calculateMacroblocks,
  clampDimension,
  AVC_LEVEL_CONFIGS,
};
