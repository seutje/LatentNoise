import { listPresets } from './presets.js';
import { analyzeFile } from './byom-intake.js';
import { logByomDataset } from './diagnostics.js';
import { FEATURE_LABELS, FEATURE_TYPES } from './audio-features.js';
import { PARAM_NAMES as OUTPUT_LABELS } from './map.js';
import { PRIMARY_WEIGHT, SECONDARY_WEIGHT, formatCorrelation } from './correlation-math.js';
import { FRESH_MODEL_ID, FRESH_MODEL_LABEL } from './byom-constants.js';

const STATUS = Object.freeze({
  IDLE: 'idle',
  PICKING: 'picking',
  ANALYZING: 'analyzing',
  READY: 'ready',
  ERROR: 'error',
});

const STATUS_MESSAGES = {
  [STATUS.IDLE]: () => 'Select a local MP3 file to begin.',
  [STATUS.PICKING]: (ctx) =>
    ctx.fileName
      ? `${ctx.fileName} selected. Choose a baseline preset and model to continue.`
      : 'Choose a baseline preset and model to continue.',
  [STATUS.ANALYZING]: (ctx) =>
    ctx.fileName ? `Analyzing ${ctx.fileName}…` : 'Analyzing audio…',
  [STATUS.READY]: (ctx) => {
    if (ctx.summary) {
      const parts = [
        `${ctx.summary.durationFormatted}`,
        `${ctx.summary.frameCount} frames`,
      ];
      if (ctx.summary.trainFrames !== undefined && ctx.summary.validationFrames !== undefined) {
        parts.push(`train ${ctx.summary.trainFrames} / val ${ctx.summary.validationFrames}`);
      }
      return `Dataset ready — ${parts.join(' · ')}`;
    }
    return ctx.fileName
      ? `${ctx.fileName} is staged. Review settings then start training when ready.`
      : 'Inputs ready. Review settings then start training when ready.';
  },
  [STATUS.ERROR]: (ctx) => ctx.errorMessage ?? 'Analysis failed. Adjust inputs and try again.',
};

const TRAINING_STATUS = Object.freeze({
  IDLE: 'idle',
  PREPARING: 'preparing',
  RUNNING: 'running',
  PAUSED: 'paused',
  CANCELLING: 'cancelling',
  COMPLETED: 'completed',
  CANCELLED: 'cancelled',
  ERROR: 'error',
});

const ACTIVE_TRAINING_STATUSES = new Set([
  TRAINING_STATUS.PREPARING,
  TRAINING_STATUS.RUNNING,
  TRAINING_STATUS.PAUSED,
  TRAINING_STATUS.CANCELLING,
]);

const DEFAULT_HYPERPARAMETERS = Object.freeze({
  epochs: 400,
  learningRate: 0.01,
  batchSize: 1,
  l2: 0,
});

const FOCUSABLE_SELECTORS = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
  'details summary',
];
const FOCUSABLE_QUERY = FOCUSABLE_SELECTORS.join(',');

const ORIENTATION_INFO = Object.freeze({
  direct: { label: 'Direct', orientation: 'direct', inverse: false, sign: 1 },
  inverse: { label: 'Inverse', orientation: 'inverse', inverse: true, sign: -1 },
});

function makeCorrelationId(featureIndex, outputIndex, orientationSign) {
  const sign = orientationSign === -1 ? 'inv' : 'dir';
  return `${featureIndex}:${outputIndex}:${sign}`;
}

function getFeatureTypeByName(name) {
  const type = FEATURE_TYPES[name];
  return type === 'signed' ? 'signed' : 'positive';
}

function resolveOrientation(inverse) {
  return inverse ? ORIENTATION_INFO.inverse : ORIENTATION_INFO.direct;
}

function assignCorrelationWeights() {
  state.correlations.forEach((correlation, index) => {
    correlation.weight = index === 0 ? PRIMARY_WEIGHT : SECONDARY_WEIGHT;
  });
}

function clearCorrelationMetrics() {
  state.correlationResults.clear();
  renderCorrelationList();
}

function updateCorrelationControlsDisabledState() {
  const disabled = !state.support || state.inputsDisabled;
  if (state.elements.addCorrelationButton) {
    state.elements.addCorrelationButton.disabled = disabled;
    state.elements.addCorrelationButton.setAttribute('aria-disabled', disabled ? 'true' : 'false');
  }
  if (!state.elements.correlationList) {
    return;
  }
  const buttons = state.elements.correlationList.querySelectorAll('button[data-action="remove-correlation"]');
  buttons.forEach((button) => {
    if (!(button instanceof HTMLButtonElement)) {
      return;
    }
    button.disabled = disabled;
    button.setAttribute('aria-disabled', disabled ? 'true' : 'false');
  });
}

function renderCorrelationList() {
  const list = state.elements.correlationList;
  const emptyState = state.elements.correlationEmpty;
  if (!list) {
    return;
  }
  list.innerHTML = '';
  if (state.correlations.length === 0) {
    if (emptyState) {
      emptyState.hidden = false;
    }
    updateCorrelationControlsDisabledState();
    return;
  }
  if (emptyState) {
    emptyState.hidden = true;
  }
  state.correlations.forEach((correlation) => {
    const item = document.createElement('li');
    item.className = 'byom-correlation-item';

    const removeButton = document.createElement('button');
    removeButton.type = 'button';
    removeButton.className = 'byom-inline-button';
    removeButton.dataset.action = 'remove-correlation';
    removeButton.dataset.id = correlation.id;
    removeButton.textContent = 'Remove';
    removeButton.disabled = state.inputsDisabled || !state.support;
    removeButton.setAttribute('aria-disabled', removeButton.disabled ? 'true' : 'false');

    const label = document.createElement('span');
    label.className = 'byom-correlation-label';
    const featureStrong = document.createElement('strong');
    featureStrong.textContent = correlation.featureName;
    label.append(featureStrong);
    label.append(document.createTextNode(` → ${correlation.outputName} (${resolveOrientation(correlation.inverse).label})`));

    const result = state.correlationResults.get(correlation.id);
    const value = result && Number.isFinite(result.correlation) ? result.correlation : null;
    const resultSpan = document.createElement('span');
    resultSpan.className = 'byom-correlation-result';
    resultSpan.textContent = value === null ? '—' : formatCorrelation(value);
    if (value !== null) {
      resultSpan.title = `Pearson correlation ${formatCorrelation(value)}`;
    } else {
      resultSpan.removeAttribute('title');
    }

    item.append(removeButton, label, resultSpan);
    list.append(item);
  });
  updateCorrelationControlsDisabledState();
}

function populateCorrelationSelectors() {
  const featureSelect = state.elements.correlationFeatureSelect;
  const outputSelect = state.elements.correlationOutputSelect;
  if (featureSelect) {
    featureSelect.innerHTML = '';
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = 'Select feature';
    placeholder.disabled = true;
    placeholder.selected = true;
    placeholder.hidden = true;
    featureSelect.append(placeholder);
    FEATURE_LABELS.forEach((label, index) => {
      const option = document.createElement('option');
      option.value = String(index);
      option.textContent = label;
      featureSelect.append(option);
    });
  }
  if (outputSelect) {
    outputSelect.innerHTML = '';
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = 'Select output';
    placeholder.disabled = true;
    placeholder.selected = true;
    placeholder.hidden = true;
    outputSelect.append(placeholder);
    OUTPUT_LABELS.forEach((label, index) => {
      const option = document.createElement('option');
      option.value = String(index);
      option.textContent = label;
      outputSelect.append(option);
    });
  }
}

function resetCorrelationDialog() {
  state.elements.correlationForm?.reset();
  if (state.elements.correlationError) {
    state.elements.correlationError.textContent = '';
    state.elements.correlationError.hidden = true;
  }
}

function openCorrelationDialog() {
  if (!state.support) {
    return;
  }
  resetCorrelationDialog();
  const dialog = state.elements.correlationDialog;
  if (!dialog) {
    return;
  }
  try {
    if (typeof dialog.showModal === 'function') {
      dialog.showModal();
    } else {
      dialog.setAttribute('open', 'true');
      dialog.dataset.open = 'true';
    }
  } catch (error) {
    console.warn('[byom] Failed to open correlation dialog', error);
  }
  state.elements.correlationFeatureSelect?.focus();
}

function closeCorrelationDialog() {
  const dialog = state.elements.correlationDialog;
  if (!dialog) {
    return;
  }
  if (typeof dialog.close === 'function') {
    dialog.close();
  } else {
    dialog.removeAttribute('open');
    dialog.dataset.open = 'false';
  }
  resetCorrelationDialog();
}

function showCorrelationError(message) {
  if (!state.elements.correlationError) {
    return;
  }
  state.elements.correlationError.hidden = false;
  state.elements.correlationError.textContent = message;
}

function createCorrelationDefinition({ featureIndex, outputIndex, inverse }) {
  const featureIdx = Number(featureIndex);
  const outputIdx = Number(outputIndex);
  if (!Number.isInteger(featureIdx) || featureIdx < 0 || featureIdx >= FEATURE_LABELS.length) {
    return null;
  }
  if (!Number.isInteger(outputIdx) || outputIdx < 0 || outputIdx >= OUTPUT_LABELS.length) {
    return null;
  }
  const featureName = FEATURE_LABELS[featureIdx] ?? `feature-${featureIdx}`;
  const outputName = OUTPUT_LABELS[outputIdx] ?? `output-${outputIdx}`;
  const orientation = resolveOrientation(Boolean(inverse));
  return {
    id: makeCorrelationId(featureIdx, outputIdx, orientation.sign),
    featureIndex: featureIdx,
    featureName,
    featureType: getFeatureTypeByName(featureName),
    outputIndex: outputIdx,
    outputName,
    inverse: orientation.inverse,
    orientation: orientation.orientation,
    orientationSign: orientation.sign,
    weight: 0,
  };
}

function removeCorrelation(id) {
  const index = state.correlations.findIndex((correlation) => correlation.id === id);
  if (index === -1) {
    return;
  }
  state.correlations.splice(index, 1);
  state.correlationResults.delete(id);
  assignCorrelationWeights();
  renderCorrelationList();
  updateTrainAvailability();
  updateStatusMessage();
}

function handleCorrelationListClick(event) {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  if (target.dataset.action === 'remove-correlation') {
    const id = target.dataset.id;
    if (typeof id === 'string' && id.length > 0) {
      removeCorrelation(id);
    }
  }
}

function addCorrelation(definition) {
  if (!definition) {
    return false;
  }
  const exists = state.correlations.some((correlation) => correlation.id === definition.id);
  if (exists) {
    showCorrelationError('That correlation already exists. Choose another combination.');
    return false;
  }
  state.correlations.push(definition);
  state.correlationResults.delete(definition.id);
  assignCorrelationWeights();
  renderCorrelationList();
  updateTrainAvailability();
  updateStatusMessage();
  return true;
}

function handleCorrelationFormSubmit(event) {
  event.preventDefault();
  if (state.inputsDisabled || !state.support) {
    return;
  }
  const featureValue = state.elements.correlationFeatureSelect?.value ?? '';
  const outputValue = state.elements.correlationOutputSelect?.value ?? '';
  const inverse = Boolean(state.elements.correlationInverse?.checked);
  const definition = createCorrelationDefinition({ featureIndex: featureValue, outputIndex: outputValue, inverse });
  if (!definition) {
    showCorrelationError('Select a valid feature and output to continue.');
    return;
  }
  const added = addCorrelation(definition);
  if (added) {
    closeCorrelationDialog();
    state.elements.addCorrelationButton?.focus();
  }
}

function handleAddCorrelationClick(event) {
  event.preventDefault();
  if (state.inputsDisabled || !state.support) {
    return;
  }
  openCorrelationDialog();
}

function handleCorrelationDialogCancel(event) {
  event.preventDefault();
  closeCorrelationDialog();
}

function getCorrelationDefinitions() {
  assignCorrelationWeights();
  return state.correlations.map((correlation) => ({
    featureIndex: correlation.featureIndex,
    featureName: correlation.featureName,
    featureType: correlation.featureType,
    outputIndex: correlation.outputIndex,
    outputName: correlation.outputName,
    inverse: correlation.inverse,
    orientation: correlation.orientation,
    orientationSign: correlation.orientationSign,
    weight: correlation.weight,
  }));
}

function applyCorrelationMetrics(metrics) {
  if (!metrics) {
    return;
  }
  const entries = Array.isArray(metrics.perCorrelation)
    ? metrics.perCorrelation
    : Array.isArray(metrics.correlations)
      ? metrics.correlations
      : [];
  if (entries.length === 0) {
    return;
  }
  state.correlationResults.clear();
  entries.forEach((entry) => {
    const featureIndex = Number(entry.featureIndex);
    const outputIndex = Number(entry.outputIndex);
    const orientationSign = Number(entry.orientationSign) === -1 ? -1 : 1;
    const id = makeCorrelationId(featureIndex, outputIndex, orientationSign);
    state.correlationResults.set(id, {
      correlation: Number(entry.correlation),
      fitness: Number(entry.fitness),
      orientationSign,
    });
  });
  renderCorrelationList();
}

function detectSupport() {
  if (typeof window === 'undefined') {
    return false;
  }
  const hasFileApi = 'FileReader' in window && 'File' in window && 'Blob' in window;
  const hasWorkers = 'Worker' in window;
  const hasIndexedDb = 'indexedDB' in window;
  return hasFileApi && hasWorkers && hasIndexedDb;
}

const state = {
  mounted: false,
  open: false,
  status: STATUS.IDLE,
  support: detectSupport(),
  inputsDisabled: false,
  file: null,
  fileName: '',
  objectUrl: '',
  dataset: null,
  datasetSummary: null,
  datasetContext: null,
  analysisController: null,
  analysisToken: 0,
  analysisActive: false,
  lastError: null,
  correlations: [],
  correlationResults: new Map(),
  training: {
    status: TRAINING_STATUS.IDLE,
    active: false,
    progress: 0,
    epoch: 0,
    epochs: 0,
    etaMs: 0,
    trainLoss: null,
    valLoss: null,
    learningRate: null,
    message: '',
    error: null,
  },
  elements: {
    drawer: null,
    toggle: null,
    closeButton: null,
    cancelButton: null,
    trainButton: null,
    fileInput: null,
    presetSelect: null,
    modelSelect: null,
    statusText: null,
    progress: null,
    form: null,
    backdrop: null,
    uploadSection: null,
    summary: null,
    correlationSection: null,
    correlationList: null,
    correlationEmpty: null,
    addCorrelationButton: null,
    correlationDialog: null,
    correlationForm: null,
    correlationFeatureSelect: null,
    correlationOutputSelect: null,
    correlationInverse: null,
    correlationError: null,
  },
  options: {
    modelOptions: [],
  },
  handlers: {
    onTrain: null,
    onCancel: null,
    onPause: null,
    onResume: null,
  },
  lastFocusedElement: null,
};

function getFocusableElements() {
  if (!state.open || !state.elements.drawer) {
    return [];
  }
  const candidates = state.elements.drawer.querySelectorAll(FOCUSABLE_QUERY);
  return Array.from(candidates).filter((el) => {
    if (!(el instanceof HTMLElement)) {
      return false;
    }
    if (el.hasAttribute('disabled') || el.getAttribute('aria-hidden') === 'true') {
      return false;
    }
    if (el.tabIndex < 0) {
      return false;
    }
    return true;
  });
}

function updateStatusMessage() {
  let message = '';
  if (state.training.status !== TRAINING_STATUS.IDLE) {
    message = formatTrainingStatusMessage();
  } else {
    const messageFactory = STATUS_MESSAGES[state.status] ?? STATUS_MESSAGES[STATUS.IDLE];
    message = messageFactory({
      fileName: state.fileName,
      summary: state.datasetSummary,
      errorMessage: state.lastError && typeof state.lastError.message === 'string'
        ? state.lastError.message
        : typeof state.lastError === 'string'
          ? state.lastError
          : undefined,
    });
    if (state.status === STATUS.READY && state.correlations.length === 0) {
      message = 'Dataset ready — add at least one correlation to enable training.';
    }
  }
  if (state.elements.statusText) {
    state.elements.statusText.textContent = message;
  }
  updateSummaryDisplay();
}

function updateTrainAvailability() {
  if (!state.elements.trainButton) {
    return;
  }
  if (state.training.status !== TRAINING_STATUS.IDLE &&
    state.training.status !== TRAINING_STATUS.COMPLETED &&
    state.training.status !== TRAINING_STATUS.CANCELLED &&
    state.training.status !== TRAINING_STATUS.ERROR) {
    renderTrainingControls();
    return;
  }
  const ready = state.status === STATUS.READY && state.correlations.length > 0;
  state.elements.trainButton.disabled = !ready;
  state.elements.trainButton.setAttribute('aria-disabled', ready ? 'false' : 'true');
  renderTrainingControls();
}

function updateSummaryDisplay() {
  const summaryEl = state.elements.summary;
  if (!summaryEl) {
    return;
  }
  const summary = state.datasetSummary;
  if (!summary) {
    summaryEl.textContent = '';
    summaryEl.hidden = true;
    return;
  }
  const warnings = Array.isArray(summary.warnings) ? summary.warnings : [];
  if (warnings.length === 0) {
    summaryEl.textContent = '';
    summaryEl.hidden = true;
    return;
  }
  summaryEl.hidden = false;
  summaryEl.textContent = warnings.map((warning) => `⚠︎ ${warning}`).join(' ');
}

function formatLoss(value) {
  if (!Number.isFinite(value)) {
    return '—';
  }
  if (Math.abs(value) >= 1) {
    return value.toFixed(3);
  }
  return value.toFixed(4);
}

function formatEta(ms) {
  if (!Number.isFinite(ms) || ms <= 0) {
    return '';
  }
  const totalSeconds = Math.round(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
}

function isTrainingBusy() {
  return ACTIVE_TRAINING_STATUSES.has(state.training.status);
}

function applyTrainingDetail(detail = {}) {
  if (detail.message !== undefined) {
    state.training.message = detail.message;
  }
  if (Number.isFinite(detail.progress)) {
    state.training.progress = clampProgress(detail.progress);
  }
  if (Number.isFinite(detail.epoch)) {
    state.training.epoch = detail.epoch;
  }
  if (Number.isFinite(detail.epochs)) {
    state.training.epochs = detail.epochs;
  }
  if (Number.isFinite(detail.etaMs)) {
    state.training.etaMs = Math.max(detail.etaMs, 0);
  }
  if (Number.isFinite(detail.trainLoss)) {
    state.training.trainLoss = detail.trainLoss;
  }
  if (Number.isFinite(detail.valLoss)) {
    state.training.valLoss = detail.valLoss;
  }
  if (Number.isFinite(detail.learningRate)) {
    state.training.learningRate = detail.learningRate;
  }
  if (detail.error !== undefined) {
    state.training.error = detail.error;
  }
}

function setTrainingStatusInternal(status, detail = {}) {
  if (!Object.values(TRAINING_STATUS).includes(status)) {
    return;
  }
  state.training.status = status;
  state.training.active = ACTIVE_TRAINING_STATUSES.has(status);
  applyTrainingDetail(detail);
  if (detail?.correlationMetrics) {
    applyCorrelationMetrics(detail.correlationMetrics);
  } else if (Array.isArray(detail?.correlations)) {
    applyCorrelationMetrics({ perCorrelation: detail.correlations });
  }
  if (state.training.active) {
    setInputsDisabled(true);
  } else if (!state.analysisActive) {
    setInputsDisabled(false);
  }
  if (Number.isFinite(state.training.progress)) {
    updateProgress(state.training.progress);
  }
  updateStatusMessage();
  renderTrainingControls();
}

function updateTrainingProgressState(progress = {}) {
  applyTrainingDetail(progress);
  if (Number.isFinite(state.training.progress)) {
    updateProgress(state.training.progress);
  }
  updateStatusMessage();
  renderTrainingControls();
}

function resetTrainingState() {
  state.training.status = TRAINING_STATUS.IDLE;
  state.training.active = false;
  state.training.progress = 0;
  state.training.epoch = 0;
  state.training.epochs = 0;
  state.training.etaMs = 0;
  state.training.trainLoss = null;
  state.training.valLoss = null;
  state.training.learningRate = null;
  state.training.message = '';
  state.training.error = null;
  if (!state.analysisActive) {
    setInputsDisabled(false);
  }
  updateProgress(state.dataset ? 1 : 0);
  updateStatusMessage();
  renderTrainingControls();
}

function renderTrainingControls() {
  const trainButton = state.elements.trainButton;
  const cancelButton = state.elements.cancelButton;
  if (!trainButton || !cancelButton) {
    return;
  }

  const { status } = state.training;
  const readyForNewRun =
    status === TRAINING_STATUS.IDLE ||
    status === TRAINING_STATUS.COMPLETED ||
    status === TRAINING_STATUS.CANCELLED ||
    status === TRAINING_STATUS.ERROR;
  const datasetReady = state.status === STATUS.READY && state.correlations.length > 0;

  let trainLabel = 'Train Model';
  let trainDisabled = !datasetReady;
  let cancelLabel = 'Cancel';
  let cancelDisabled = false;

  if (status === TRAINING_STATUS.PREPARING) {
    trainLabel = 'Initializing…';
    trainDisabled = true;
    cancelLabel = 'Stop';
    cancelDisabled = false;
  } else if (status === TRAINING_STATUS.RUNNING) {
    trainLabel = 'Pause';
    trainDisabled = false;
    cancelLabel = 'Stop';
    cancelDisabled = false;
  } else if (status === TRAINING_STATUS.PAUSED) {
    trainLabel = 'Resume';
    trainDisabled = false;
    cancelLabel = 'Stop';
    cancelDisabled = false;
  } else if (status === TRAINING_STATUS.CANCELLING) {
    trainLabel = 'Pause';
    trainDisabled = true;
    cancelLabel = 'Stopping…';
    cancelDisabled = true;
  } else if (status === TRAINING_STATUS.COMPLETED) {
    trainLabel = 'Train Again';
    trainDisabled = !datasetReady;
    cancelLabel = 'Close';
    cancelDisabled = false;
  } else if (!readyForNewRun) {
    trainDisabled = true;
  }

  trainButton.textContent = trainLabel;
  trainButton.disabled = trainDisabled;
  trainButton.setAttribute('aria-disabled', trainDisabled ? 'true' : 'false');

  cancelButton.textContent = cancelLabel;
  cancelButton.disabled = cancelDisabled;
  cancelButton.setAttribute('aria-disabled', cancelDisabled ? 'true' : 'false');
}

function formatTrainingStatusMessage() {
  const training = state.training;
  if (typeof training.message === 'string' && training.message.length > 0) {
    return training.message;
  }
  switch (training.status) {
    case TRAINING_STATUS.PREPARING:
      return 'Preparing training…';
    case TRAINING_STATUS.RUNNING: {
      const segments = [];
      if (training.epochs > 0) {
        const epoch = Math.max(1, Math.min(training.epoch, training.epochs));
        segments.push(`Epoch ${epoch}/${training.epochs}`);
      } else if (training.epoch > 0) {
        segments.push(`Epoch ${training.epoch}`);
      }
      const metrics = [];
      if (Number.isFinite(training.trainLoss)) {
        metrics.push(`loss ${formatLoss(training.trainLoss)}`);
      }
      if (Number.isFinite(training.valLoss)) {
        metrics.push(`val ${formatLoss(training.valLoss)}`);
      }
      if (Number.isFinite(training.learningRate)) {
        metrics.push(`lr ${formatLoss(training.learningRate)}`);
      }
      if (metrics.length > 0) {
        segments.push(metrics.join(' · '));
      }
      const etaText = formatEta(training.etaMs);
      if (etaText) {
        segments.push(`ETA ${etaText}`);
      }
      if (segments.length === 0) {
        return 'Training in progress…';
      }
      return `Training ${segments.join(' — ')}`;
    }
    case TRAINING_STATUS.PAUSED: {
      const epoch = training.epochs > 0 ? `${Math.max(1, Math.min(training.epoch, training.epochs))}/${training.epochs}` : `${training.epoch}`;
      return `Training paused at epoch ${epoch}.`;
    }
    case TRAINING_STATUS.CANCELLING:
      return 'Stopping training…';
    case TRAINING_STATUS.COMPLETED: {
      const metrics = [];
      if (Number.isFinite(training.trainLoss)) {
        metrics.push(`loss ${formatLoss(training.trainLoss)}`);
      }
      if (Number.isFinite(training.valLoss)) {
        metrics.push(`val ${formatLoss(training.valLoss)}`);
      }
      return metrics.length > 0
        ? `Training complete — ${metrics.join(' · ')}`
        : 'Training complete.';
    }
    case TRAINING_STATUS.CANCELLED:
      return 'Training cancelled.';
    case TRAINING_STATUS.ERROR: {
      if (training.error && typeof training.error.message === 'string') {
        return `Training failed — ${training.error.message}`;
      }
      if (typeof training.error === 'string' && training.error.length > 0) {
        return `Training failed — ${training.error}`;
      }
      return 'Training failed.';
    }
    default:
      return STATUS_MESSAGES[state.status]?.({
        fileName: state.fileName,
        summary: state.datasetSummary,
      }) ?? STATUS_MESSAGES[STATUS.IDLE]({ fileName: state.fileName, summary: state.datasetSummary });
  }
}

function collectHyperparameters() {
  const form = state.elements.form;
  if (!(form instanceof HTMLFormElement)) {
    return { ...DEFAULT_HYPERPARAMETERS };
  }
  const getNumber = (selector, fallback) => {
    const el = form.querySelector(selector);
    if (!(el instanceof HTMLInputElement)) {
      return fallback;
    }
    const value = Number(el.value);
    return Number.isFinite(value) ? value : fallback;
  };

  return {
    epochs: getNumber('#byom-epochs', DEFAULT_HYPERPARAMETERS.epochs),
    learningRate: getNumber('#byom-lr', DEFAULT_HYPERPARAMETERS.learningRate),
    batchSize: getNumber('#byom-batch', DEFAULT_HYPERPARAMETERS.batchSize),
    l2: getNumber('#byom-l2', DEFAULT_HYPERPARAMETERS.l2),
  };
}

function releaseObjectUrl() {
  if (state.objectUrl) {
    URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = '';
  }
}

function clearDataset() {
  state.dataset = null;
  state.datasetSummary = null;
  state.datasetContext = null;
  resetTrainingState();
  updateProgress(0);
  updateSummaryDisplay();
  logByomDataset(null);
}

function abortAnalysis() {
  if (state.analysisController) {
    state.analysisController.abort();
    state.analysisController = null;
  }
  state.analysisActive = false;
  setInputsDisabled(false);
}

function selectFile(file) {
  abortAnalysis();
  clearDataset();
  releaseObjectUrl();
  state.file = file ?? null;
  state.fileName = file ? file.name : '';
  state.lastError = null;
  clearCorrelationMetrics();
  if (file) {
    try {
      state.objectUrl = URL.createObjectURL(file);
    } catch {
      state.objectUrl = '';
    }
  }
}

function getSelectedPresetId() {
  return state.elements.presetSelect?.value ?? '';
}

function getSelectedModelId() {
  return state.elements.modelSelect?.value ?? '';
}

function getFileSignature(file) {
  if (!file) {
    return '';
  }
  const modified = Number.isFinite(file.lastModified) ? file.lastModified : 0;
  return `${file.name}:${file.size}:${modified}`;
}

function setInputsDisabled(disabled) {
  const targets = [state.elements.fileInput, state.elements.presetSelect, state.elements.modelSelect];
  const lock = disabled || state.training.active;
  state.inputsDisabled = lock;
  targets.forEach((el) => {
    if (!el) {
      return;
    }
    if (lock) {
      el.setAttribute('aria-disabled', 'true');
      el.disabled = true;
    } else {
      el.removeAttribute('aria-disabled');
      el.disabled = false;
    }
  });
  updateCorrelationControlsDisabledState();
}

function ensureDataset() {
  if (!state.support) {
    return;
  }
  if (!state.file) {
    abortAnalysis();
    clearDataset();
    return;
  }
  const presetId = getSelectedPresetId();
  const modelId = getSelectedModelId();
  if (!presetId || !modelId) {
    abortAnalysis();
    clearDataset();
    return;
  }
  const signature = getFileSignature(state.file);
  if (
    state.dataset &&
    state.datasetContext &&
    state.datasetContext.fileSignature === signature &&
    state.datasetContext.presetId === presetId &&
    state.datasetContext.modelId === modelId
  ) {
    return;
  }
  startAnalysis({ file: state.file, presetId, modelId, fileSignature: signature });
}

function startAnalysis({ file, presetId, modelId, fileSignature }) {
  abortAnalysis();
  clearDataset();
  state.analysisToken += 1;
  const token = state.analysisToken;
  const controller = new AbortController();
  state.analysisController = controller;
  state.analysisActive = true;
  state.lastError = null;
  setInputsDisabled(true);
  setStatus(STATUS.ANALYZING);

  analyzeFile({
    file,
    presetId,
    modelUrl: modelId,
    signal: controller.signal,
    onProgress: (info) => {
      if (token !== state.analysisToken) {
        return;
      }
      if (info && typeof info.value === 'number' && Number.isFinite(info.value)) {
        updateProgress(info.value);
      }
    },
  })
    .then(({ dataset, summary }) => {
      if (token !== state.analysisToken) {
        return;
      }
      state.dataset = dataset;
      state.datasetSummary = {
        ...summary,
        warnings: summary?.warnings ?? [],
        presetId,
        modelId,
      };
      state.datasetContext = {
        fileSignature,
        presetId,
        modelId,
      };
      state.analysisActive = false;
      state.analysisController = null;
      updateProgress(1);
      logByomDataset(state.datasetSummary);
      setInputsDisabled(false);
      setStatus(evaluateStatus());
    })
    .catch((error) => {
      if (token !== state.analysisToken) {
        return;
      }
      state.analysisActive = false;
      state.analysisController = null;
      setInputsDisabled(false);
      if (error && error.name === 'AbortError') {
        updateProgress(0);
        return;
      }
      console.error('[byom] dataset analysis failed', error);
      state.lastError = error instanceof Error ? error : new Error(String(error));
      clearDataset();
      logByomDataset(null);
      setStatus(STATUS.ERROR);
    });
}

function setStatus(nextStatus) {
  if (!Object.values(STATUS).includes(nextStatus)) {
    return;
  }
  state.status = nextStatus;
  if (state.elements.drawer) {
    state.elements.drawer.setAttribute('data-status', nextStatus);
  }
  if (state.elements.toggle) {
    state.elements.toggle.setAttribute('data-status', nextStatus);
  }
  updateStatusMessage();
  updateTrainAvailability();
}

function evaluateStatus() {
  if (state.analysisActive) {
    return STATUS.ANALYZING;
  }
  if (!state.file) {
    return STATUS.IDLE;
  }
  const presetSelected = Boolean(state.elements.presetSelect && state.elements.presetSelect.value);
  const modelSelected = Boolean(state.elements.modelSelect && state.elements.modelSelect.value);
  if (!presetSelected || !modelSelected) {
    return STATUS.PICKING;
  }
  if (state.dataset) {
    return STATUS.READY;
  }
  if (state.lastError) {
    return STATUS.ERROR;
  }
  if (state.file) {
    return STATUS.ANALYZING;
  }
  return STATUS.READY;
}

function clampProgress(value) {
  if (!Number.isFinite(value) || value <= 0) {
    return 0;
  }
  if (value >= 1) {
    return 1;
  }
  return value;
}

function updateProgress(value) {
  if (!state.elements.progress) {
    return;
  }
  const clamped = clampProgress(value);
  state.elements.progress.value = clamped;
  state.elements.progress.textContent = `${Math.round(clamped * 100)}%`;
}

function focusFirstElement() {
  const focusables = getFocusableElements();
  if (focusables.length > 0) {
    focusables[0].focus();
  } else if (state.elements.closeButton) {
    state.elements.closeButton.focus();
  }
}

function openDrawer() {
  if (!state.support || state.open) {
    return;
  }
  if (!state.elements.drawer || !state.elements.toggle) {
    return;
  }
  state.open = true;
  state.lastFocusedElement = /** @type {HTMLElement|null} */ (document.activeElement);
  state.elements.drawer.dataset.state = 'open';
  state.elements.drawer.setAttribute('aria-hidden', 'false');
  state.elements.toggle.setAttribute('aria-expanded', 'true');
  if (document.body) {
    document.body.classList.add('byom-active');
  }
  focusFirstElement();
}

function closeDrawer({ restoreFocus = true } = {}) {
  if (!state.open || !state.elements.drawer || !state.elements.toggle) {
    return;
  }
  state.open = false;
  state.elements.drawer.dataset.state = 'closed';
  state.elements.drawer.setAttribute('aria-hidden', 'true');
  state.elements.toggle.setAttribute('aria-expanded', 'false');
  if (document.body) {
    document.body.classList.remove('byom-active');
  }
  if (
    restoreFocus &&
    state.lastFocusedElement &&
    typeof document.contains === 'function' &&
    document.contains(state.lastFocusedElement)
  ) {
    state.lastFocusedElement.focus();
  }
  state.lastFocusedElement = null;
}

function toggleDrawer() {
  if (!state.support) {
    return;
  }
  if (state.open) {
    closeDrawer();
  } else {
    openDrawer();
  }
}

function clearFileSelection() {
  abortAnalysis();
  releaseObjectUrl();
  state.file = null;
  state.fileName = '';
  clearDataset();
  if (state.elements.fileInput) {
    state.elements.fileInput.value = '';
  }
}

function resetForm() {
  clearFileSelection();
  if (state.elements.form) {
    state.elements.form.reset();
  }
  if (state.elements.presetSelect) {
    state.elements.presetSelect.selectedIndex = 0;
  }
  if (state.elements.modelSelect) {
    const select = state.elements.modelSelect;
    select.value = FRESH_MODEL_ID;
    if (select.value !== FRESH_MODEL_ID) {
      const options = Array.from(select.options);
      const freshOption = options.find((option) => option.value === FRESH_MODEL_ID);
      if (freshOption) {
        freshOption.selected = true;
      } else if (options.length > 0) {
        select.selectedIndex = 1 < options.length ? 1 : 0;
      }
    }
  }
  state.lastError = null;
  state.analysisToken += 1;
  setInputsDisabled(false);
  setStatus(STATUS.IDLE);
}

function handleFileChange(event) {
  const input = event.target;
  if (!(input instanceof HTMLInputElement)) {
    return;
  }
  if (!input.files || input.files.length === 0) {
    clearFileSelection();
    setStatus(STATUS.IDLE);
    return;
  }
  const file = input.files[0];
  selectFile(file);
  updateStatusFromInputs();
}

function updateStatusFromInputs() {
  ensureDataset();
  const next = evaluateStatus();
  setStatus(next);
}

function handleFormInput(event) {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  if (target === state.elements.presetSelect || target === state.elements.modelSelect) {
    updateStatusFromInputs();
  }
}

function handleToggleClick() {
  toggleDrawer();
}

function handleDrawerClick(event) {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  if (target.dataset.action === 'close' || target === state.elements.backdrop) {
    closeDrawer();
  }
}

function handleDragEnter(event) {
  if (!state.support) {
    return;
  }
  event.preventDefault();
  state.elements.uploadSection?.classList.add('is-drop-active');
}

function handleDragOver(event) {
  if (!state.support) {
    return;
  }
  event.preventDefault();
  if (event.dataTransfer) {
    event.dataTransfer.dropEffect = 'copy';
  }
  state.elements.uploadSection?.classList.add('is-drop-active');
}

function handleDragLeave(event) {
  if (!state.support) {
    return;
  }
  if (!state.elements.uploadSection) {
    return;
  }
  if (!event.relatedTarget || !state.elements.uploadSection.contains(event.relatedTarget)) {
    state.elements.uploadSection.classList.remove('is-drop-active');
  }
}

function handleDrop(event) {
  if (!state.support) {
    return;
  }
  event.preventDefault();
  state.elements.uploadSection?.classList.remove('is-drop-active');
  const files = event.dataTransfer?.files;
  if (!files || files.length === 0) {
    return;
  }
  const file = files[0];
  selectFile(file);
  if (state.elements.fileInput) {
    try {
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      state.elements.fileInput.files = dataTransfer.files;
    } catch {
      // Ignore inability to sync the input field (older browsers).
    }
  }
  updateStatusFromInputs();
}

function handleCancel(event) {
  event.preventDefault();
  const wasTraining = isTrainingBusy();
  if (wasTraining) {
    const handled = typeof state.handlers.onCancel === 'function'
      ? state.handlers.onCancel({ training: true, status: state.training.status })
      : undefined;
    if (handled !== false) {
      return;
    }
  }
  const previousStatus = state.training.status;
  resetForm();
  if (typeof state.handlers.onCancel === 'function') {
    state.handlers.onCancel({ training: false, status: previousStatus });
  }
  closeDrawer();
}

function handleTrain(event) {
  event.preventDefault();
  const trainingStatus = state.training.status;
  if (trainingStatus === TRAINING_STATUS.RUNNING) {
    if (typeof state.handlers.onPause === 'function') {
      state.handlers.onPause();
    }
    return;
  }
  if (trainingStatus === TRAINING_STATUS.PAUSED) {
    if (typeof state.handlers.onResume === 'function') {
      state.handlers.onResume();
    }
    return;
  }
  if (trainingStatus === TRAINING_STATUS.PREPARING || trainingStatus === TRAINING_STATUS.CANCELLING) {
    return;
  }
  if (state.status !== STATUS.READY || !state.dataset) {
    return;
  }
  if (state.correlations.length === 0) {
    showCorrelationError('Add at least one correlation before training.');
    return;
  }
  const correlations = getCorrelationDefinitions();
  if (correlations.length === 0) {
    showCorrelationError('Add at least one correlation before training.');
    return;
  }
  clearCorrelationMetrics();
  if (typeof state.handlers.onTrain === 'function') {
    state.handlers.onTrain({
      file: state.file,
      objectUrl: state.objectUrl,
      preset: state.elements.presetSelect?.value ?? '',
      model: state.elements.modelSelect?.value ?? '',
      dataset: state.dataset,
      summary: state.datasetSummary,
      hyperparameters: collectHyperparameters(),
      correlations,
    });
  } else {
    console.info(
      '[byom] Train requested (stub)',
      state.fileName,
      state.elements.presetSelect?.value,
      state.elements.modelSelect?.value,
      state.datasetSummary,
      correlations,
    );
  }
}

function handleFocusTrap(event) {
  if (!state.open || event.key !== 'Tab') {
    return;
  }
  const focusables = getFocusableElements();
  if (focusables.length === 0) {
    event.preventDefault();
    return;
  }
  const first = focusables[0];
  const last = focusables[focusables.length - 1];
  const active = document.activeElement;
  if (event.shiftKey) {
    if (active === first || !focusables.includes(active)) {
      event.preventDefault();
      last.focus();
    }
  } else if (active === last) {
    event.preventDefault();
    first.focus();
  }
}

function isTypingTarget(target) {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const tag = target.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || target.isContentEditable) {
    return true;
  }
  return false;
}

function handleGlobalKeydown(event) {
  if (!state.support) {
    return;
  }
  if (event.defaultPrevented) {
    return;
  }
  if (event.key === 'Escape' && state.open) {
    event.preventDefault();
    closeDrawer();
    return;
  }
  if (event.code === 'KeyY' && !event.altKey && !event.metaKey && !event.ctrlKey) {
    if (isTypingTarget(event.target) && !state.open) {
      return;
    }
    event.preventDefault();
    toggleDrawer();
  }
}

function populatePresetOptions() {
  if (!state.elements.presetSelect) {
    return;
  }
  const presets = listPresets();
  state.elements.presetSelect.innerHTML = '';
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = 'Select preset';
  placeholder.disabled = true;
  placeholder.selected = true;
  placeholder.hidden = true;
  state.elements.presetSelect.append(placeholder);

  presets.forEach((preset) => {
    const option = document.createElement('option');
    option.value = preset.id;
    option.textContent = preset.title ?? preset.id;
    state.elements.presetSelect.append(option);
  });
  state.elements.presetSelect.selectedIndex = 0;
}

function sanitizeModelOption(entry) {
  if (!entry || typeof entry.id !== 'string') {
    return null;
  }
  const id = entry.id;
  const label = typeof entry.label === 'string' && entry.label.trim().length > 0 ? entry.label.trim() : id;
  return { id, label };
}

function buildModelOptionEntries(modelOptions) {
  const source = Array.isArray(modelOptions) ? modelOptions : [];
  const entries = [];
  const seen = new Set();
  const providedFresh = source.find((entry) => entry && entry.id === FRESH_MODEL_ID);
  const freshOption = sanitizeModelOption(providedFresh) ?? { id: FRESH_MODEL_ID, label: FRESH_MODEL_LABEL };
  entries.push(freshOption);
  seen.add(FRESH_MODEL_ID);
  source.forEach((entry) => {
    const sanitized = sanitizeModelOption(entry);
    if (!sanitized || seen.has(sanitized.id)) {
      return;
    }
    entries.push(sanitized);
    seen.add(sanitized.id);
  });
  return entries;
}

function populateModelOptions(modelOptions) {
  if (!state.elements.modelSelect) {
    return;
  }
  const previousValue = state.elements.modelSelect.value;
  const options = buildModelOptionEntries(Array.isArray(modelOptions) ? modelOptions : state.options.modelOptions);
  state.options.modelOptions = options;
  state.elements.modelSelect.innerHTML = '';
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = 'Select model';
  placeholder.disabled = true;
  placeholder.hidden = true;
  state.elements.modelSelect.append(placeholder);

  options.forEach((entry) => {
    if (!entry) {
      return;
    }
    const option = document.createElement('option');
    option.value = String(entry.id ?? '');
    option.textContent = entry.label ?? option.value;
    state.elements.modelSelect.append(option);
  });

  const defaultValue = options.some((entry) => entry.id === previousValue && entry.id) ? previousValue : FRESH_MODEL_ID;
  state.elements.modelSelect.value = defaultValue;
  if (state.elements.modelSelect.value !== defaultValue) {
    state.elements.modelSelect.value = FRESH_MODEL_ID;
  }
}

function updateSupportVisibility() {
  if (!state.elements.toggle) {
    return;
  }
  if (state.support) {
    state.elements.toggle.removeAttribute('disabled');
    state.elements.toggle.removeAttribute('aria-disabled');
    state.elements.toggle.title = '';
    state.elements.correlationSection?.removeAttribute('aria-hidden');
    updateCorrelationControlsDisabledState();
    return;
  }
  state.elements.toggle.disabled = true;
  state.elements.toggle.setAttribute('aria-disabled', 'true');
  state.elements.toggle.title = 'BYOM requires File API, Web Workers, and IndexedDB support.';
  if (state.elements.drawer) {
    state.elements.drawer.setAttribute('aria-hidden', 'true');
    state.elements.drawer.dataset.state = 'unsupported';
  }
  if (state.elements.statusText) {
    state.elements.statusText.textContent = 'BYOM mode is unavailable in this environment.';
  }
  if (state.elements.correlationSection) {
    state.elements.correlationSection.setAttribute('aria-hidden', 'true');
  }
  updateCorrelationControlsDisabledState();
}

export function mount({ drawer, toggle, modelOptions = [], onTrain, onCancel } = {}) {
  if (state.mounted) {
    return;
  }
  if (!(drawer instanceof HTMLElement) || !(toggle instanceof HTMLElement)) {
    console.warn('[byom] Drawer or toggle element missing; BYOM mode will remain inactive.');
    return;
  }
  state.elements.drawer = drawer;
  state.elements.toggle = toggle;
  state.elements.closeButton = drawer.querySelector('#byom-close');
  state.elements.cancelButton = drawer.querySelector('#byom-cancel');
  state.elements.trainButton = drawer.querySelector('#byom-train');
  state.elements.fileInput = drawer.querySelector('#byom-file');
  state.elements.presetSelect = drawer.querySelector('#byom-preset');
  state.elements.modelSelect = drawer.querySelector('#byom-model');
  state.elements.statusText = drawer.querySelector('#byom-status');
  state.elements.progress = drawer.querySelector('#byom-progress');
  state.elements.form = drawer.querySelector('#byom-form');
  state.elements.backdrop = drawer.querySelector('.byom-backdrop');
  state.elements.uploadSection = drawer.querySelector('.byom-upload');
  state.elements.correlationSection = drawer.querySelector('.byom-correlations');
  state.elements.correlationList = drawer.querySelector('#byom-correlation-list');
  state.elements.correlationEmpty = drawer.querySelector('#byom-correlation-empty');
  state.elements.addCorrelationButton = drawer.querySelector('#byom-add-correlation');
  state.elements.correlationDialog = drawer.querySelector('#byom-correlation-dialog');
  state.elements.correlationForm = drawer.querySelector('#byom-correlation-form');
  state.elements.correlationFeatureSelect = drawer.querySelector('#byom-correlation-feature');
  state.elements.correlationOutputSelect = drawer.querySelector('#byom-correlation-output');
  state.elements.correlationInverse = drawer.querySelector('#byom-correlation-inverse');
  state.elements.correlationError = drawer.querySelector('#byom-correlation-error');

  if (!state.elements.summary) {
    const summary = document.createElement('p');
    summary.id = 'byom-summary';
    summary.className = 'byom-summary byom-hint';
    summary.hidden = true;
    state.elements.summary = summary;
    state.elements.statusText?.insertAdjacentElement('afterend', summary);
  }

  state.handlers.onTrain = typeof onTrain === 'function' ? onTrain : null;
  state.handlers.onCancel = typeof onCancel === 'function' ? onCancel : null;

  state.options.modelOptions = Array.isArray(modelOptions) ? modelOptions.slice() : [];
  populatePresetOptions();
  populateModelOptions(state.options.modelOptions);
  populateCorrelationSelectors();
  renderCorrelationList();
  updateSupportVisibility();
  updateProgress(0);
  setStatus(STATUS.IDLE);
  updateCorrelationControlsDisabledState();

  state.elements.toggle.addEventListener('click', handleToggleClick);
  state.elements.drawer.addEventListener('click', handleDrawerClick);
  state.elements.drawer.addEventListener('keydown', handleFocusTrap);
  document.addEventListener('keydown', handleGlobalKeydown);

  state.elements.closeButton?.addEventListener('click', () => closeDrawer());
  state.elements.cancelButton?.addEventListener('click', handleCancel);
  state.elements.trainButton?.addEventListener('click', handleTrain);
  state.elements.fileInput?.addEventListener('change', handleFileChange);
  state.elements.form?.addEventListener('input', handleFormInput, true);
  state.elements.form?.addEventListener('change', handleFormInput, true);
  state.elements.uploadSection?.addEventListener('dragenter', handleDragEnter);
  state.elements.uploadSection?.addEventListener('dragover', handleDragOver);
  state.elements.uploadSection?.addEventListener('dragleave', handleDragLeave);
  state.elements.uploadSection?.addEventListener('drop', handleDrop);
  state.elements.addCorrelationButton?.addEventListener('click', handleAddCorrelationClick);
  state.elements.correlationList?.addEventListener('click', handleCorrelationListClick);
  state.elements.correlationForm?.addEventListener('submit', handleCorrelationFormSubmit);
  const correlationCancelButton = state.elements.correlationDialog?.querySelector('button[data-action="cancel-correlation"]');
  correlationCancelButton?.addEventListener('click', handleCorrelationDialogCancel);
  state.elements.correlationDialog?.addEventListener('cancel', handleCorrelationDialogCancel);

  state.mounted = true;
}

export function isOpen() {
  return state.open;
}

export function open() {
  openDrawer();
}

export function close({ restoreFocus } = {}) {
  closeDrawer({ restoreFocus: restoreFocus !== false });
}

export function toggle() {
  toggleDrawer();
}

export function getState() {
  return {
    status: state.status,
    open: state.open,
    support: state.support,
    fileName: state.fileName,
  };
}

export function setModelOptions(modelOptions) {
  state.options.modelOptions = Array.isArray(modelOptions) ? modelOptions.slice() : [];
  populateModelOptions(state.options.modelOptions);
  updateStatusFromInputs();
}

export function setHandlers({ onTrain, onCancel, onPause, onResume } = {}) {
  state.handlers.onTrain = typeof onTrain === 'function' ? onTrain : null;
  state.handlers.onCancel = typeof onCancel === 'function' ? onCancel : null;
  state.handlers.onPause = typeof onPause === 'function' ? onPause : null;
  state.handlers.onResume = typeof onResume === 'function' ? onResume : null;
}

export function setTrainingStatus(status, detail) {
  setTrainingStatusInternal(status, detail);
}

export function updateTrainingProgress(progress) {
  updateTrainingProgressState(progress);
}

export function reset() {
  resetForm();
}
