import { listPresets } from './presets.js';
import { analyzeFile } from './byom-intake.js';
import { logByomDataset } from './diagnostics.js';

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
  },
  options: {
    modelOptions: [],
  },
  handlers: {
    onTrain: null,
    onCancel: null,
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
  const messageFactory = STATUS_MESSAGES[state.status] ?? STATUS_MESSAGES[STATUS.IDLE];
  if (state.elements.statusText) {
    state.elements.statusText.textContent = messageFactory({
      fileName: state.fileName,
      summary: state.datasetSummary,
      errorMessage: state.lastError && typeof state.lastError.message === 'string'
        ? state.lastError.message
        : typeof state.lastError === 'string'
          ? state.lastError
          : undefined,
    });
  }
  updateSummaryDisplay();
}

function updateTrainAvailability() {
  const ready = state.status === STATUS.READY;
  if (state.elements.trainButton) {
    state.elements.trainButton.disabled = !ready;
    state.elements.trainButton.setAttribute('aria-disabled', ready ? 'false' : 'true');
  }
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
  targets.forEach((el) => {
    if (!el) {
      return;
    }
    if (disabled) {
      el.setAttribute('aria-disabled', 'true');
      el.disabled = true;
    } else {
      el.removeAttribute('aria-disabled');
      el.disabled = false;
    }
  });
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
    state.elements.modelSelect.selectedIndex = 0;
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
  resetForm();
  if (typeof state.handlers.onCancel === 'function') {
    state.handlers.onCancel();
  }
  closeDrawer();
}

function handleTrain(event) {
  event.preventDefault();
  if (state.status !== STATUS.READY) {
    return;
  }
  if (typeof state.handlers.onTrain === 'function') {
    state.handlers.onTrain({
      file: state.file,
      objectUrl: state.objectUrl,
      preset: state.elements.presetSelect?.value ?? '',
      model: state.elements.modelSelect?.value ?? '',
      dataset: state.dataset,
      summary: state.datasetSummary,
    });
  } else {
    console.info(
      '[byom] Train requested (stub)',
      state.fileName,
      state.elements.presetSelect?.value,
      state.elements.modelSelect?.value,
      state.datasetSummary,
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

function populateModelOptions(modelOptions) {
  if (!state.elements.modelSelect) {
    return;
  }
  const options = Array.isArray(modelOptions) ? modelOptions : state.options.modelOptions;
  state.elements.modelSelect.innerHTML = '';
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = 'Select model';
  placeholder.disabled = true;
  placeholder.selected = true;
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

  state.elements.modelSelect.selectedIndex = 0;
}

function updateSupportVisibility() {
  if (!state.elements.toggle) {
    return;
  }
  if (state.support) {
    state.elements.toggle.removeAttribute('disabled');
    state.elements.toggle.removeAttribute('aria-disabled');
    state.elements.toggle.title = '';
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
  updateSupportVisibility();
  updateProgress(0);
  setStatus(STATUS.IDLE);

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

export function setHandlers({ onTrain, onCancel } = {}) {
  state.handlers.onTrain = typeof onTrain === 'function' ? onTrain : null;
  state.handlers.onCancel = typeof onCancel === 'function' ? onCancel : null;
}

export function reset() {
  resetForm();
}
