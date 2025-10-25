const DEFAULT_DURATION_MS = 5000;

let container = null;
const activeTimers = new Map();

function resolveDocument(root) {
  if (!root) {
    return typeof document !== 'undefined' ? document : null;
  }
  if (root.ownerDocument) {
    return root.ownerDocument;
  }
  if (root.documentElement) {
    return root;
  }
  return typeof document !== 'undefined' ? document : null;
}

export function init(root = document) {
  const doc = resolveDocument(root);
  if (!doc) {
    return null;
  }
  if (container && container.ownerDocument === doc) {
    return container;
  }
  const existing = doc.getElementById('notification-stack');
  if (existing) {
    container = existing;
  } else {
    container = doc.createElement('div');
    container.id = 'notification-stack';
    container.setAttribute('aria-live', 'polite');
    container.setAttribute('aria-atomic', 'true');
    if (doc.body) {
      doc.body.appendChild(container);
    }
  }
  return container;
}

function ensureContainer() {
  if (container) {
    return container;
  }
  return init();
}

const timerHost = typeof window !== 'undefined' ? window : globalThis;
const raf =
  typeof requestAnimationFrame === 'function'
    ? requestAnimationFrame
    : (cb) => timerHost.setTimeout(cb, 16);

function removeCard(card) {
  if (!card) {
    return;
  }
  const handle = () => {
    card.removeEventListener('transitionend', handle);
    if (card.parentElement) {
      card.parentElement.removeChild(card);
    }
  };
  card.addEventListener('transitionend', handle);
  card.dataset.state = 'exit';
  timerHost.setTimeout(handle, 450);
}

function clearTimer(card) {
  const timerId = activeTimers.get(card);
  if (timerId) {
    timerHost.clearTimeout(timerId);
    activeTimers.delete(card);
  }
}

function scheduleRemoval(card, duration) {
  clearTimer(card);
  const timerId = timerHost.setTimeout(() => {
    clearTimer(card);
    removeCard(card);
  }, duration);
  activeTimers.set(card, timerId);
}

export function dismiss(card) {
  if (!card) {
    return;
  }
  clearTimer(card);
  removeCard(card);
}

export function notify(message, options = {}) {
  if (!message) {
    return null;
  }
  const host = ensureContainer();
  if (!host || !host.ownerDocument) {
    console.info('[notify]', message);
    return null;
  }
  const { duration = DEFAULT_DURATION_MS, tone = 'info' } = options;
  const doc = host.ownerDocument;
  const card = doc.createElement('div');
  card.className = 'notification-card';
  card.dataset.tone = tone;
  card.dataset.state = 'enter';
  card.setAttribute('role', tone === 'error' ? 'alert' : 'status');

  const text = doc.createElement('div');
  text.className = 'notification-text';
  text.textContent = message;
  card.appendChild(text);

  const close = doc.createElement('button');
  close.type = 'button';
  close.className = 'notification-close';
  close.setAttribute('aria-label', 'Dismiss notification');
  close.innerHTML = '&times;';
  close.addEventListener('click', () => dismiss(card));
  card.appendChild(close);

  host.appendChild(card);
  raf(() => {
    card.dataset.state = 'visible';
  });

  const safeDuration = Number.isFinite(duration) && duration > 0 ? duration : DEFAULT_DURATION_MS;
  scheduleRemoval(card, safeDuration);

  card.addEventListener('pointerenter', () => {
    clearTimer(card);
  });
  card.addEventListener('pointerleave', () => {
    scheduleRemoval(card, safeDuration);
  });

  return card;
}
