export function createRepeatController(options = {}) {
  const listeners = new Set();
  if (typeof options.onChange === 'function') {
    listeners.add(options.onChange);
  }
  let enabled = Boolean(options.initialEnabled);

  function emit() {
    for (const listener of listeners) {
      try {
        listener(enabled);
      } catch (error) {
        // Surface listener errors without stopping other listeners.
        console.error('[repeat] listener error', error);
      }
    }
  }

  return {
    isEnabled() {
      return enabled;
    },
    setEnabled(value) {
      const nextValue = Boolean(value);
      if (nextValue === enabled) {
        return enabled;
      }
      enabled = nextValue;
      emit();
      return enabled;
    },
    toggle() {
      enabled = !enabled;
      emit();
      return enabled;
    },
    onChange(listener) {
      if (typeof listener !== 'function') {
        return () => {};
      }
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}
