import { jest } from '@jest/globals';
import { dismiss, init, notify } from '../notifications.js';

describe('notifications', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    document.body.innerHTML = '';
    const host = init();
    if (host) {
      host.replaceChildren();
      if (!document.body.contains(host)) {
        document.body.appendChild(host);
      }
    }
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
  });

  test('init creates and reuses the notification container', () => {
    const container = init();
    expect(container).not.toBeNull();
    expect(container.id).toBe('notification-stack');
    expect(document.getElementById('notification-stack')).toBe(container);

    const second = init();
    expect(second).toBe(container);
  });

  test('notify renders a card that auto-dismisses after the duration', () => {
    const container = init();
    const card = notify('Saved successfully', { duration: 1000, tone: 'success' });

    expect(container.children).toHaveLength(1);
    expect(card.dataset.tone).toBe('success');
    expect(card.querySelector('.notification-text').textContent).toBe('Saved successfully');

    jest.advanceTimersByTime(1000);
    jest.runOnlyPendingTimers();
    expect(container.children).toHaveLength(0);
  });

  test('dismiss removes a card immediately without waiting for timeout', () => {
    const container = init();
    const card = notify('Keep around', { duration: 5000 });

    expect(container.children).toHaveLength(1);

    dismiss(card);
    jest.runOnlyPendingTimers();

    expect(container.children).toHaveLength(0);
  });
});
