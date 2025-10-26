import { createRepeatController } from '../repeat-controller.js';

test('toggle switches repeat state and notifies listeners', () => {
  const changes = [];
  const controller = createRepeatController({
    initialEnabled: false,
    onChange(value) {
      changes.push(value);
    },
  });

  expect(controller.isEnabled()).toBe(false);

  const first = controller.toggle();
  expect(first).toBe(true);
  expect(controller.isEnabled()).toBe(true);

  const second = controller.toggle();
  expect(second).toBe(false);
  expect(controller.isEnabled()).toBe(false);

  expect(changes).toEqual([true, false]);
});

test('setEnabled avoids duplicate notifications', () => {
  const changes = [];
  const controller = createRepeatController();
  const unsubscribe = controller.onChange((value) => {
    changes.push(value);
  });

  controller.setEnabled(true);
  controller.setEnabled(true);
  controller.setEnabled(false);

  expect(controller.isEnabled()).toBe(false);
  expect(changes).toEqual([true, false]);

  unsubscribe();
  controller.setEnabled(true);
  expect(changes).toEqual([true, false]);
});
