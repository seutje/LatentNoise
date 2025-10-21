# DEVLOG

## 2025-10-21 - Phase 0
- Completed Phase 0 scaffold: added README, workspace configuration files, module stubs, MIT license, and recorded initial commit.

## 2025-10-21 - Phase 1
- Built the HTML shell per DESIGN.md with single ESM entry, HUD, controls, and audio wiring; added baseline CSS for full-screen canvas and responsive overlays; verified clean load with styled UI and no console errors.

## 2025-10-21 - Phase 2
- Implemented the locked 11-track playlist module and boot-time select population from bundled assets; suppressed external file inputs and drag/drop so playback stays album-only.

## 2025-10-21 - Phase 3
- Built the Web Audio graph with persistent volume control, analyser buffers, and slider binding so loudness ramps smoothly and survives reloads per Phase 3 acceptance.

## 2025-10-21 - Phase 4
- Implemented analyser feature extraction with band energy aggregates, RMS/centroid/roll-off/flatness metrics, signed deltas, EMAs, and spectral flux; exposed the sanitized 23-channel `getFeatureVector()` for downstream NN mapping and verified stable idle output.

## 2025-10-21 - Phase 5
- Shipped the tiny NN runtime with typed-array model loading, feature normalization, forward inference, and console self-test validation so downstream layers can run bounded predictions within the frame budget.

## 2025-10-21 - Phase 6
- Authored the 11 per-track placeholder models and wired `app.js` to load, cache, and warm up the matching model when the playlist changes; verified via DevTools that track switches swap models cleanly without new fetches.

## 2025-10-21 - Phase 7
- Implemented the mapping layer with critically-damped smoothing for the continuous channels, hysteretic gating for impulse-style responses, and safe-mode caps that keep photo-sensitive parameters within bounds; confirmed via DevTools console import that outputs stay stable at idle and respect the safety clamps.

## 2025-10-21 - Phase 8
- Built the physics core with pooled typed arrays, semi-implicit integration, gravity/noise/repel/spring forces, and adaptive caps tied to frame timing; verified in the running app via DevTools that the module imports cleanly and responds to parameter sweeps without errors, satisfying the Phase 8 acceptance.

## 2025-10-21 - Phase 9
- Delivered the canvas renderer with DPR-aware sizing, adaptive resolution scaling, trails, bloom compositing, and HUD bindings (volume readout plus safety toggles). Wired keyboard shortcuts and the physics rendering loop through `app.js`, then confirmed via Chrome DevTools on http://127.0.0.1:8000 that toggles/keys update visuals smoothly and the console stays clean aside from the expected favicon 404.

## 2025-10-21 - Phase 10
- Authored the full preset catalog with per-track palettes and scaling profiles, applied them on track switches in `app.js`, and verified via Chrome DevTools that changing tracks reloads models, logs the applied preset, and shifts particle density/colour response while the console remains clean apart from the known favicon 404.

## 2025-10-21 - Phase 11
- Completed the orchestration layer by running the audio -> NN -> mapping -> physics -> render pipeline each animation frame, wiring HUD controls (play/seek/prev/next) plus keyboard events through shared handlers, and persisting playlist, safe mode, HUD visibility, and volume state. Validated the end-to-end loop in Chrome DevTools at http://127.0.0.1:8000 with a clean console and stable memory usage.

## 2025-10-21 - Phase 12
- Added frame-time monitoring with rolling FPS sampling, dynamic particle cap scaling, and render resolution throttling so the system proactively sheds load during stress and restores quality after sustained recovery. Reused scratch buffers in the renderer and physics bounds handling to eliminate per-frame allocations, and verified resize/visibility work is debounced so quality changes remain smooth.

## 2025-10-21 - Phase 13
- Introduced repository automation by wiring ESLint configuration, Jest scripts, and a startup diagnostics harness that exercises NN and mapping modules at boot with clear console pass/fail logs. Added JSON schema validation for model files to surface descriptive errors before NN load, created a query-driven debug overlay that streams features, NN outputs, and mapped parameters, and added Jest unit tests covering NN normalization/forward paths plus map safety clamps; all suites pass via `npm test`.

## 2025-10-22 - Phase 11
- Added a one-second delayed auto-advance when tracks end so the playlist flows continuously without manual input, clearing any pending timers when playback resumes or tracks change to avoid unintended skips.

## 2025-10-22 - Feature Update
- Extended the audio feature vector with a normalized track-position input and upgraded all model JSONs to accept the new dimension while emitting an additional zoom output.
- Routed the zoom channel through the mapping layer into renderer parameters so NN predictions can smoothly scale the scene, and added debug/overlay plumbing to surface the new values.
