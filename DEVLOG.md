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

## 2025-10-22 - Fix
- Labeled the debug overlay feature list with human-readable names and surfaced the track-position input alongside the zoom output so the new telemetry is visible when inspecting diagnostics.

## 2025-10-23 - Tooling
- Added a `npm run models` CLI that regenerates placeholder neural-network JSON files with random weights for all tracks (or a targeted subset via index/slug) so artists can quickly spin up fresh model scaffolds during experimentation.

## 2025-10-23 - Mapping Tuning
- Doubled the swing on the spawn-rate, hue-shift, and zoom mapping channels (including safe-mode envelopes) so neural outputs drive more pronounced density, color, and scale shifts while keeping the existing clamps in place; verified via `npm test` and `npm run lint` that automated checks remain green.

## 2025-10-24 - Palette Refresh
- Updated the presets for Built on the Steppers and Traffic Jam with a deep green base and complementary red/orange accents, and re-themed Backpack around gold tones with bright yellow highlights to align with the latest art direction. Validated the tweak with `npm test` and `npm run lint`.

## 2025-10-24 - Preset Baseline Fix
- Updated the mapping layer so preset-provided defaults persist as the active baselines for both continuous and impulse parameters, keeping preset scalings alive after the NN runs. Added a regression test to ensure `map.reset` honors custom baselines, and confirmed the suite stays green with `npm test` plus `npm run lint`.

## 2025-10-24 - Palette Application Fix
- Routed preset palettes through the renderer and UI so accent colors and backgrounds now update alongside per-track baselines. Added renderer unit tests that exercise palette normalization and DOM styling both before and after initialization, and verified the suite with `npm test` plus `npm run lint`.

## 2025-10-24 - Band Feature Amplification
- Amplified the analyser-derived sub-bass, low-mid, mid, and high bands before normalization so the neural network receives higher-contrast feature inputs, and remapped those channels (and their EMAs) into a signed [-1, 1] range.
- Revalidated the project health via `npm run lint` followed by `npm test`.

## 2025-10-24 - Sync Tuning
- Added a 50ms animation look-ahead so the physics and rendering layers anticipate upcoming audio events and stay aligned with the music playback.
- Confirmed no regressions via `npm test` and `npm run lint`.

## 2025-10-24 - Audio Activity Fix
- Rewired the audio graph so analyser measurements occur before gain adjustments, keeping the derived activity metric stable regardless of the HUD volume slider and ensuring diagnostics reflect the underlying track energy.
- Verified behavior with `npm test` and `npm run lint`.

## 2025-10-24 - Activity Scaling Fix
- Re-mapped the audio RMS signal to a perceptual activity scale with a -55 dB floor so debug overlays and silence detection reflect musical intensity, exposed the helper via `audio.getActivityLevel`, and updated the render loop to consume the normalized metric. Added unit coverage for the conversion and confirmed `npm test` plus `npm run lint` continue to pass.

## 2025-10-24 - Spawn Rate Silence Fix
- Updated the mapping layer so the spawn-rate channel rests at zero during silence, letting the physics system stop generating new particles when nothing is playing.
- Lowered the physics spawn-rate floor to zero so a neural output of -1 truly pauses spawning while 1 still produces the maximum emission rate.
- Verified the change set with `npm test` and `npm run lint`.

## 2025-10-24 - Spawn Rate Floor Follow-Up
- Prevented preset baselines from lifting the spawn-rate rest value so silence always decays to zero, even after preset scaling.
- Removed the application-layer minimum clamp on spawn rate so mapped values can fully reach zero when the NN or silence gating demand it.
- Re-ran `npm test` and `npm run lint` to confirm the adjustments remain stable.

## 2025-10-24 - Spawn Rate Pause Gate
- Added a playback-silence override so the mapping layer instantly resets spawn rate (and related impulse envelopes) to zero whenever the player is paused or stopped, guaranteeing no new particles appear without audio.
- Passed a new regression test that asserts the forced-silence path clamps spawn rate to rest and verified the suite with `npm test` plus `npm run lint`.

## 2025-10-25 - README Operations Guide
- Expanded the README with setup instructions, architecture overview, keyboard shortcuts, and troubleshooting notes so newcomers can run and understand the visualizer without digging into DESIGN.md.
- No code changes required verification; documentation only.

## 2025-10-24 - Idle Spawn Reset
- When loading a track without autoplay, run the mapper through a forced-silence update and reset the physics pool so the simulation starts from a clean slate with zero particles until playback begins.
- Re-verified the suite via `npm test` and `npm run lint`.

## 2025-10-24 - Mapping Smoothing Removal
- Disabled the critically-damped smoother in the mapping layer so NN outputs drive the continuous parameters directly, per the latest requirement.
- Updated the implementation plan to note the change and confirmed via `npm test` and `npm run lint`.

## 2025-10-24 - Zoom Amplification
- Amplified the mapping and render clamps for the zoom channel so neural outputs can drive a tenfold wider scale range, and adjusted application-layer scaling to boost the final zoom parameter accordingly.
- Updated preset limits and validated the expanded range through `npm test` and `npm run lint`.

## 2025-10-24 - Track Transition Intermission
- Added a reusable track-transition intermission in `app.js` that resets the physics pool and suppresses spawning for one second whenever tracks change or auto-advance, guaranteeing an empty field between songs.
- Updated playlist, HUD, and auto-advance handlers to trigger the intermission and verified the behavior with `npm run lint` and `npm test`.

## 2025-10-24 - Dropdown Track Delay
- Introduced a guarded autoplay timer so playlist dropdown changes wait one second before resuming playback, aligning the audio start with the existing particle intermission.
- Cancelled any pending autoplay attempt when switching tracks again to avoid stale resume calls and reuse the intermission constant for consistency.
- Verified with `npm run lint` and `npm test`.

## 2025-10-24 - Neural Offset Telemetry
- Extended the mapper to expose NN-driven offsets for spawn, glow, sparkle, and hue while surfacing the repeller envelope so diagnostics reflect real-time network modulation.
- Wired the application layer to track those offsets separately from manual tweaks and expanded the debug overlay to show both NN and manual contributions.
- Added unit coverage to confirm offsets respect custom baselines and follow raw network outputs; validated with `npm run lint` and `npm test`.

## 2025-10-24 - Track Skip Autoplay Delay
- Added an optional autoplay delay to the next/prev handlers so button clicks wait one second before resuming playback, matching the existing particle intermission window.
- Threaded the delay through the shared track-loading path while preserving immediate transitions for auto-advance and diagnostic controls.
- Reconfirmed repository health via `npm run lint` and `npm test`.

## 2025-10-25 - Correlation Training CLI
- Added a Node.js training utility that optimizes a per-track model for a requested feature/output correlation with optional inverse mode, deterministic seeding, and tunable training hyperparameters.
- Shared the track registry across generator and trainer scripts to avoid duplication and ensure consistent metadata lookups.
- Verified repository health with `npm run lint` and `npm test` after generating a sample session.

## 2025-10-25 - Fullscreen Toggle
- Replaced the HUD visibility checkbox with a fullscreen toggle that hides HUD, transport controls, and the debug overlay while requesting browser fullscreen.
- Added body-level fullscreen styling and accessibility attributes so UI chrome stays suppressed until fullscreen exits, including via keyboard shortcuts.
- Confirmed the refactor passes `npm run lint` and `npm test`.

## 2025-10-25 - Neural Link Rendering
- Drew thin connective strands between roughly ten percent of the active particles using deterministic pairing so the swarm hints at neural pathways.
- Tuned stroke color, alpha, and density to stay subtle behind the additive particle blooms while respecting dynamic scale and existing trail fades.
- Re-verified repository health with `npm run lint` and `npm test`.

## 2025-10-25 - Low Zoom Coverage
- Expanded the particle render scale to match the larger screen dimension and clamp the minimum zoomed-out footprint so low zoom values fill the full viewport instead of a central box.
- Kept zoom-in behavior intact by applying the new baseline before the zoom multiplier so higher zoom values still magnify the swarm responsively.
- Confirmed stability with `npm run lint` and `npm test`.

## 2025-10-25 - Keyboard Skip Delay
- Routed keyboard next/previous shortcuts through the one-second autoplay delay so transitions match the particle intermission window.
- Confirmed parity with button handlers and ensured repository health via `npm run lint` and `npm test`.

## 2025-10-25 - Playlist Select Styling
- Updated the playlist dropdown background and border to match the transport buttons so the control row has a unified visual treatment.
- Re-verified repository health with `npm run lint` and `npm test`.
