# PLAN.md — Implementation Plan for **Latent Noise** Visualizer

**Reference:** See **DESIGN.md** (v1.6). This plan is optimized for AI coding agents (Codex/Copilot/Claude, etc.). Every actionable item has a checkbox. Human‑in‑the‑loop steps are minimized and grouped at the end.

> **Target repo layout**
> ```
> /latent-noise
>  ├─ index.html
>  ├─ css/style.css
>  ├─ js/
>  │   ├─ app.js
>  │   ├─ audio.js
>  │   ├─ nn.js
>  │   ├─ map.js
>  │   ├─ physics.js
>  │   ├─ render.js
>  │   ├─ presets.js
>  │   ├─ playlist.js
>  │   ├─ byom.js
>  │   ├─ training.js
>  │   └─ workers/train-worker.js
>  ├─ models/*.json
>  └─ assets/audio/*.mp3 (11 tracks)
> ```

---

## Phase 0 — Project Init & Spec Ingestion
- [x] Create repository root `/latent-noise`.
- [x] Add `README.md` with one‑paragraph summary + link to **DESIGN.md**.
- [x] Add `.gitignore` (DS_Store, *.log) and `.editorconfig` (UTF‑8, LF, 2 spaces).
- [x] Create empty files and folders per tree above.
- [x] Add `LICENSE` (MIT) with placeholder author.
- [x] Commit initial scaffold `feat: init scaffold`.

**Acceptance:**
- [x] Opening `index.html` (blank) yields **no console errors**.

---

## Phase 1 — HTML Shell (Single ESM Entry) & Basic CSS
- [x] Implement `index.html` per DESIGN.md with `<script type="module" src="js/app.js">` only.
- [x] Add minimal HUD/controls markup (Prev, Play, Next, Seek, Playlist, Volume, NN Bypass).
- [x] Create `css/style.css` with full‑screen canvas, system UI font, HUD layout, responsive rules.
- [x] Wire `#c` canvas, `#hud`, `#controls`, `#player` elements.

**Acceptance:**
- [x] Page loads with styled HUD and an empty canvas; **no JS errors**.

---

## Phase 2 — Playlist (Album‑Only Lockdown)
- [x] Implement `js/playlist.js` with static array of 11 tracks (titles/filenames from DESIGN.md).
- [x] Export `getList()`, `resolveUrl(i)`, `count()`.
- [x] In `app.js`, populate `<select id="playlist">` at boot from `playlist.getList()`.
- [x] Disable arbitrary file inputs (no drag‑and‑drop, no file picker, no mic).

**Acceptance:**
- [x] Playlist shows exactly **11 options** and cannot be modified at runtime.

---

## Phase 3 — Audio Graph & Volume Control
- [x] Implement `js/audio.js` with `init(audioEl)` that creates `AudioContext` on user gesture.
- [x] Connect `MediaElementAudioSourceNode → GainNode(volume) → AnalyserNode → destination`.
- [x] Expose `setVolume(0..1)` and persist under `ln.volume` in `localStorage`.
- [x] Configure `AnalyserNode` (FFT 2048, smoothing 0.8) and typed arrays.
- [x] Export `getAnalyser()`, `frame()` to update cached audio data.
- [x] In `app.js`, bind `#volume` slider to `audio.setVolume` (log mapping optional).

**Acceptance:**
- [x] Volume slider adjusts loudness smoothly with **no clipping** at 1.0.

---

## Phase 4 — Feature Extraction
- [x] Implement band summation for `sub, bass, lowMid, mid, high`.
- [x] Compute RMS, spectral centroid, roll-off(85%), flatness.
- [x] Compute deltas Δ(bands), Δ(RMS) and EMAs (200–400ms windows).
- [x] Optional: spectral flux → onset flags (kick/snare/hat proxies).
- [x] Export `getFeatureVector()` returning `Float32Array` (16–32 length).
- [x] Add guard rails: replace NaN/Inf with zeros; clamp ranges.

**Acceptance:**
- [x] `getFeatureVector()` returns stable values at idle and during playback.

---

## Phase 5 — Tiny NN Runtime (No External ML Libs)
- [x] Implement `js/nn.js` with MLP forward pass using `Float32Array` and preallocated buffers.
- [x] Support activations: `relu`, `elu`, `tanh`, `linear`.
- [x] Implement `loadModel(urlOrObject)`, `normalize(F)`, `forward(F_norm, out)`.
- [x] Add simple in-page tests (console): compare output of dummy model vs known reference.

**Acceptance:**
- [x] Forward pass < **0.2ms/frame** desktop; outputs finite and bounded.

---

## Phase 6 — Models Loader (Per‑Track JSON)
- [x] Create **11 model JSON** placeholders in `/models/` following DESIGN.md schema.
- [x] Map playlist entry → model path in `app.js`.
- [x] On track selection, fetch model JSON, run a warm‑up forward pass.
- [x] Cache last‑used model to avoid refetch on reselection.

**Acceptance:**
- [x] Switching tracks swaps models without visible frame hitch.

---

## Phase 7 — Mapping Layer (NN → Params)
- [x] Implement `js/map.js` mapping NN outputs → params: `spawnRate, fieldStrength, cohesion, repelImpulse, trailFade, glow, sizeJitter, hueShift, sparkleDensity, vortexAmount`.
- [x] Add critically-damped low-pass smoothing on continuous channels. *(Currently disabled per latest requirements; mapper now applies NN outputs directly without smoothing.)*
- [x] Implement hysteresis/gating on impulse-like channels.
- [x] Respect photosensitive caps from settings.

**Acceptance:**
- [x] Parameters remain within safe bounds; no chatter with silent audio.

---

## Phase 8 — Physics Core
- [x] Implement particle pool using typed arrays; free‑list allocator.
- [x] Semi‑implicit Euler integrator; global drag; soft bounds (wrap or reflect).
- [x] Forces: gravity wells, noise flow, repellers (from mapping), spring cohesion.
- [x] Spawning/killing controlled by `spawnRate`.
- [x] Implement dynamic particle cap based on frame time.

**Acceptance:**
- [x] 5k particles @ 60 FPS desktop on default quality; auto‑caps when overloaded.

---

## Phase 9 — Renderer (Canvas 2D) & HUD
- [x] DPR‑aware canvas sizing; adaptive downscale on frame drops.
- [x] Trail pass via alpha decay; particles rendered additive.
- [x] Optional glow (2-pass box blur to temp canvas).
- [x] HUD: FPS, track title/time, **volume slider**, toggles (Fullscreen, Bloom, Trails, Grid, Safe, NN Bypass).
- [x] Keyboard shortcuts per DESIGN.md.

**Acceptance:**
- [x] Visuals are smooth; HUD reflects state; toggles operate correctly.

---

## Phase 10 — Presets (Per‑Track Motifs)
- [x] Implement `js/presets.js` palettes/motifs for the 11 tracks.
- [x] Provide `getPreset(name)` and `applyPreset(preset, params)`.
- [x] Ensure presets scale NN outputs, not overwrite them.

**Acceptance:**
- [x] Clear visual distinction when switching tracks with the same audio passage.

---

## Phase 11 — App Orchestration (ES Modules)
- [x] In `js/app.js`, import modules and bootstrap audio, NN, map, physics, render, presets, playlist.
- [x] Implement main loop: `features → model.forward → map → physics → render` via `requestAnimationFrame`.
- [x] Implement handlers for play/pause/prev/next/seek; playlist change loads audio+model+preset.
- [x] Persist UI state: last track, volume, safe mode, NN bypass; fullscreen resets per session because the browser requires a user gesture.

**Acceptance:**
- [x] End-to-end system runs continuously with no memory growth.

---

## Phase 12 — Performance Guardrails & Stability
- [x] Implement FPS rolling average; adjust particle cap/resolution when FPS < target.
- [x] Pre‑allocate temp arrays; ban `new` in hot loops (lint or assert).
- [x] Debounce expensive operations on resize/visibility change.

**Acceptance:**
- [x] Under stress, app reduces quality gracefully and restores when load subsides.

---

## Phase 13 — Automated Tests & Diagnostics
- [x] Add lint config.
- [x] Add simple test harness (in page) that runs NN/Map unit checks on boot and logs pass/fail.
- [x] Add full test suite (using Jest) that runs unit tests.
- [x] Validate `/models/*.json` schema before loading; log descriptive errors.
- [x] Add `?debug=1` query flag to show overlay diagnostics (feature values, NN outputs, params).

**Acceptance:**
- [x] All tests pass; debug overlay toggles and updates live.

---

## Phase 14 — Packaging & Deployment
- [x] Ensure all paths are relative; confirm works from `file://` and HTTP.
- [x] Publish to GitHub Pages (or static host); verify identical behavior to local.

**Acceptance:**
- [x] Public URL loads and functions; console is clean.

---

## Phase 15 — Human‑In‑The‑Loop (Minimized)
- [x] Provide final **11 audio files** into `/assets/audio/` with exact filenames from DESIGN.md.
- [x] Provide artist‑approved **model weights** (`/models/*.json`) or accept auto‑generated defaults.
- [x] Review and tweak **palettes/motifs** if desired.
- [x] Approve **UX wording** (HUD labels, track titles).
- [x] Confirm **licensing/clearances** for bundling audio.

---

## Phase 16 — Immersive Intro Overlay
- [x] Introduce a launch overlay with the specified heading, narrative copy, and call-to-action button to gate playback.
- [x] Style the overlay to blanket the viewport with responsive typography, glassmorphism accents, and motion-safe transitions.
- [x] Bind the overlay play action to request fullscreen, suppress HUD chrome, and start audio playback immediately.

**Acceptance:**
- [x] On initial load the overlay is visible until Play is pressed, at which point fullscreen engages, HUD elements hide, and playback begins without manual intervention.

---

## Phase 17 — BYOM Mode UI Scaffold
- [x] Add BYOM toggle button (HUD + `Y` hotkey) that opens/closes a dedicated drawer/dialog.
- [x] Lay out BYOM form sections: file picker, preset/model dropdowns, manual tweak group, hyperparameter inputs, progress bar.
- [x] Wire basic state machine in `byom.js` (`idle → picking → ready`) with inert stubs for training hooks.
- [x] Ensure drawer is keyboard accessible, trap focus while open, and closes on escape/cancel.

**Acceptance:**
- [x] BYOM drawer opens/closes smoothly on button/hotkey; focus management passes accessibility smoke checks.
- [x] Form inputs validate required selections before enabling the Train CTA.

---

## Phase 18 — BYOM Audio Intake & Dataset Prep
- [x] Allow local MP3 selection via File Picker and drag/drop; generate/revoke Object URLs safely.
- [x] Decode audio into PCM using `AudioContext.decodeAudioData` and stream features with the existing `audio.js` extractor (offline mode).
- [x] Capture baseline targets: clone selected preset/model outputs or collect manual keyframes; partition dataset into train/validation.
- [x] Surface dataset summary in UI (duration, frame count, segments) and warn if audio duration < 30s or file exceeds size threshold.

**Acceptance:**
- [x] Selecting a file produces a dataset descriptor logged via BYOM debug overlay with correct counts.
- [x] Cancelling mid-analysis cleans up and re-enables the picker without leaks.

---

## Phase 19 — Web Worker Training Pipeline
- [x] Implement `training.js` coordinator to spawn `workers/train-worker.js`, transfer datasets, and manage lifecycle (start/pause/cancel).
- [x] Build worker gradient descent loop with learning-rate decay, gradient clipping, validation loss tracking, and progress messaging.
- [x] Update UI with live progress, estimated time remaining, and allow cancelling/resuming; ensure UI remains responsive.
- [x] Validate trained weights by running warm-up inference on the main thread and clamping outputs before activation.

**Acceptance:**
- [x] Training completes on a reference MP3 in < 2 minutes on mid-tier hardware, emitting loss curves in console or overlay.
- [x] Cancelling training terminates the worker promptly and frees transferred buffers.

---

## Phase 20 — BYOM Persistence & Playback Integration
- [x] Serialize trained model + normalization + preset metadata to DESIGN §18 schema and store in IndexedDB (`ln.byom.models`).
- [x] Expose BYOM entries in playlist UI under a separate group with ability to rename/delete entries.
- [x] On app boot, reload stored BYOM models, prompt user to re-select source file to refresh Object URL, and reinstate presets.
- [x] Start playback using the new model: audio pipeline uses user file, NN outputs drive visuals, presets apply correctly.

**Acceptance:**
- [x] After training, the new BYOM entry appears in the playlist and persists across reloads (with graceful prompt to reconnect file).
- [x] Selecting the entry plays the user audio and reproduces the trained visual behavior without console errors.

---

## Phase 21 — BYOM QA, Safety, & Polish
- [ ] Add guard rails for hyperparameters (safe ranges, warnings) and photosensitive presets in BYOM mode.
- [ ] Implement BYOM diagnostics overlay (loss curves, sample outputs) toggle via `?debug=1` or HUD control.
- [ ] Write documentation snippet (README/DEVLOG) summarizing BYOM workflow and limitations.
- [ ] Run manual regression to ensure album-only mode unaffected (playlist, performance, volume).

**Acceptance:**
- [ ] QA checklist signed off: BYOM instructions documented, safe-mode clamps verified, and legacy album playback unchanged.
- [ ] Debug overlay accurately reflects training metrics and hides by default.

---

## Appendix — Quick Agent Bootstrap
- [x] Generate stubs for every module with exported functions + TODOs.
- [x] Insert minimal CSS for layout; render FPS counter to confirm loop.
- [x] Add placeholder models with deterministic random weights for early visuals.
- [x] Record a 10‑second demo clip to validate end‑to‑end.
