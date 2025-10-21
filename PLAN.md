# PLAN.md — Implementation Plan for **Latent Noise** Visualizer

**Reference:** See **DESIGN.md** (v1.5). This plan is optimized for AI coding agents (Codex/Copilot/Claude, etc.). Every actionable item has a checkbox. Human‑in‑the‑loop steps are minimized and grouped at the end.

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
>  │   └─ playlist.js   
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
- [x] HUD: FPS, track title/time, **volume slider**, toggles (HUD, Bloom, Trails, Grid, Safe, NN Bypass).
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
- [x] Persist UI state: last track, volume, safe mode, HUD visibility.

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

## Appendix — Quick Agent Bootstrap
- [x] Generate stubs for every module with exported functions + TODOs.
- [x] Insert minimal CSS for layout; render FPS counter to confirm loop.
- [x] Add placeholder models with deterministic random weights for early visuals.
- [x] Record a 10‑second demo clip to validate end‑to‑end.
