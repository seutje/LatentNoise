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
- [ ] Implement `js/playlist.js` with static array of 11 tracks (titles/filenames from DESIGN.md).
- [ ] Export `getList()`, `resolveUrl(i)`, `count()`.
- [ ] In `app.js`, populate `<select id="playlist">` at boot from `playlist.getList()`.
- [ ] Disable arbitrary file inputs (no drag‑and‑drop, no file picker, no mic).

**Acceptance:**
- [ ] Playlist shows exactly **11 options** and cannot be modified at runtime.

---

## Phase 3 — Audio Graph & Volume Control
- [ ] Implement `js/audio.js` with `init(audioEl)` that creates `AudioContext` on user gesture.
- [ ] Connect `MediaElementAudioSourceNode → GainNode(volume) → AnalyserNode → destination`.
- [ ] Expose `setVolume(0..1)` and persist under `ln.volume` in `localStorage`.
- [ ] Configure `AnalyserNode` (FFT 2048, smoothing 0.8) and typed arrays.
- [ ] Export `getAnalyser()`, `frame()` to update cached audio data.
- [ ] In `app.js`, bind `#volume` slider to `audio.setVolume` (log mapping optional).

**Acceptance:**
- [ ] Volume slider adjusts loudness smoothly with **no clipping** at 1.0.

---

## Phase 4 — Feature Extraction
- [ ] Implement band summation for `sub, bass, lowMid, mid, high`.
- [ ] Compute RMS, spectral centroid, roll‑off(85%), flatness.
- [ ] Compute deltas Δ(bands), Δ(RMS) and EMAs (200–400ms windows).
- [ ] Optional: spectral flux → onset flags (kick/snare/hat proxies).
- [ ] Export `getFeatureVector()` returning `Float32Array` (16–32 length).
- [ ] Add guard rails: replace NaN/Inf with zeros; clamp ranges.

**Acceptance:**
- [ ] `getFeatureVector()` returns stable values at idle and during playback.

---

## Phase 5 — Tiny NN Runtime (No External ML Libs)
- [ ] Implement `js/nn.js` with MLP forward pass using `Float32Array` and preallocated buffers.
- [ ] Support activations: `relu`, `elu`, `tanh`, `linear`.
- [ ] Implement `loadModel(urlOrObject)`, `normalize(F)`, `forward(F_norm, out)`.
- [ ] Add simple in‑page tests (console): compare output of dummy model vs known reference.

**Acceptance:**
- [ ] Forward pass < **0.2ms/frame** desktop; outputs finite and bounded.

---

## Phase 6 — Models Loader (Per‑Track JSON)
- [ ] Create **11 model JSON** placeholders in `/models/` following DESIGN.md schema.
- [ ] Map playlist entry → model path in `app.js`.
- [ ] On track selection, fetch model JSON, run a warm‑up forward pass.
- [ ] Cache last‑used model to avoid refetch on reselection.

**Acceptance:**
- [ ] Switching tracks swaps models without visible frame hitch.

---

## Phase 7 — Mapping Layer (NN → Params)
- [ ] Implement `js/map.js` mapping NN outputs → params: `spawnRate, fieldStrength, cohesion, repelImpulse, trailFade, glow, sizeJitter, hueShift, sparkleDensity, vortexAmount`.
- [ ] Add critically‑damped low‑pass smoothing on continuous channels.
- [ ] Implement hysteresis/gating on impulse‑like channels.
- [ ] Respect photosensitive caps from settings.

**Acceptance:**
- [ ] Parameters remain within safe bounds; no chatter with silent audio.

---

## Phase 8 — Physics Core
- [ ] Implement particle pool using typed arrays; free‑list allocator.
- [ ] Semi‑implicit Euler integrator; global drag; soft bounds (wrap or reflect).
- [ ] Forces: gravity wells, noise flow, repellers (from mapping), spring cohesion.
- [ ] Spawning/killing controlled by `spawnRate`.
- [ ] Implement dynamic particle cap based on frame time.

**Acceptance:**
- [ ] 5k particles @ 60 FPS desktop on default quality; auto‑caps when overloaded.

---

## Phase 9 — Renderer (Canvas 2D) & HUD
- [ ] DPR‑aware canvas sizing; adaptive downscale on frame drops.
- [ ] Trail pass via alpha decay; particles rendered additive.
- [ ] Optional glow (2‑pass box blur to temp canvas).
- [ ] HUD: FPS, track title/time, **volume slider**, toggles (HUD, Bloom, Trails, Grid, Safe, NN Bypass).
- [ ] Keyboard shortcuts per DESIGN.md.

**Acceptance:**
- [ ] Visuals are smooth; HUD reflects state; toggles operate correctly.

---

## Phase 10 — Presets (Per‑Track Motifs)
- [ ] Implement `js/presets.js` palettes/motifs for the 11 tracks.
- [ ] Provide `getPreset(name)` and `applyPreset(preset, params)`.
- [ ] Ensure presets scale NN outputs, not overwrite them.

**Acceptance:**
- [ ] Clear visual distinction when switching tracks with the same audio passage.

---

## Phase 11 — App Orchestration (ES Modules)
- [ ] In `js/app.js`, import modules and bootstrap audio, NN, map, physics, render, presets, playlist.
- [ ] Implement main loop: `features → model.forward → map → physics → render` via `requestAnimationFrame`.
- [ ] Implement handlers for play/pause/prev/next/seek; playlist change loads audio+model+preset.
- [ ] Persist UI state: last track, volume, safe mode, HUD visibility.

**Acceptance:**
- [ ] End‑to‑end system runs continuously with no memory growth.

---

## Phase 12 — Performance Guardrails & Stability
- [ ] Implement FPS rolling average; adjust particle cap/resolution when FPS < target.
- [ ] Pre‑allocate temp arrays; ban `new` in hot loops (lint or assert).
- [ ] Debounce expensive operations on resize/visibility change.

**Acceptance:**
- [ ] Under stress, app reduces quality gracefully and restores when load subsides.

---

## Phase 13 — Automated Tests & Diagnostics
- [ ] Add simple test harness (in page) that runs NN/Map unit checks on boot and logs pass/fail.
- [ ] Validate `/models/*.json` schema before loading; log descriptive errors.
- [ ] Add `?debug=1` query flag to show overlay diagnostics (feature values, NN outputs, params).

**Acceptance:**
- [ ] All tests pass; debug overlay toggles and updates live.

---

## Phase 14 — Packaging & Deployment
- [ ] Ensure all paths are relative; confirm works from `file://` and HTTP.
- [ ] Add optional `manifest.json` (name/icons/display=standalone).
- [ ] Publish to GitHub Pages (or static host); verify identical behavior to local.

**Acceptance:**
- [ ] Public URL loads and functions; console is clean.

---

## Phase 15 — Human‑In‑The‑Loop (Minimized)
- [ ] Provide final **11 audio files** into `/assets/audio/` with exact filenames from DESIGN.md.
- [ ] Provide artist‑approved **model weights** (`/models/*.json`) or accept auto‑generated defaults.
- [ ] Review and tweak **palettes/motifs** if desired.
- [ ] Approve **UX wording** (HUD labels, track titles).
- [ ] Confirm **licensing/clearances** for bundling audio.

---

## Appendix — Quick Agent Bootstrap
- [ ] Generate stubs for every module with exported functions + TODOs.
- [ ] Insert minimal CSS for layout; render FPS counter to confirm loop.
- [ ] Add placeholder models with deterministic random weights for early visuals.
- [ ] Record a 10‑second demo clip to validate end‑to‑end.
