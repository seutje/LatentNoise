# DESIGN.md — Latent Noise Visualizer

**Project:** Latent Noise — Audio‑Reactive Physics Visuals  
**Scope:** Multi‑file JavaScript simulation with separate CSS and JS modules. Playback is restricted to bundled album tracks only. Visuals are driven by a tiny per‑track neural network (NN) that maps audio features → visual parameters.  
**ES Modules:** The HTML references **a single ES module entry (`js/app.js`)**. All other modules are imported via ES module `import` statements.  
**Author:** (ChatGPT-5)  
**Version:** 1.6

---

## 1) Objectives
- Deliver a modular HTML application with **ES modules**; the HTML page loads **one** `<script type="module" src="js/app.js"></script>` and nothing else.  
- Each track ships with its own small, pre‑trained **neural network** (lightweight MLP) that converts real‑time audio features to sim parameters.  
- Audio‑reactive visuals (Canvas 2D + Web Audio) with **60 FPS target** desktop and **>30 FPS** mobile.  
- **Primary experience** defaults to the Latent Noise album (11 bundled tracks).
- Per‑track **models + presets** define unique behavior and look.  
- Zero build tools required; open `index.html` and choose a track.
- Provide a sandboxed **BYOM (Bring Your Own Music) mode** that lets users load a local MP3, assign presets or clone an existing model, optionally tweak training hyperparameters, and produce an in-browser trained model via Web Workers.

---

## 2) Constraints & Non‑Goals
- **Minimal dependencies:** no runtime frameworks; no TF.js.  
- No networked audio inputs; BYOM accepts **local files only** via the File Picker / drag & drop sandbox.
- Canvas 2D only (no heavy WebGL).  
- Lightweight particle physics (no external engine).  
- Optional localStorage for last track/preset/volume/HUD state.

---

## 3) User Stories
1. I can select a **Latent Noise** track; its dedicated NN drives the visuals.  
2. I can switch tracks via UI or hotkeys; models swap seamlessly.  
3. I can tweak high‑level parameters (intensity, particle count, palette shift) that **scale** NN outputs.  
4. I can pause/resume and seek within the current track.  
5. **I can control volume** via a slider in the HUD.
6. I can enter **BYOM mode**, pick a local MP3, reuse or customize presets/models, and watch the system adapt after in-browser training completes.

---

## 4) Tech Overview
- **Rendering:** HTML5 Canvas 2D, `requestAnimationFrame`, composite modes (`lighter`, `screen`, `multiply`).  
- **Audio:** Web Audio API (`AudioContext`, `MediaElementAudioSourceNode`, `AnalyserNode`, `GainNode` for **volume slider**), FFT 2048–4096, `smoothingTimeConstant ~0.8`.  
- **Input:** `HTMLAudioElement` (bundled album sources by default, or Object URLs produced by BYOM mode).  
- **Neural layer:** Tiny **MLP** per track (e.g., `[F] → 32 ReLU → 16 ReLU → P tanh`) implemented in `nn.js` with typed arrays; weights as JSON.  
- **State:** `AppState` orchestrates audio, NN, physics, render, playlist, and UI.
- **BYOM tooling:** File API (Object URLs), Web Workers for background training, and IndexedDB storage for user models/presets.

---

## 5) Architecture
```
index.html
  └─ <script type="module" src="js/app.js"> (single entry)
js/app.js (ESM)
  ├─ import './audio.js'
  ├─ import './nn.js'
  ├─ import './map.js'
  ├─ import './physics.js'
  ├─ import './render.js'
  ├─ import './presets.js'
  └─ import './playlist.js'
```

```
+-----------------------+   +-------------------+   +--------------------+   +----------------+
|  HTMLAudioElement     |   |  Audio Analysis   |   |  Per‑Track NN      |   |  Physics Core  |
|  (album track only)   |-> | (FFT, bands, RMS) |-> | (features -> params)|->| (particles, fx)|
+-----------------------+   +-------------------+   +--------------------+   +----------------+
                                                                    |
                                                                    v
                                                            +---------------+
                                                            | Renderer 2D   |
                                                            +---------------+
```

### File Structure
```
/latent-noise
 ├── index.html
 ├── css/
 │   └── style.css
 ├── js/
 │   ├── app.js          // single ESM entry, wires everything
 │   ├── audio.js        // Web Audio + features + GainNode volume
 │   ├── nn.js           // tiny MLP runtime + loaders
 │   ├── map.js          // NN outputs → sim params (clamps/smoothing)
 │   ├── physics.js      // particles, fields, integrator
 │   ├── render.js       // canvas passes, trails, glow, HUD (incl. volume slider)
 │   ├── presets.js      // visual motifs & constants per track
 │   ├── playlist.js     // fixed album playlist
 │   ├── byom.js         // BYOM mode state machine + UI glue
 │   ├── training.js     // training coordinator (main thread)
 │   └── workers/
 │       └── train-worker.js // Web Worker running gradient descent loop
 ├── models/             // per‑track NN weights & normalization
 └── assets/
     └── audio/ (11 album tracks)
```

### BYOM Mode Architecture
- `byom.js` flips between album playback and BYOM states (`idle → analyzing → training → ready`) while keeping `app.js` as the single orchestration hub.
- Local files enter through the File Picker or drag & drop; we wrap the selected `File` in an Object URL and hand it to `audio.js` (no network ever).
- Feature extraction, mapping, physics, and rendering pipelines remain unchanged once BYOM hands off a trained model; only data sources differ.

#### Workflow
1. User enables BYOM mode, selects a local MP3, and optionally chooses a baseline (existing preset/model combo or manual preset sliders).
2. `training.js` decodes the entire track via `AudioContext.decodeAudioData`, streams feature frames, and builds a training dataset (features + target params). Targets come either from the chosen baseline model (teacher forcing) or user-entered preset curves.
3. Dataset arrays, metadata, and hyperparameters (`epochs`, `learningRate`, `batchSize`, `regularization`, `seed`) are transferred to `workers/train-worker.js` via `postMessage` with transferable buffers.
4. `train-worker.js` performs batched gradient descent on the MLP (same schema as DESIGN §18) using typed arrays. It posts progress messages (`epoch`, `loss`, `valLoss`, `samplePreview`) for UI feedback.
5. On completion the worker returns the trained weights + normalization stats. `training.js` validates, serializes to the existing model JSON schema, and supplies it to `nn.js` for warm-up.
6. `byom.js` registers the new model/preset under a generated ID, stores it in IndexedDB (`ln.byom.models` store), and switches the active playback pipeline to the user track.

#### Preset & Model Controls
- Users can clone presets/models from bundled tracks, tweak scalars (spawn rate bias, palette, turbulence) before training, or switch to "manual" mode where they keyframe targets per section in a compact timeline UI.
- Hyperparameter UI exposes safe ranges with guarded defaults; sliders/inputs map to the worker payload.
- Training can be paused/cancelled; `training.js` aborts the worker and releases buffers if requested.

#### Storage & Persistence
- Trained models persist in IndexedDB along with preset metadata and a hash of the source file name + size for quick reuse.
- Lightweight entries (<1 MB) are mirrored to `localStorage` for quick bootstrapping; full weight blobs stay in IndexedDB.
- BYOM entries appear in the playlist under a separate group, selectable across sessions (provided the original file path is reselected so the Object URL can be refreshed).

#### Compatibility
- If Web Workers or IndexedDB are unavailable, BYOM mode is disabled with an explanatory tooltip; the baseline album experience remains intact.

---

## 6) Audio Feature Extraction
- **Feature vector F (per frame):** bands energies (sub..high), RMS, centroid, roll‑off, flatness, deltas, EMAs, optional onset flags. `|F| ≈ 16–32`.  
- **Normalization:** Per‑model mean/std → `F_norm`.
- **BYOM preprocessing:** Offline pass walks the full track at 60 Hz, aggregates feature stats, and bins frames into sections for supervised training and validation splits (80/20).

---

## 7) Neural Response Layer (Per‑Track Models)
- Default architecture `|F| → 32 ReLU → 16 ReLU → P tanh`.  
- **Outputs `Y` (−1..1)** mapped to: `spawnRate, fieldStrength, cohesion, repelImpulse, trailFade, glow, sizeJitter, hueShift, sparkleDensity, vortexAmount` (extendable).  
- **Stability:** low‑pass smoothing, hysteresis for impulses, safety clamps (photosensitive‑safe).
- **Training constraints (BYOM):** Worker enforces gradient clipping, learning-rate decay, and early stopping once validation loss plateaus for 5 epochs to avoid runaway weights.

---

## 8) Physics Design
- Particle struct, semi‑implicit Euler, global drag, soft bounds, wells, noise flow, repellers, springs.  
- Spawn rate = base + NN channel; pooling + typed arrays; glow via 2‑pass blur.

---

## 9) Rendering Pipeline (per frame)
1. Read audio → compute features `F`.  
2. Normalize `F`; **NN forward** → `Y`.  
3. Map `Y` (+ preset scaling) → `Params`.  
4. Spawn/kill → integrate → render (trails/glow).  
5. Update HUD (track, time, FPS, **volume**).

---

## 10) Controls & UX
- **Track select:** number keys `1–0` + `-` or playlist panel.  
- **Playback:** `Space` play/pause, `N` next, `P` previous, arrows seek.  
- **Toggles:** `H` HUD, `B` Bloom, `T` Trails, `G` Grid, `P` Photosensitive safe, `K` NN bypass.  
- **Knobs:** `[`/`]` particles, `;`/`'` intensity, `,`/`.` palette.  
- **BYOM mode toggle:** `Y` hotkey or HUD button opens the BYOM drawer (drawer overlays controls on desktop, full-screen sheet on mobile).
- **BYOM panel:** file picker, preset/model selector, manual sliders, hyperparameter inputs (epochs, learning rate, batch size, regularization), and a progress bar with cancel/resume.
- **HUD Volume Slider:** range 0–100% controlling a **GainNode** (log‑scaled for perceptual linearity). Persist last value.

---

## 11) Per‑Track Identity (Models + Motifs)
1. Meditation — slow‑response NN; haze motif.  
2. Built on the Steppers — steady grid; cohesion‑biased NN.  
3. Unsound — transient‑sensitive mid/high; shockwaves.  
4. System.js — vortex & centroid‑driven hue.  
5. Binary Mirage — inward bias; RMS→glow.  
6. Traffic Jam — stop‑go gating; lane flows.  
7. Backpack — transient focus; short life.  
8. Last Pack — flock alignment; arcs.  
9. Clouds — buoyant; soft bloom.
10. Ease Up — sway; long trails.  
11. Epoch ∞ — sparse; slow hue drift.
12. BYOM Library — dynamic slots generated per user upload; display source filename and training preset/model summary.

---

## 12) Public API
```js
// Exposed by app.js (ESM)
export function playTrack(indexOrName) {}
export function play() {}
export function pause() {}
export function toggle() {}
export function next() {}
export function prev() {}
export function seek(seconds) {}
export function setVolume(linear0to1) {}
export function loadModel(name) {}
export function loadPreset(nameOrIndex) {}
export function setParam(key, value) {}
export function setNNBypass(on) {}
export function enterBYOMMode() {}
export function trainBYOM(options) {}
export function listBYOMEntries() {}
export function loadBYOMEntry(id) {}
export function cancelBYOMTraining() {}
```

---

## 13) Performance Budget & Tuning
- **NN inference:** < 0.2 ms/frame desktop; reuse buffers; no GC in loop.  
- **Particles:** 1k–12k with dynamic cap.  
- **Canvas:** fixed internal res; adaptive downscale.  
- **Audio FFT:** 2048 default; 4096 if headroom.
- **Training worker (BYOM):** batches 256 frames, aims for < 16 ms/iteration; yields every epoch to keep UI responsive; main thread throttles dataset streaming to 30 MB/s.

---

## 14) Accessibility & Safety
- Safe mode limits luminance/flash rate and clamps NN outputs.  
- Color‑blind palettes and legible HUD.  
- Volume slider defaults to 70%; keyboard accessible.
- BYOM presets inherit safe-mode clamps; manual inputs are limited to photosensitive-safe ranges with inline warnings.

---

## 15) Security & Permissions
- No mic; no network required.  
- Local files for audio and models.
- BYOM never forwards file contents externally; Object URLs revoked after playback and IndexedDB entries store only derived weights/metadata (no raw audio).

---

## 16) Fallbacks
- Autoplay policy gate (user gesture).  
- NN bypass key `K` for slow devices.
- BYOM toggle hides when File API, Web Workers, or IndexedDB are unsupported; UI explains fallback.

---

## 17) Implementation Sketch
```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Latent Noise Visualizer</title>
  <link rel="stylesheet" href="css/style.css">
</head>
<body>
  <canvas id="c"></canvas>
  <div id="controls" class="ui">
    <button id="prev">Prev</button>
    <button id="play">Play</button>
    <button id="next">Next</button>
    <input id="seek" type="range" min="0" max="100" value="0">
    <select id="playlist"></select>
    <label class="vol">Volume <input id="volume" type="range" min="0" max="1" step="0.01" value="0.7"></label>
    <label><input type="checkbox" id="bypass"> NN Bypass</label>
    <button id="byomToggle" type="button">BYOM</button>
  </div>
  <dialog id="byomDrawer" class="byom">
    <form method="dialog">
      <header>Bring Your Own Music</header>
      <input id="byomFile" type="file" accept="audio/mpeg">
      <section class="baseline">
        <label>Preset <select id="byomPreset"></select></label>
        <label>Model <select id="byomModel"></select></label>
      </section>
      <section class="manual">
        <details>
          <summary>Manual tweaks</summary>
          <div class="sliders">...</div>
        </details>
      </section>
      <section class="hyper">
        <label>Epochs <input id="byomEpochs" type="number" min="1" max="500" value="50"></label>
        <label>Learning Rate <input id="byomLR" type="number" step="0.0001" value="0.001"></label>
        <label>Batch Size <input id="byomBatch" type="number" min="32" max="1024" value="256"></label>
        <label>L2 <input id="byomL2" type="number" step="0.0001" value="0.0005"></label>
      </section>
      <progress id="byomProgress" value="0" max="1"></progress>
      <footer>
        <button type="button" id="byomCancel">Cancel</button>
        <button type="button" id="byomTrain">Train</button>
      </footer>
    </form>
  </dialog>
  <div id="hud" hidden></div>
  <audio id="player" preload="metadata"></audio>
  <script type="module" src="js/app.js"></script>
</body>
</html>
```

### app.js (ES module) sketch
```js
import * as Audio from './audio.js';
import * as NN from './nn.js';
import * as Map from './map.js';
import * as Phys from './physics.js';
import * as Render from './render.js';
import * as Presets from './presets.js';
import * as Playlist from './playlist.js';
import * as BYOM from './byom.js';
import * as Training from './training.js';

// wire volume slider → GainNode
const vol = document.getElementById('volume');
Audio.setVolume(+vol.value);
vol.addEventListener('input', () => Audio.setVolume(+vol.value));

BYOM.mount({ drawer: document.getElementById('byomDrawer') });

// main loop skeleton...
```

---

## 18) Model JSON Schema (unchanged)
```json
{
  "norm": {"mean": [..], "invStd": [..]},
  "layers": [
    {"W": [[..],[..]], "b": [..], "act": "relu"},
    {"W": [[..],[..]], "b": [..], "act": "relu"},
    {"W": [[..],[..]], "b": [..], "act": "tanh"}
  ],
  "meta": {"name": "meditation", "inputs": 24, "outputs": 10}
}
```

---

## 19) Testing & Deployment
- Confirm single‑script ESM boot works across browsers (Chrome, Edge, Safari, Firefox).  
- Verify volume slider maps to **GainNode** and persists to localStorage.  
- Ensure each track loads its correct model; profile frame time.  
- Confirm BYOM UI hides gracefully when prerequisites (File API, Workers, IndexedDB) are absent.
- Exercise BYOM flow: load sample MP3, run training (<=2 min on mid-tier laptop), ensure worker progress updates and resulting model persists to IndexedDB and reloads after refresh.
- Static hosting (GitHub Pages, etc.).

---
