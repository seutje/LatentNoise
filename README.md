# Latent Noise

Latent Noise is a browser-based audio-reactive physics visualizer that combines a curated playlist with real-time neural mapping and particle dynamics, built with vanilla HTML/CSS/JS and the Web Audio API. Refer to the [Design Specification](DESIGN.md) for architecture, constraints, and detailed feature requirements.

## Getting Started

### Prerequisites
- **Node.js 18+** (ships with `npm`).
- Audio playback capability in the browser (the visualizer uses the Web Audio API and requires user interaction to start playback).

### Install dependencies
```bash
npm install
```

### Run the development server
Start a static server from the project root:
```bash
npm start
```
Then open [http://localhost:8000](http://localhost:8000) in a Chromium, Firefox, or Safari browser. The visualizer boots once you click **Play** or press **Space** to unlock audio; the canvas will remain idle until the browser receives a user gesture.

> **Alternative:** Any static HTTP server works (for example, `python -m http.server 8000`). The app is fully client-side and does not require a backend.

### Optional scripts
- `npm run lint` — ESLint over the JavaScript source.
- `npm test` — Jest suite for modules that expose test hooks.
- `npm run models` / `npm run train` — utilities for regenerating neural-network assets.

## How the visualizer works
Latent Noise follows the pipeline defined in the design document:

1. **Playlist & presets (`js/playlist.js`, `js/presets.js`)** — The UI exposes an album-locked playlist of 11 tracks. Selecting a track also selects its visual motif and neural model metadata.
2. **Audio graph (`js/audio.js`)** — After the first user gesture, an `AudioContext` spins up connecting the `<audio>` element through a `GainNode` to an `AnalyserNode`. The analyser produces FFT data that is cached each animation frame.
3. **Feature extraction (`js/audio.js`)** — Frequency bands, RMS, spectral centroid, roll-off, flatness, deltas, and exponential moving averages are condensed into a fixed-length feature vector.
4. **Neural response (`js/nn.js`, `models/*.json`)** — Each track loads a tiny MLP described by its JSON weights. The feature vector is normalized and run through the network to produce parameters in the range -1..1.
5. **Mapping layer (`js/map.js`)** — Neural outputs map to simulation parameters such as spawn rate, field strength, trail fade, glow, and hue shift. Safety clamps ensure photosensitive-friendly behavior.
6. **Physics core (`js/physics.js`)** — A pooled particle system integrates forces (gravity wells, noise flow, repellers, cohesion) with semi-implicit Euler steps and adaptive particle caps based on frame time.
7. **Renderer & HUD (`js/render.js`, `css/style.css`)** — Canvas 2D draws additive particle trails, optional glow passes, and overlays the HUD with FPS, track metadata, and toggles. DPR-aware resizing keeps performance stable.
8. **App orchestration (`js/app.js`)** — Wires all modules together, persists UI state to `localStorage`, and drives the main `requestAnimationFrame` loop.

## Keyboard & HUD controls
| Action | Shortcut |
| --- | --- |
| Play / pause | `Space`
| Next / previous track | `N` / `P`
| Seek forward / backward | Right / Left Arrow |
| Direct track selection | `1`–`0`, `-` |
| Toggle fullscreen | `F` |
| Toggle HUD | `H` |
| Adjust particle count | `[` / `]` |
| Adjust intensity | `;` / `'` |
| Cycle palette | `,` / `.` |

HUD controls mirror these shortcuts: the volume slider controls the `GainNode` (persisted across sessions), a dedicated fullscreen button sits before playback controls, and buttons in the toolbar handle play, pause, previous/next, seeking, and playlist selection.

## Troubleshooting
- **No audio or visuals?** Ensure you have clicked inside the page (audio contexts must be unlocked by a gesture) and confirm the browser has access to audio output.
- **Performance dips?** Let the adaptive quality scaling respond, or nudge particle density and intensity with the bracket and semicolon/quote shortcuts.
- **Saved settings missing?** The app stores volume, last track, safe mode, and NN bypass in `localStorage`. Clearing site data resets them.

---

Latent Noise is licensed under the MIT License. See [LICENSE](LICENSE) for details.
