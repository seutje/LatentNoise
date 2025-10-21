# AGENTS.md — Guidelines for AI Coding Agents

**Project:** Latent Noise — Audio‑Reactive Physics Visualizer  
**Specs:** Follow **DESIGN.md (v1.4)** and **PLAN.md**.  
**Purpose:** Provide unified operating procedures for AI coding agents collaborating on this repository.

---

## 1) Core Principles
- **Single source of truth:** DESIGN.md defines architecture and constraints.  
- **Plan compliance:** PLAN.md defines the executable roadmap and must remain synchronized with code progress.  
- **Autonomy:** Agents should independently execute and verify tasks before committing.  
- **ES Modules:** Maintain single entry (`js/app.js`) as the import root.  
- **Safety:** Maintain performance and photosensitive-safe behavior at all times.  
- **Transparency:** Each agent must log its work and mark progress clearly for review.

---

## 2) Workflow Overview
1. **Select target phase:** Determine which phase in PLAN.md to execute next.  
2. **Open PLAN.md:** Read the unchecked `[ ]` tasks in that phase.  
3. **Implement tasks:** Modify only the files relevant to those tasks.  
4. **Verify:** Follow the corresponding acceptance criteria in PLAN.md to confirm success.  
5. **Check off:** When a task is fully implemented and verified, change its box to `[x]` in PLAN.md.  
6. **Commit:**
   - Use a commit message in the format:  
     `feat(phase-X): description` or `fix(phase-X): description`.  
   - Include a summary of which tasks were checked off and short verification notes.  
7. **Log:** Append a brief entry to `DEVLOG.md` describing the phase and completion status.  
8. **Push/PR:** Submit your code and ensure PLAN.md reflects accurate progress.

---

## 3) Task Completion Rules
- A checkbox may be marked `[x]` **only if**:
  - Implementation meets the acceptance criteria in PLAN.md.  
  - The feature runs locally without console errors.  
  - No regression is introduced in previous phases.  
  - The user or verifier can reproduce the expected output without manual fixes.  
- Never check off a task pre‑emptively or partially. Use a separate comment `<!-- in-progress -->` if partial work must be committed.

---

## 4) Commit Standards
- Keep commits **atomic** (one phase or sub‑feature at a time).  
- Include PLAN.md diff when possible, showing new `[x]` boxes.  
- Example:
  ```
  feat(phase-3): implemented GainNode volume control
  - Checked off PLAN.md Phase 3: Audio Graph & Volume Control
  - Verified locally (smooth volume transition, no clipping)
  ```

---

## 5) Verification Guidelines
- Always test locally with a static HTTP server (e.g., `python -m http.server`).  
- Confirm console is error‑free.  
- Use browser DevTools to verify functionality matches PLAN.md acceptance notes.  
- No dependencies beyond vanilla JS/CSS/HTML are permitted.

---

## 6) Collaboration Etiquette
- Before modifying PLAN.md or DESIGN.md, open an issue or add an inline comment explaining rationale.  
- Respect existing structure and naming conventions.  
- If multiple agents are active, each should lock a phase (via comment or branch name) to avoid conflicts.  
- Always rebase/merge after pulling new PLAN.md updates.

---

## 7) Human Verification
The user can verify progress by:
- Reviewing PLAN.md to see which boxes are checked.  
- Running the project locally and confirming acceptance conditions are met.  
- Reading DEVLOG.md for recent agent activity summaries.

---

## 8) Blockers & Escalation
If an agent cannot proceed:
- Create an issue titled `BLOCKER: <summary>`.  
- Describe the dependency or ambiguity.  
- Suggest at least one resolution path.

---

**End of AGENTS.md**

