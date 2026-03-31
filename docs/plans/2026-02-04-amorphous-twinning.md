# Amorphous and Twinning Perturbations Implementation Plan

> **For Trae:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Implement amorphous structure generation and twinning boundary perturbations in NepTrain.

**Architecture:** 
- Add `amorphous.py` for "Shake & Bake" generation (random displacement + repulsive optimization).
- Add `twinning.py` for reflection twin generation (rotation + reflection).
- Integrate both into `run.py` workflow and `cli.py`.

**Tech Stack:** Python, ASE, NumPy.

---

### Task 1: Amorphous Structure Module

**Files:**
- Create: `src/NepTrain/core/perturb/amorphous.py`
- Test: `test_scripts/perturb/test_amorphous.py`

**Step 1: Write failing test**
Create `test_scripts/perturb/test_amorphous.py` testing randomization and minimum distance check.

**Step 2: Implement core logic**
Implement `generate_amorphous` in `src/NepTrain/core/perturb/amorphous.py`.
- Apply large random displacements.
- Enforce minimum distance using a simple repulsive loop or ASE optimizer.
- Recommended: Simple repulsive force loop for dependency minimization.

**Step 3: Verify**
Run unit tests.

### Task 2: Twinning Boundary Module

**Files:**
- Create: `src/NepTrain/core/perturb/twinning.py`
- Test: `test_scripts/perturb/test_twinning.py`

**Step 1: Write failing test**
Create `test_scripts/perturb/test_twinning.py` testing twin generation on FCC crystal.

**Step 2: Implement core logic**
Implement `generate_twin` in `src/NepTrain/core/perturb/twinning.py`.
- Rotate crystal so twin plane is along Z.
- Reflect top half atoms.
- Handle PBC and overlap.

**Step 3: Verify**
Run unit tests.

### Task 3: CLI and Workflow Integration

**Files:**
- Modify: `src/NepTrain/cli/cli.py`
- Modify: `src/NepTrain/core/perturb/run.py`

**Step 1: Add CLI arguments**
- `--amorphous` (bool), `--amorphous-shake` (float)
- `--twin` (bool), `--twin-plane` (str: "1,1,1")

**Step 2: Integrate in run.py**
- Import new modules.
- Add to `perturb` function.
- Add to `run_perturb` argument passing.
- Ordering: Surface -> GB -> Twin -> Dislocation -> Amorphous (Amorphous likely overwrites everything else, so maybe exclusive or last step if enabled).
- Decision: If `amorphous` is True, it overrides other geometric perturbations or is applied as a final "melting" step. Let's make it a distinct mode or apply it last. Since it destroys structure, applying it last makes sense if combined, but usually it's standalone. Let's put it at the end of geometric perturbations.

### Task 4: Workflow Verification

**Files:**
- Create: `test_scripts/perturb/test_workflow_advanced.py`

**Step 1: Create workflow test**
- Test end-to-end generation for amorphous and twinning.
- Verify Config_type tags.

**Step 2: Run test**
- Ensure it passes.
