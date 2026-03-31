# Dislocation Perturbation Implementation Plan

> **For Trae:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Implement dislocation generation (edge and screw) as a perturbation type in NepTrain.

**Architecture:** 
- Add `dislocation.py` module with `generate_dislocation` function applying Volterra displacement fields.
- Integrate into `run.py` workflow (after surface/GB, before other perturbations).
- Add CLI arguments for dislocation parameters.

**Tech Stack:** Python, ASE, NumPy.

---

### Task 1: Core Dislocation Module

**Files:**
- Create: `src/NepTrain/core/perturb/dislocation.py`
- Test: `test_scripts/perturb/test_dislocation.py`

**Step 1: Write failing test**
Create `test_scripts/perturb/test_dislocation.py` testing edge and screw generation.

**Step 2: Implement core logic**
Implement `generate_dislocation` in `src/NepTrain/core/perturb/dislocation.py`.
- Support 'edge' and 'screw' types.
- Apply displacement field based on Burgers vector and position.
- Handle PBC (optional/warning or simple displacement).

**Step 3: Verify**
Run unit tests.

### Task 2: CLI and Workflow Integration

**Files:**
- Modify: `src/NepTrain/cli/cli.py`
- Modify: `src/NepTrain/core/perturb/run.py`

**Step 1: Add CLI arguments**
- `--dislocation` (bool)
- `--dislocation-type` (str: edge/screw)
- `--dislocation-axis` (int: 0,1,2 - direction of line)
- `--dislocation-burgers` (float: magnitude)

**Step 2: Integrate in run.py**
- Import `generate_dislocation`.
- Add to `perturb` function.
- Add to `run_perturb` argument passing.
- Execute in loop (ordering: Surface -> GB -> Dislocation -> others).

### Task 3: Workflow Verification

**Files:**
- Create: `test_scripts/perturb/test_workflow_dislocation.py`

**Step 1: Create workflow test**
- Test end-to-end generation via `perturb` function.
- Verify Config_type and atomic displacements.

**Step 2: Run test**
- Ensure it passes.
