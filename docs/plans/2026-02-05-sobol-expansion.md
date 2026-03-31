# Sobol Sequence Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Sobol sequence sampling for `magnetic` and `rotate` perturbations to improve configuration space coverage.

**Architecture:** 
Extend the existing Sobol sampler in `run.py` to cover magnetic and rotation parameters. Refactor `magnetic.py` and `rotate.py` to accept pre-generated uniform random numbers (from Sobol) and transform them into the required distributions (Normal, Rotation Matrix). This ensures a unified low-discrepancy sequence across all perturbation dimensions.

**Tech Stack:** Python, NumPy, SciPy (stats.qmc, spatial.transform), ASE.

---

### Task 1: Refactor Magnetic Perturbation

**Files:**
- Modify: `src/NepTrain/core/perturb/magnetic.py`
- Test: `test_scripts/perturb/test_magnetic.py`

**Step 1: Create reproduction script/test**
Create a test that verifies `apply_magnetic_perturbation` works with default random behavior, and then test it with injected random values to ensure determinism.

**Step 2: Modify `apply_magnetic_perturbation` signature**
Add `rng_values=None` argument.

**Step 3: Implement Uniform to Normal/Boolean transformation**
Inside `apply_magnetic_perturbation`:
- If `rng_values` is provided:
  - For `noise` and `non_collinear`: Use `scipy.stats.norm.ppf` to convert uniform to normal.
  - For `random_collinear`: Use uniform values directly for probability check.
  - Ensure correct slicing of `rng_values` based on needed dimensions.

**Step 4: Update Tests**
Verify that passing the same `rng_values` produces identical magnetic configurations.

---

### Task 2: Refactor Rotate Perturbation

**Files:**
- Modify: `src/NepTrain/core/perturb/rotate.py`
- Test: `test_scripts/perturb/test_rotate.py`

**Step 1: Create reproduction script/test**
Verify existing `rotate_fragments_by_formula`.

**Step 2: Implement Uniform to Rotation Logic**
Add helper function `uniform_to_rotation_matrix(u1, u2, u3)` using Shoemake's algorithm (Unit Quaternion generation).
- $q = [\sqrt{1-u_1} \sin(2\pi u_2), \sqrt{1-u_1} \cos(2\pi u_2), \sqrt{u_1} \sin(2\pi u_3), \sqrt{u_1} \cos(2\pi u_3)]$
- Convert $q$ to rotation matrix.

**Step 3: Modify `rotate_fragments_by_formula`**
Add `rng_values=None` argument.
If provided, consume 3 values per fragment to generate rotations.

**Step 4: Update Tests**
Verify determinism with `rng_values`.

---

### Task 3: Integrate into Run Workflow

**Files:**
- Modify: `src/NepTrain/core/perturb/run.py`
- Test: `test_scripts/perturb/test_workflow_sampler.py` (Create new if needed)

**Step 1: Dimension Calculation Logic**
In `perturb` function (before the loop):
- Determine `n_mag_dims`:
  - `collinear` + noise: `n_mag_atoms` (if noise > 0)
  - `random_collinear`: `n_atoms`
  - `non_collinear`: `3 * n_mag_atoms`
- Determine `n_rot_dims`:
  - Call `get_molecules` to find `n_frags`.
  - Dim = `3 * n_frags`.
- Total Dim `d = 9 + n_mag_dims + n_rot_dims`.

**Step 2: Generate Sobol Sequence**
- Initialize `SobolSampler(d=d)`.
- Generate `num` samples.

**Step 3: Pass Samples to Functions**
- In the loop, slice the sample vector.
- Pass appropriate slices to `apply_magnetic_perturbation` and `rotate_fragments_by_formula`.

**Step 4: Verify Integration**
Run a full perturbation workflow with `sampler='sobol'` and `mag_mode`, `rotate_formula` enabled. Check if it runs without errors and produces valid structures.

---

### Task 4: Verification and Cleanup

**Files:**
- Test: `test_scripts/perturb/test_sobol_full.py`

**Step 1: Create Comprehensive Test**
Run `NepTrain perturb` command with all features enabled and Sobol sampler.
Verify output `nep.txt` or `xyz` files.

**Step 2: Documentation**
Update docstrings to reflect new `rng_values` parameters.
