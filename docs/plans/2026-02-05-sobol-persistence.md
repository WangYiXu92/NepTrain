# Sobol Sequence State Persistence Implementation Plan

> **For Trae:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Implement a mechanism to save and load the state of the Sobol sequence generator, allowing users to resume or extend perturbation runs without overlapping or repeating configurations.

**Architecture:**
- Create a JSON state file (e.g., `perturb_state.json`) alongside the output file.
- Save `num_generated`, `seed`, and `total_d` (total dimensions) after each run.
- Add a `resume` parameter to `run_perturb` and `perturb` to load this state.
- When resuming, validate `total_d` matches the current configuration.
- Fast-forward the Sobol engine by `num_generated` steps.

**Tech Stack:** Python, JSON, SciPy (Sobol)

---

### Task 1: Create State Management Logic

**Files:**
- Modify: `src/NepTrain/core/perturb/run.py`
- Test: `tests/test_sobol_persistence.py`

**Step 1: Write the failing test**

```python
import unittest
import os
import json
import numpy as np
from ase import Atoms
from NepTrain.core.perturb.run import perturb

class TestSobolPersistence(unittest.TestCase):
    def setUp(self):
        self.atoms = Atoms('Si2', positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 5, 5], pbc=True)
        self.state_file = 'test_state.json'
        if os.path.exists(self.state_file):
            os.remove(self.state_file)

    def tearDown(self):
        if os.path.exists(self.state_file):
            os.remove(self.state_file)

    def test_save_state(self):
        # Generate 5 structures and save state
        gen = perturb(self.atoms, num=5, sampler='sobol', state_file=self.state_file)
        list(gen)
        
        self.assertTrue(os.path.exists(self.state_file))
        with open(self.state_file, 'r') as f:
            state = json.load(f)
        self.assertEqual(state['num_generated'], 5)
        self.assertIn('seed', state)
        self.assertIn('total_d', state)

    def test_resume_state(self):
        # 1. Generate first 5
        gen1 = perturb(self.atoms, num=5, sampler='sobol', state_file=self.state_file)
        list(gen1)
        
        # 2. Generate next 5 (resuming)
        # Note: In practice, resuming means "start from N and generate M more".
        # But our perturb(num=X) usually means "generate X structures total" or "X more"?
        # Standard interpretation: num is "number to generate in THIS run".
        # So if we resume, we fast-forward by saved state, then generate num.
        gen2 = perturb(self.atoms, num=5, sampler='sobol', state_file=self.state_file, resume=True)
        structs2 = list(gen2)
        self.assertEqual(len(structs2), 5)
        
        # 3. Verify total generated is 10 in state file
        with open(self.state_file, 'r') as f:
            state = json.load(f)
        self.assertEqual(state['num_generated'], 10)

    def test_resume_consistency(self):
        # Verify that splitting 10 into 5+5 yields same as 10 at once
        # This requires fixing the seed
        seed = 42
        
        # Run A: 10 at once
        gen_a = perturb(self.atoms, num=10, sampler='sobol', seed=seed)
        structs_a = list(gen_a)
        
        # Run B: 5 then 5
        state_file_b = 'test_state_b.json'
        gen_b1 = perturb(self.atoms, num=5, sampler='sobol', seed=seed, state_file=state_file_b)
        structs_b1 = list(gen_b1)
        
        gen_b2 = perturb(self.atoms, num=5, sampler='sobol', seed=seed, state_file=state_file_b, resume=True)
        structs_b2 = list(gen_b2)
        
        # Check positions of 6th structure (index 0 of second batch vs index 5 of first batch)
        # Note: Floating point exact match might be tricky, use almost equal
        pos_a = structs_a[5].get_positions()
        pos_b = structs_b2[0].get_positions()
        np.testing.assert_allclose(pos_a, pos_b, atol=1e-6)
        
        if os.path.exists(state_file_b):
            os.remove(state_file_b)

```

**Step 2: Run test to verify it fails**

Run: `uv run python tests/test_sobol_persistence.py`
Expected: FAIL with "unexpected keyword argument 'state_file'"

**Step 3: Implement `state_file` and `resume` logic in `run.py`**

- Update `perturb` signature to accept `state_file=None` and `resume=False`.
- Implement `_load_state` and `_save_state` helper functions (or inline logic).
- In `perturb`:
    - If `resume` and `state_file` exists: load `num_generated` (skip), `seed`.
    - Initialize `SobolSampler` with loaded seed (or provided seed if new).
    - If resuming, verify `total_d` matches.
    - If resuming, call `sampler.fast_forward(num_generated)`.
    - Generate `num` samples.
    - Update state: `new_total = num_generated + num`.
    - Write state to `state_file`.

**Step 4: Run test to verify it passes**

Run: `uv run python tests/test_sobol_persistence.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/NepTrain/core/perturb/run.py tests/test_sobol_persistence.py
git commit -m "feat: add Sobol sequence state persistence and resume capability"
```

---

### Task 2: Update CLI Wrapper

**Files:**
- Modify: `src/NepTrain/cli/cli.py`

**Step 1: Update `perturb_cli_wrapper`**

- Map CLI arguments to `state_file` and `resume`.
- We can auto-generate `state_file` name from `output_file` if not provided (e.g., `output.xyz` -> `output.state.json`).
- Or just add explicit arguments. Let's add explicit `--state-file` and `--resume` flag.

**Step 2: Add arguments to `parser_perturb`**

```python
parser_perturb.add_argument("--state-file", dest="state_file", type=str, default=None, help="JSON file to save/load Sobol state.")
parser_perturb.add_argument("--resume", dest="resume", action="store_true", help="Resume Sobol sequence from state file.")
```

**Step 3: Update wrapper to pass these args**

In `perturb_cli_wrapper(args)`:
```python
if 'state_file' in kwargs:
    # Optional: if state_file is None but sampler is Sobol, maybe default to output_file + ".state.json"?
    # For now, let's keep it explicit as per plan.
    pass
```

**Step 4: Manual Verification**

Run a CLI command:
`neptrain perturb --sampler sobol --num 5 --out result.xyz --state-file state.json`
Then:
`neptrain perturb --sampler sobol --num 5 --out result2.xyz --state-file state.json --resume`

**Step 5: Commit**

```bash
git add src/NepTrain/cli/cli.py
git commit -m "feat: add CLI support for Sobol state persistence"
```

---
## Status Update (2026-02-05)
- [x] Task 1: Create State Management Logic
- [x] Task 2: Update CLI Wrapper
- [x] Validation Tests Passed
