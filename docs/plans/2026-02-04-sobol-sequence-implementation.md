# Sobol Sequence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Sobol Sequence with scrambling for generating perturbation parameters (strain, displacement) to maximize coverage of the configuration superspace.

**Architecture:**
Introduce a `Sampler` abstraction. Create a `SobolSampler` wrapping `scipy.stats.qmc.Sobol`. Integrate this sampler into the perturbation workflow to replace `np.random.uniform` when requested. This ensures better distribution of parameters in high-dimensional spaces compared to pseudo-random sampling.

**Tech Stack:** Python, NumPy, SciPy (`scipy.stats.qmc`), ASE

---

### Task 1: Create Sampler Module

**Files:**
- Create: `src/NepTrain/core/perturb/sampler.py`
- Test: `test_scripts/perturb/test_sampler.py`

**Step 1: Write the failing test**

```python
import unittest
import numpy as np
from NepTrain.core.perturb.sampler import SobolSampler, RandomSampler

class TestSampler(unittest.TestCase):
    def test_random_sampler(self):
        sampler = RandomSampler(d=2)
        samples = sampler.random(n=10)
        self.assertEqual(samples.shape, (10, 2))
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples <= 1))

    def test_sobol_sampler(self):
        sampler = SobolSampler(d=2, scramble=True)
        samples = sampler.random(n=4) # Sobol usually works best with powers of 2
        self.assertEqual(samples.shape, (4, 2))
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples <= 1))
        
        # Check if it's actually Sobol (basic property check or just type check)
        # For now, just ensuring it runs and returns correct shape
        
    def test_scaling(self):
        # Test scaling from [0, 1] to [min, max]
        sampler = SobolSampler(d=1)
        samples = sampler.random(n=5)
        scaled = sampler.scale(samples, l_bounds=[-1], u_bounds=[1])
        self.assertTrue(np.all(scaled >= -1))
        self.assertTrue(np.all(scaled <= 1))
```

**Step 2: Run test to verify it fails**

Run: `uv run python test_scripts/perturb/test_sampler.py`
Expected: FAIL (ImportError)

**Step 3: Implement `Sampler` classes**

```python
import numpy as np
from scipy.stats import qmc

class BaseSampler:
    def __init__(self, d: int, seed: int = None):
        self.d = d
        self.seed = seed

    def random(self, n: int = 1):
        raise NotImplementedError

    def scale(self, sample, l_bounds, u_bounds):
        return qmc.scale(sample, l_bounds, u_bounds)

class RandomSampler(BaseSampler):
    def __init__(self, d: int, seed: int = None):
        super().__init__(d, seed)
        self.rng = np.random.default_rng(seed)

    def random(self, n: int = 1):
        return self.rng.random((n, self.d))

class SobolSampler(BaseSampler):
    def __init__(self, d: int, scramble: bool = True, seed: int = None):
        super().__init__(d, seed)
        self.engine = qmc.Sobol(d=d, scramble=scramble, seed=seed)

    def random(self, n: int = 1):
        # Sobol sequence can be exhausted or reset. 
        # For continuous generation, we just call random.
        return self.engine.random(n=n)
```

**Step 4: Run test to verify it passes**

Run: `uv run python test_scripts/perturb/test_sampler.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/NepTrain/core/perturb/sampler.py test_scripts/perturb/test_sampler.py
git commit -m "feat: implement Sobol and Random samplers"
```

---

### Task 2: Integrate Sampler into CLI

**Files:**
- Modify: `src/NepTrain/cli/cli.py`
- Test: `test_scripts/cli/test_cli_consistency.py` (Verify args parsing)

**Step 1: Write test for new args**

Create `test_scripts/cli/test_cli_sampler.py`:
```python
import unittest
import argparse
from NepTrain.cli.cli import build_perturb

class TestCLISampler(unittest.TestCase):
    def test_sampler_args(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        build_perturb(subparsers)
        
        # Test default
        args = parser.parse_args(['perturb', 'dummy.xyz'])
        self.assertEqual(args.sampler, 'random')
        
        # Test sobol
        args = parser.parse_args(['perturb', 'dummy.xyz', '--sampler', 'sobol'])
        self.assertEqual(args.sampler, 'sobol')
        
        # Test scramble
        args = parser.parse_args(['perturb', 'dummy.xyz', '--no-scramble'])
        self.assertFalse(args.scramble)
```

**Step 2: Run test**

Run: `uv run python test_scripts/cli/test_cli_sampler.py`
Expected: FAIL (unrecognized arguments)

**Step 3: Modify CLI**

In `src/NepTrain/cli/cli.py`:
```python
# Add to build_perturb function:
    parser_perturb.add_argument("--sampler",
                                dest="sampler",
                                type=str,
                                default="random",
                                choices=["random", "sobol"],
                                help="Sampling method for perturbation parameters (default: random).")

    parser_perturb.add_argument("--no-scramble",
                                dest="scramble",
                                action='store_false',
                                default=True,
                                help="Disable scrambling for Sobol sequence (default: scrambling enabled).")
```

**Step 4: Run test**

Run: `uv run python test_scripts/cli/test_cli_sampler.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/NepTrain/cli/cli.py test_scripts/cli/test_cli_sampler.py
git commit -m "feat: add CLI arguments for sampling method"
```

---

### Task 3: Integrate Sampler into Perturb Workflow

**Files:**
- Modify: `src/NepTrain/core/perturb/run.py`
- Test: `test_scripts/perturb/test_workflow_sampler.py`

**Step 1: Write verification test**

```python
import unittest
import os
import shutil
from ase.build import bulk
from ase.io import write
from NepTrain.core.perturb.run import perturb
import numpy as np

class TestWorkflowSampler(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'test_sampler_workflow_output'
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        self.atoms = bulk('Cu', 'fcc', a=3.6)
        self.test_file = os.path.join(self.test_dir, 'input.xyz')
        write(self.test_file, self.atoms)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_sobol_execution(self):
        # Run with sobol
        # We can't easily verify statistical properties with 1 run,
        # but we can verify it runs without error and produces valid structures.
        results = perturb(
            self.test_file,
            num=4,
            sampler='sobol',
            scramble=True,
            cell_pert_fraction=0.05,
            min_distance=0.1
        )
        self.assertEqual(len(results), 1)
        structures = results[0]
        self.assertEqual(len(structures), 4)
        
        # Check config type
        self.assertIn('sobol', structures[0].info['Config_type'])
```

**Step 2: Run test**

Run: `uv run python test_scripts/perturb/test_workflow_sampler.py`
Expected: FAIL (Arguments not handled, functionality not implemented)

**Step 3: Modify `run.py`**

1. Import `SobolSampler`, `RandomSampler`.
2. Initialize sampler based on argument.
3. Use sampler to generate parameters for `generate_strained_structure` and `generate_deformed_structure`.
   - **Note:** The current `generate_strained_structure` uses `np.random.uniform` internally. We need to refactor this to accept external parameters or the sampler itself.
   - **Refactoring Strategy:** Modify `generate_strained_structure` and `generate_deformed_structure` to accept `strain_values` (array) or similar, OR pass the sampler. Passing specific values is cleaner.

**Refactoring Plan in `run.py`:**
- In `perturb` function loop:
  - Generate a batch of samples for all iterations upfront? Sobol works best in batches (power of 2).
  - Or generate on the fly. `scipy.stats.qmc.Sobol` is stateful.
  - Dimension `d`: 
    - 3 for `strained` (diagonal strains)
    - 9 for `deformed` (full tensor) -> actually usually 6 (symmetric) or 9 (general). `generate_deformed_structure` uses 3x3 uniform. So `d=9`.
    - `perturb_position` uses `size=positions.shape`. That's `3*N_atoms`. This is too high dim for Sobol to be effective (usually < 20-30 dims).
    - **Decision:** Use Sobol for *global* parameters (cell strain/deformation) and keep Random for *atomic* displacement (rattle), OR use a separate low-dim Sobol for rattle magnitude/direction? Rattle is high-dimensional. Let's stick to Random for rattle for now, and Sobol for Cell parameters.

- `perturb` function:
  - Initialize `strain_sampler` (d=3) and `deformation_sampler` (d=9).
  - Generate `num` samples.
  - Scale them to `[-cell_pert_fraction, cell_pert_fraction]`.
  - Pass these specific `strain` or `deformation` tensors to the generation functions.

- Update `generate_strained_structure(..., strain_tensor=None)`
- Update `generate_deformed_structure(..., deformation_tensor=None)`

**Step 4: Run test**

Run: `uv run python test_scripts/perturb/test_workflow_sampler.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/NepTrain/core/perturb/run.py test_scripts/perturb/test_workflow_sampler.py
git commit -m "feat: integrate Sobol sampler into perturb workflow"
```

---
