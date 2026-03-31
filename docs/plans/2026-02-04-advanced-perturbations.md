# Advanced Perturbations Implementation Plan

> **For Trae:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.

**Goal:** Implement advanced perturbation types: Surfaces, Grain Boundaries, and Dislocations, starting with Surfaces.

**Architecture:** 
- Add new modules in `src/NepTrain/core/perturb/` for each defect type.
- Integrate into `run.py` workflow, ensuring they run before normal perturbations.
- Update CLI to accept parameters for these new perturbations.

**Tech Stack:** Python, ASE (Atomic Simulation Environment)

---

## Phase 1: Surface Perturbation

### Task 1.1: Create Surface Perturbation Module

**Files:**
- Create: `src/NepTrain/core/perturb/surface.py`
- Test: `test_scripts/perturb/test_surface.py`

**Step 1: Write the failing test**

```python
import unittest
from ase.build import bulk
from NepTrain.core.perturb.surface import generate_surface

class TestSurface(unittest.TestCase):
    def test_generate_surface(self):
        atoms = bulk('Cu', 'fcc', a=3.6)
        # Generate (1,1,1) surface with 10A vacuum and 3 layers
        slab = generate_surface(atoms, indices=(1,1,1), vacuum=10.0, layers=3)
        
        self.assertTrue(slab.pbc[2] == False)  # Z direction usually non-periodic for slab
        self.assertTrue(len(slab) >= 3)
        self.assertTrue(slab.cell[2,2] > 20) # Vacuum included
```

**Step 2: Run test to verify it fails**
`python -m unittest test_scripts/perturb/test_surface.py`

**Step 3: Write implementation**

```python
from ase.build import surface as ase_surface
from ase.atoms import Atoms

def generate_surface(atoms: Atoms, indices: tuple = (1, 1, 1), vacuum: float = 10.0, layers: int = 3) -> Atoms:
    """
    Generate a surface slab from a bulk structure.
    
    Args:
        atoms: Input bulk Atoms object
        indices: Miller indices (h, k, l)
        vacuum: Vacuum size in Angstroms on both sides (total vacuum = 2 * vacuum if using ase.build.surface default behavior, 
                but usually we want total vacuum. ase.build.surface takes 'vacuum' as size on top/bottom? 
                Let's check ase docs behavior: 'vacuum' is added on both sides.
        layers: Number of atomic layers
        
    Returns:
        Atoms object representing the slab
    """
    # ase.build.surface returns a slab with vacuum
    # Note: input atoms should be bulk standard cell ideally
    slab = ase_surface(atoms, indices, layers)
    slab.center(vacuum=vacuum, axis=2)
    return slab
```

**Step 4: Run test to verify it passes**

**Step 5: Commit**

### Task 1.2: Integrate Surface into `run.py`

**Files:**
- Modify: `src/NepTrain/core/perturb/run.py`
- Modify: `src/NepTrain/cli/cli.py`
- Test: `test_scripts/perturb/test_workflow_surface.py`

**Step 1: Write the failing test**
Create a test that runs `perturb` with surface arguments and checks the output.

**Step 2: Update CLI arguments**
Add `--surface`, `--miller-indices`, `--vacuum`, `--layers` to `cli.py`.

**Step 3: Update `run.py`**
Import `generate_surface`.
Add logic to call `generate_surface` if enabled, *before* normal perturbation.
Note: Surface generation changes the number of atoms and cell, so it must be done early.

**Step 4: Verify**

---

## Phase 2: Grain Boundary (To Be Detailed Later)
...

## Phase 3: Dislocation (To Be Detailed Later)
...
