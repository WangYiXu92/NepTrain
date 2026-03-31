# Sobol Expansion for Shuffle and Vacancy Perturbations

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Extend Sobol sequence sampling to `shuffle` and `vacancy` perturbations to ensure better coverage of chemical ordering and defect distribution.

**Architecture:** 
- Update `shuffle.py` and `vacancy.py` to accept `rng_values` for deterministic behavior.
- Update `run.py` to calculate required dimensions ($D_{vac}$ and $D_{shuf}$) and pass Sobol slices.
- Use "sort-based selection/permutation" (assigning a random value to each item and sorting) as the mapping strategy for Sobol values.

**Tech Stack:** Python, NumPy, ASE

---

### Task 1: Refactor `vacancy.py`

**Files:**
- Modify: `src/NepTrain/core/perturb/vacancy.py`
- Test: `test_scripts/perturb/test_vacancy_sobol.py` (New)

**Step 1: Write the failing test**

```python
import unittest
import numpy as np
from ase import Atoms
from NepTrain.core.perturb.vacancy import generate_vacancies

class TestVacancySobol(unittest.TestCase):
    def test_deterministic_vacancy(self):
        atoms = Atoms('Fe10')
        # We want to remove 3 atoms. Candidates are all 10.
        # If we provide rng_values, the selection should be deterministic based on sorting.
        # rng_values: assign low values to specific indices to force their selection.
        # e.g. indices 0, 5, 9 get low values.
        
        rng = np.ones(10) # High values
        rng[0] = 0.1
        rng[5] = 0.2
        rng[9] = 0.3
        
        # We expect indices 0, 5, 9 to be selected for vacancy (marked 'X')
        new_atoms, _ = generate_vacancies(atoms, ['Fe'], num_vacancies=3, mode='random', rng_values=rng)
        
        symbols = new_atoms.get_chemical_symbols()
        self.assertEqual(symbols[0], 'X')
        self.assertEqual(symbols[5], 'X')
        self.assertEqual(symbols[9], 'X')
        self.assertEqual(symbols.count('X'), 3)
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest test_scripts/perturb/test_vacancy_sobol.py`
Expected: `TypeError` (unexpected argument `rng_values`)

**Step 3: Write minimal implementation**

Modify `generate_vacancies` in `src/NepTrain/core/perturb/vacancy.py`:
- Add `rng_values: np.ndarray = None` argument.
- Implement logic:
    ```python
    if rng_values is not None:
        if len(rng_values) < len(candidate_indices):
            raise ValueError(f"Not enough random values. Needed {len(candidate_indices)}, got {len(rng_values)}")
        
        # Use first N values corresponding to candidates
        # We need to decide which ones to pick.
        # Let's say we pick the ones with SMALLEST random values (or largest).
        # It's a random selection without replacement.
        
        # Sort candidates by their random value
        # We only need values for the candidates.
        current_rng = rng_values[:len(candidate_indices)]
        
        # Get indices that would sort the array
        sorted_args = np.argsort(current_rng)
        
        # Pick top K
        picked_args = sorted_args[:num_vacancies]
        
        # Map back to original candidate indices
        selected_indices = [candidate_indices[i] for i in picked_args]
        selected_indices = np.array(selected_indices)
    else:
        # Existing random choice
        selected_indices = np.random.choice(candidate_indices, num_vacancies, replace=False)
    ```

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest test_scripts/perturb/test_vacancy_sobol.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/NepTrain/core/perturb/vacancy.py test_scripts/perturb/test_vacancy_sobol.py
git commit -m "feat: add Sobol support to vacancy generation"
```

---

### Task 2: Refactor `shuffle.py`

**Files:**
- Modify: `src/NepTrain/core/perturb/shuffle.py`
- Test: `test_scripts/perturb/test_shuffle_sobol.py` (New)

**Step 1: Write the failing test**

```python
import unittest
import numpy as np
from ase import Atoms
from NepTrain.core.perturb.shuffle import shuffle_element_positions

class TestShuffleSobol(unittest.TestCase):
    def test_deterministic_shuffle(self):
        atoms = Atoms('ABCDE')
        # Shuffle all
        # rng_values: define a specific permutation via argsort
        # Values: [0.5, 0.1, 0.9, 0.2, 0.4]
        # Sorted indices: 1 (0.1), 3 (0.2), 4 (0.4), 0 (0.5), 2 (0.9)
        # Expected order: B, D, E, A, C
        
        rng = np.array([0.5, 0.1, 0.9, 0.2, 0.4])
        
        new_atoms, _ = shuffle_element_positions(atoms, element_range='0:5', rng_values=rng)
        
        self.assertEqual("".join(new_atoms.get_chemical_symbols()), "BDEAC")
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest test_scripts/perturb/test_shuffle_sobol.py`
Expected: `TypeError` (unexpected argument `rng_values`)

**Step 3: Write minimal implementation**

Modify `shuffle_element_positions` in `src/NepTrain/core/perturb/shuffle.py`:
- Add `rng_values: np.ndarray = None` argument.
- Implement logic:
    ```python
    if rng_values is not None:
        if len(rng_values) < len(indices):
             raise ValueError(...)
        
        # Permutation via argsort
        perm = np.argsort(rng_values[:len(indices)])
        new_positions_subset = selected_positions[perm]
    else:
        # Existing logic
    ```

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest test_scripts/perturb/test_shuffle_sobol.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/NepTrain/core/perturb/shuffle.py test_scripts/perturb/test_shuffle_sobol.py
git commit -m "feat: add Sobol support to shuffle"
```

---

### Task 3: Update `run.py`

**Files:**
- Modify: `src/NepTrain/core/perturb/run.py`
- Test: `test_scripts/perturb/test_sobol_full_integration.py` (New)

**Step 1: Write the failing test**

```python
import unittest
from NepTrain.core.perturb.run import perturb

class TestSobolFull(unittest.TestCase):
    def test_full_chain(self):
        # Run perturb with vacancy and shuffle using Sobol
        # Verify it runs without error and returns results
        # Hard to verify exact determinism without mocking, but we check execution flow.
        
        # Create a dummy file
        with open("test_full_sobol.xyz", "w") as f:
            f.write("4\n\nFe 0 0 0\nFe 1 0 0\nNi 0 1 0\nNi 1 1 0\n")
            
        gen = perturb("test_full_sobol.xyz", 
                      num=2, 
                      sampler='sobol',
                      vac_elements='Fe', vac_num=1,
                      shuffle_elements='Ni',
                      skip_normal=True) # Skip normal to isolate logic
                      
        results = list(gen)
        # Should succeed
        self.assertTrue(len(results) > 0)
        
        # Cleanup
        import os
        os.remove("test_full_sobol.xyz")
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest test_scripts/perturb/test_sobol_full_integration.py`
Expected: Likely fail or ignore Sobol for new features (since logic isn't connected yet). Or fail if we added args but didn't pass them. Actually it will run but use random for vac/shuffle, which is not what we want to verify. We want to verify `rng_values` are passed.
To verify strictly, we might need to mock or inspect internals, but for now, ensuring it runs and doesn't crash with "not enough random values" or similar is a good first step.
The real failure will be if we *don't* implement it, the Sobol sampler won't account for dimensions, so if we try to use it later it might desynchronize? No, without implementation, it just uses Random.
So this test is more of an Integration Test.
To make it "fail" in a meaningful way regarding features, we rely on the unit tests.
This step mainly ensures `run.py` is updated to *connect* the pieces.

**Step 3: Write implementation**

Modify `src/NepTrain/core/perturb/run.py`:
- Update dimension calculation block (Sobol initialization):
    - Calculate `d_vac` using `generate_vacancies` logic (identifying candidates on `dummy_atoms`).
    - Calculate `d_shuf` using `shuffle_element_positions` helper logic (parsing range on `dummy_atoms`).
    - `total_d += d_vac + d_shuf`
- Update sampling loop:
    - Slice `raw_samples` to get `sobol_vac_samples` and `sobol_shuf_samples`.
    - Pass `rng_values` to `generate_vacancies` and `shuffle_element_positions`.

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest test_scripts/perturb/test_sobol_full_integration.py`

**Step 5: Commit**

```bash
git add src/NepTrain/core/perturb/run.py test_scripts/perturb/test_sobol_full_integration.py
git commit -m "feat: integrate Sobol for vacancy and shuffle in run.py"
```
