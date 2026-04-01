# perturb
**Description:**  
Generates perturbed structures for training datasets, supporting various defects, topological changes, and sampling methods.

## Input Parameters

**Usage:**  
```bash
NepTrain perturb <model_path> [options]
```

**Options:**  
- `<model_path>`  
  The structure path or file for calculation (supports `xyz`, `vasp`, `cif`, `POSCAR` formats).
- `-n, --num`  
  Number of perturbations for each structure. Default: `20`.
- `-c, --cell`  
  Deformation ratio (max strain). Default: `0.03`.
- `-d, --distance`
  Minimum atom distance (Å) for random perturbation. Default: `0.1`.
- `-o, --out`
  Output file path for perturbed structures. Default: `./perturb.xyz`.
- `-a, --append`
  Append to output file instead of overwriting. Default: `False`.
- `--skip-normal`
  Skip standard cell deformation and atomic rattling. Useful when applying only specific topological defects. Default: `False`.

**Sampling & Persistence:**
- `--sampler`
  Sampling method: `random` (default) or `sobol` (Quasi-Monte Carlo).
- `--no-scramble`
  Disable scrambling for Sobol sequence.
- `--state-file`
  JSON file to save/load Sobol sequence state for resuming runs.
- `--resume`
  Resume Sobol sequence from state file.
- `--seed`
  Random seed for reproducibility.

**Defects & Chemistry:**
- `--vac-elements`
  Elements to consider for vacancies (comma-separated, e.g. 'Fe,O').
- `--vac-num`
  Number of vacancies to generate per structure. Default: `0`.
- `--shuffle-elements`
  Elements or indices to shuffle (e.g. 'Fe,O' or '0:10').
- `--shuffle-method`
  Shuffling method: `fisher_yates` (default) or `random_swap`.
- `--rotate-formula`
  Chemical formula of fragments to rotate randomly (e.g. 'H2O').

**Magnetism:**
- `--mag-mode`
  Mode: `collinear`, `random_collinear`, `non_collinear`.
- `--mag-flip-prob`
  Spin flip probability for `random_collinear`. Default: `0.5`.
- `--mag-noise`
  Gaussian noise for magnetic moment magnitudes. Default: `0.0`.

**Topological Generation:**
- `--surface`
  Generate surface slabs.
- `--indices`
  Miller indices for surface (e.g. '1,1,1'). Default: `1,1,1`.
- `--vacuum`
  Vacuum size (Å). Default: `10.0`.
- `--layers`
  Number of atomic layers. Default: `3`.
- `--gb`
  Generate grain boundary.
- `--gb-axis`
  Rotation axis (e.g. '0,0,1'). Default: `0,0,1`.
- `--gb-angle`
  Rotation angle (degrees). Default: `30.0`.
- `--gb-dist`
  Overlap removal distance for GB. Default: `2.0`.
- `--no-gb-overlap`
  Disable overlap removal for grain boundary.
- `--dislocation`
  Generate dislocation.
- `--dislocation-type`
  Type: `edge` (default) or `screw`.
- `--dislocation-axis`
  Line axis vector. Default: `0,0,1`.
- `--dislocation-burgers`
  Burgers vector. Default: `1,0,0`.
- `--twinning`
  Generate twin boundary.
- `--twinning-indices`
  Miller indices for twin plane. Default: `1,1,1`.
- `--twinning-z`
  Fractional height for boundary. Default: `0.5`.
- `--twinning-min-dist`
  Overlap removal distance for twinning. Default: `1.5`.
- `--stacking-fault`
  Generate stacking fault.
- `--sf-indices`
  Miller indices for stacking fault plane. Default: `1,1,1`.
- `--sf-shift`
  Shift vector for stacking fault or `random` (default `random`).
- `--sf-height`
  Fractional height for stacking fault plane or `random` (default `random`).
- `--sf-min-dist`
  Minimum distance for overlap removal in stacking fault. Default: `1.5`.
- `--amorphous`
  Generate amorphous structure.
- `--amorphous-min-dist`
  Minimum distance for amorphous generation. Default: `1.5`.
- `--amorphous-rattle`
  Rattle strength. Default: `0.5`.
- `--amorphous-steps`
  Relaxation steps. Default: `100`.

**Antisite Defects:**
- `--antisite`
  Generate antisite (substitutional) defects.
- `--antisite-pairs`
  Element pairs to swap (comma-separated, e.g. 'Fe,Al' or 'Fe,Al;Cr,Fe').
- `--antisite-num`
  Number of antisite swaps per structure. Default: `1`.
- `--antisite-mode`
  Antisite generation mode: `symmetry_aware` (default) or `random`.
- `--antisite-symprec`
  Symmetry precision for spglib analysis. Default: `1e-2`.

**Symmetry-Preserving Strain:**
- `--sym-strain`
  Apply crystal-symmetry-preserving strain instead of generic cell perturbation.
- `--sym-strain-fraction`
  Maximum strain magnitude. Default: `0.03`.
- `--sym-strain-crystal-system`
  Override crystal system detection (e.g. 'cubic', 'tetragonal'). Default: auto-detect.
- `--sym-strain-symprec`
  Symmetry precision for crystal system detection. Default: `1e-2`.

**Volume Perturbation:**
- `--vol-pert-fraction`
  Volume perturbation magnitude (fraction). Default: `0.0` (disabled).

**Rigid Body:**
- `--rigid`
  Enable rigid body perturbation.
- `--rigid-method`
  Rigid body method: `auto` (default), `manual`.
- `--rigid-list`
  Manual atom group specification.
- `--rigid-mode`
  Mode: `inter` (default) or `intra`.
- `--rigid-composition`
  Composition-based grouping.

**Cell Rotation:**
- `--rotate-cell`
  Randomly rotate the unit cell.

**Filtering & Validation:**
- `--filter-bonds`
  Filter structures by bond length criteria.
- `--validate-structure`
  Validate generated structures. Default: `True`.
- `--validate-coefficient`
  Validation coefficient threshold.
- `--similarity-threshold`
  Maximum similarity between structures. Default: `0.999`.
- `--debug-plot`
  Generate debug plots. Default: `False`.

## Advanced Features

### 1. Magnetic Perturbation
Create diverse training datasets for magnetic materials.
- **collinear**: Sets initial magnetic moments based on configuration.
- **random_collinear**: Randomly flips spins (up/down).
- **non_collinear**: Generates random 3D magnetic moment vectors.

### 2. Fragment Rotation
Rotate rigid molecules or clusters (e.g., water in hydrates, octahedra) while preserving internal geometry.
```bash
NepTrain perturb water_box.xyz --rotate-formula H2O
```

### 3. Point Defects & Disorders
- **Vacancies**: Randomly remove atoms to simulate non-stoichiometry.
  ```bash
  NepTrain perturb bulk.vasp --vac-elements Fe,O --vac-num 2
  ```
- **Antisite Defects**: Swap atoms between symmetry-equivalent Wyckoff sites.
  ```bash
  NepTrain perturb fe3al.vasp --antisite --antisite-pairs Fe,Al --antisite-num 1
  ```
  Supports multiple pairs: `--antisite-pairs 'Fe,Al;Cr,Fe'`
  Use `--antisite-mode random` for random selection (no spglib required).
- **Atomic Shuffling**: Swap positions of specific species to simulate anti-site defects or solid solutions.
  ```bash
  NepTrain perturb alloy.vasp --shuffle-elements Fe,Ni
  ```
- **Amorphous**: Generate disordered/glassy structures via heavy rattling and soft-potential relaxation.
  ```bash
  NepTrain perturb glass.vasp --amorphous --amorphous-steps 200
  ```

### 4. Surfaces and Interfaces
- **Surfaces**: Create slab models with specified Miller indices and vacuum.
  ```bash
  NepTrain perturb bulk.vasp --surface --indices 1,1,1 --vacuum 15.0
  ```
- **Grain Boundaries**: Generate twist/tilt boundaries by rotating half the crystal.
  ```bash
  NepTrain perturb supercell.vasp --gb --gb-axis 0,0,1 --gb-angle 45
  ```
- **Twin Boundaries**: Create mirror-symmetric interfaces.
  ```bash
  NepTrain perturb bulk.vasp --twinning --twinning-indices 1,1,1
  ```

### 5. Dislocations
Introduce line defects into the crystal structure using Volterra displacement fields.
```bash
NepTrain perturb bulk.vasp --dislocation --dislocation-type edge --dislocation-burgers 2.5,0,0
```

### 6. Quasi-Random Sampling (Sobol)
Use Sobol sequences for efficient coverage of high-dimensional parameter spaces (strain, displacement, magnetic moments, etc.).
- **State Persistence**: Save and resume generation to avoid duplicates when extending datasets.
  ```bash
  # Initial run
  NepTrain perturb structure.vasp --sampler sobol -n 100 --state-file state.json
  # Resume later
  NepTrain perturb structure.vasp --sampler sobol -n 50 --state-file state.json --resume
  ```

### 7. Symmetry-Preserving Strain
Apply strain that respects crystal symmetry, reducing Sobol dimensions significantly.
- **Cubic**: 1 independent strain (uniform hydrostatic)
- **Tetragonal/Hexagonal/Trigonal**: 2 independent strains (in-plane + out-of-plane)
- **Orthorhombic**: 3 independent strains
- **Monoclinic**: 4 independent strains
- **Triclinic**: 6 independent strains (equivalent to generic cell perturbation)
```bash
NepTrain perturb fe3al.vasp --sym-strain --sym-strain-fraction 0.03
# Override crystal system (skip spglib detection)
NepTrain perturb structure.vasp --sym-strain --sym-strain-crystal-system cubic
```
**Note**: `--sym-strain` is mutually exclusive with `--cell` (generic cell perturbation). When enabled, generic cell deformation is automatically disabled.

### 8. Compatibility Checks
The perturb module automatically validates parameter combinations at entry:
- **Mutual exclusion**: `sym_strain` ⊥ `cell_pert`, `amorphous` ⊥ `dislocation/twinning/gb/stacking_fault/surface`
- **Dependencies**: `vacancy` requires `vac_elements` and `vac_num > 0`; `antisite` requires `antisite_pairs`
- **Warnings**: `amorphous + shuffle`, `amorphous + rigid` (may not add meaningful diversity)
- Conflicts raise `ValueError` before any computation begins.

## Python API Reference

### Core Generation

#### `perturb`
Main generator function integrating all perturbation modes.
```python
def perturb(atoms: Atoms, num=20, cell_pert_fraction=0.03, min_distance=0.1, ...):
    ...
```

### Feature-Specific Functions

#### `generate_vacancies`
```python
def generate_vacancies(atoms: Atoms, elements: str, num: int, mode='random', rng_values=None) -> tuple[Atoms, list]
```

#### `shuffle_element_positions`
```python
def shuffle_element_positions(structure_data, element_range, shuffle_method='fisher_yates', seed=None, rng_values=None) -> tuple[Atoms, dict]
```

#### `generate_surface`
```python
def generate_surface(atoms: Atoms, indices=(1,1,1), vacuum=10.0, layers=3) -> Atoms
```

#### `generate_grain_boundary`
```python
def generate_grain_boundary(atoms: Atoms, axis=[0,0,1], angle_deg=36.87, plane_height_frac=0.5, delete_overlap=True, min_dist=1.5) -> Atoms
```

#### `generate_dislocation`
```python
def generate_dislocation(atoms: Atoms, type='edge', axis=2, burgers=2.55, nu=0.33) -> Atoms
```

#### `generate_twinning`
```python
def generate_twinning(atoms: Atoms, miller_indices=(1,1,1), z_frac=0.5, min_dist=1.5) -> Atoms
```

#### `generate_amorphous`
```python
def generate_amorphous(atoms: Atoms, min_dist=1.5, rattle_strength=0.5, max_steps=100, rng_values=None) -> Atoms
```

#### `apply_magnetic_perturbation`
```python
def apply_magnetic_perturbation(atoms: Atoms, mode='collinear', mag_config=None, flip_prob=0.5, noise=0.0) -> Atoms
```

#### `rotate_fragments_by_formula`
```python
def rotate_fragments_by_formula(atoms: Atoms, formula: str, mult=1.2, seed=None) -> Atoms
```

#### `generate_antisite_defects`
```python
def generate_antisite_defects(
    structure: Atoms,
    swap_pairs: list[tuple[str, str]],
    num_swaps: int = 1,
    mode: str = 'symmetry_aware',  # 'symmetry_aware' | 'random'
    rng_values: np.ndarray | None = None,
    symprec: float = 1e-2,
) -> tuple[Atoms, dict]
```

#### `get_equivalent_sites`
```python
def get_equivalent_sites(atoms: Atoms, symprec: float = 1e-2) -> dict[str, list[list[int]]]
```

#### `generate_symmetry_preserving_strain`
```python
def generate_symmetry_preserving_strain(
    atoms: Atoms,
    strain_fraction: float = 0.03,
    crystal_system: str | None = None,
    symprec: float = 1e-2,
    rng_values: np.ndarray | None = None,
    min_distance: float = 0.1,
) -> tuple[Atoms, dict]
```

#### `detect_crystal_system_spglib`
```python
def detect_crystal_system_spglib(atoms: Atoms, symprec: float = 1e-2) -> str
```

#### `get_independent_strain_count`
```python
def get_independent_strain_count(crystal_system: str) -> int
```

#### `validate_compatibility`
```python
def validate_compatibility(**kwargs) -> list[str]  # warnings; raises ValueError on conflicts
```

## Troubleshooting

### Fragments Not Rotating
- **Cause**: Connectivity algorithm fails to identify discrete molecules.
- **Fix**: Adjust `mult` (bond cutoff multiplier) or ensure `formula` matches exactly (e.g. `H2O` vs `OH2`).

### Magnetic Moments Missing
- **Cause**: Element not in configuration or `mag_mode` not set.
- **Fix**: Define moments in `config.ini` under `[magmom]` and use `--mag-mode`.

### High Energy Structures
- **Cause**: Overlaps in generated defects (GB, amorphous).
- **Fix**: Increase `--gb-dist` or `--amorphous-min-dist`. Use `--skip-normal` to isolate the defect generation from random rattling.

## Output
Structures are written to the file specified by `--out` (default `./perturb.xyz`). Use `--append` to add to existing files.
