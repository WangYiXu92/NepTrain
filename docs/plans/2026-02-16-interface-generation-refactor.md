# Design: CSL-Based Interface Generation Refactor

## 1. Motivation
The current implementation relies on fragile manual fallbacks or heavy external dependencies (`pymatgen`). We need a robust, lightweight, and flexible generation engine for Grain Boundaries (GBs) and Twinning, built from scratch using `numpy` and `ase`.

## 2. Core Components

### A. `csl_core.py` (New Module)
A pure-Python implementation of Coincidence Site Lattice (CSL) theory for cubic systems (FCC/BCC).

*   **Theory**: Based on quaternion representation of rotations.
    *   Rotation $R$ is defined by quaternion $[m, n, p, q]$.
    *   Angle $\theta$: $\tan(\theta/2) = \frac{\sqrt{n^2+p^2+q^2}}{m}$
    *   Axis $[u, v, w] = [n, p, q]$
    *   $\Sigma = m^2 + n^2 + p^2 + q^2$ (divided by common factors)
*   **Key Functions**:
    *   `generate_csl_parameters(sigma_max: int, axis: list)`: Returns list of valid $(\theta, \Sigma)$ pairs for a given rotation axis.
    *   `get_rotation_matrix(sigma: int, axis: list)`: Returns the exact rotation matrix for a specific $\Sigma$.
    *   `get_csl_basis(primitive_cell, rotation_matrix)`: Constructs the CSL supercell basis vectors.

### B. `InterfaceBuilder` (Refactored `interface_generator.py`)
A high-level builder that uses `csl_core` to construct interfaces.

*   **Workflow**:
    1.  **Input**: Primitive structure, Sigma value, Rotation Axis, Plane Normal.
    2.  **CSL Generation**: Use `csl_core` to find the rotation.
    3.  **Grain Creation**:
        *   **Grain A**: Oriented slab (e.g., using `ase.build.surface` or custom rotation).
        *   **Grain B**: Grain A rotated by the CSL angle.
    4.  **Interface Construction**:
        *   Stack Grain A and Grain B.
        *   Apply **Rigid Body Translation (RBT)** optimization (reuse `InterfaceOptimizer`).
        *   Validate minimal distances.

### C. `grain_boundary.py` & `twinning.py` (Wrappers)
Simplified wrappers that expose a clean API to the user/registry.

*   `generate_grain_boundary(structure, sigma, axis)` -> Calls `InterfaceBuilder`.
*   `generate_twinning(structure, plane)` -> Calls `InterfaceBuilder` with fixed Sigma=3 (for FCC/BCC).

## 3. Implementation Plan

1.  **Step 1**: Implement `NepTrain.core.perturb.csl_core`.
2.  **Step 2**: Refactor `NepTrain.core.perturb.interface_generator` to use `csl_core` and remove old logic.
3.  **Step 3**: Rewrite `grain_boundary.py` and `twinning.py` to use the new builder.
4.  **Step 4**: Verify with `test_refactored_interfaces.py`.

## 4. Key Advantages
*   **Zero Dependencies**: No `pymatgen`.
*   **Exactness**: Uses analytical CSL theory, not "try-and-error" rotation.
*   **Performance**: Fast numpy operations.
*   **Flexibility**: Supports any Sigma value allowed by cubic symmetry.
