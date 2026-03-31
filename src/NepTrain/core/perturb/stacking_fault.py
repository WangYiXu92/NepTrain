import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList

def generate_stacking_fault(atoms: Atoms, plane_normal=[1, 1, 1], shift_vector=[0, 0, 0], plane_height_frac=0.5, min_dist=1.5, translation_frac=None) -> Atoms:
    """
    Generate a stacking fault perturbation by shifting the upper half of the crystal.
    
    This function mimics Atomsk's `-stacking-fault` command. It divides the crystal
    into two halves along a specified plane and shifts one half by a vector.
    
    Args:
        atoms: Input structure.
        plane_normal: Normal vector of the fault plane (Miller indices or Cartesian).
                      If integers, assumed to be Miller indices (hkl).
                      If floats, assumed to be Cartesian direction.
        shift_vector: Shift vector [vx, vy, vz] in Angstroms.
        plane_height_frac: Fractional height (0-1) along the plane normal where the fault is located.
        min_dist: Minimum distance to delete overlaps.
        translation_frac: Tuple of (tx, ty) fractional coordinates (0-1) for random translation
                          in the fault plane. Used for Gamma surface sampling.
                          The basis for this translation is derived from the projection of
                          cell vectors onto the fault plane.
        
    Returns:
        Atoms object with stacking fault.
    """
    atoms = atoms.copy()
    
    # 1. Normalize plane normal
    # If plane_normal is integer-like, treat as Miller indices
    # But for general case, let's try to interpret.
    # If using ASE, atoms.cell.reciprocal() gives reciprocal vectors.
    
    # Check if Miller indices (integers)
    # Check if all elements are integers (handle strings if passed, though type hint says list)
    # Be robust to list of ints or floats that are ints
    is_miller = all(isinstance(x, (int, np.integer)) for x in plane_normal)
    
    if is_miller:
        reciprocal_cell = atoms.cell.reciprocal() # Rows are a*, b*, c*
        normal_cart = np.dot(plane_normal, reciprocal_cell)
    else:
        normal_cart = np.array(plane_normal, dtype=float)
        
    norm = np.linalg.norm(normal_cart)
    if norm < 1e-8:
        return atoms
    normal_unit = normal_cart / norm
    
    # Calculate additional translation from fractional coordinates if provided
    extra_shift = np.zeros(3)
    if translation_frac is not None:
        # Project cell vectors onto the plane
        cell = atoms.get_cell()
        # v_proj = v - (v . n) * n
        projections = []
        for v in cell:
            p = v - np.dot(v, normal_unit) * normal_unit
            projections.append(p)
            
        # Select two basis vectors from projections
        # Heuristic: Sort by length, pick two longest non-collinear
        projections.sort(key=lambda x: np.linalg.norm(x), reverse=True)
        
        u = projections[0]
        v = None
        
        # Find second vector
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-6:
            # All projections are zero? (e.g. 1D chain along normal?)
            # Create arbitrary basis
            if abs(normal_unit[2]) < 0.9:
                u = np.cross(normal_unit, [0, 0, 1])
            else:
                u = np.cross(normal_unit, [0, 1, 0])
            v = np.cross(normal_unit, u)
        else:
            for cand in projections[1:]:
                # Check collinearity via cross product
                cross = np.cross(u, cand)
                if np.linalg.norm(cross) > 1e-4 * u_norm * np.linalg.norm(cand):
                    v = cand
                    break
            
            if v is None:
                # All non-zero projections are collinear?
                v = np.cross(normal_unit, u)
        
        extra_shift = translation_frac[0] * u + translation_frac[1] * v

    # 2. Project atoms onto the normal direction to find "height"
    positions = atoms.get_positions()
    
    # We need to define the cut plane position.
    # Project all atoms onto normal
    projections = np.dot(positions, normal_unit)
    
    # Find the range
    min_proj = np.min(projections)
    max_proj = np.max(projections)
    length = max_proj - min_proj
    
    # Ideally, for PBC, we should look at the cell dimensions along that direction.
    # But the cell might not be orthogonal or aligned.
    # Let's use the actual atom spread or the cell center.
    # If plane_height_frac is used, we assume it's relative to the range of atoms or cell center.
    # Atomsk usually uses a coordinate or '0.5*box'.
    
    # Let's assume cut is at min + frac * length
    cut_pos = min_proj + plane_height_frac * length
    
    # 3. Identify atoms above the cut
    # "Above" means projection > cut_pos
    mask = projections > cut_pos
    indices = np.where(mask)[0]
    
    if len(indices) == 0 or len(indices) == len(atoms):
        return atoms
        
    # 4. Shift atoms
    shift = np.array(shift_vector, dtype=float) + extra_shift
    positions[indices] += shift
    
    atoms.set_positions(positions)
    atoms.wrap() # Wrap is essential for PBC
    
    # 5. Annotation
    atoms.info['perturb_annotation'] = {
        'type': 'stacking_fault',
        'plane_normal': list(plane_normal), # Keep original input
        'plane_height': float(cut_pos), # Cartesian height along normal
        'shift_vector': list(shift),
        'metadata': {
            'input_shift': list(shift_vector),
            'translation_frac': list(translation_frac) if translation_frac is not None else None
        }
    }
    
    # 6. Remove overlaps if min_dist > 0
    if min_dist > 0:
        indices_to_remove = set()
        
        # Use NeighborList for efficient distance checking with PBC
        # Cutoff needs to be slightly larger than min_dist to be safe for NeighborList
        # (NeighborList uses sum of radii = cutoff, so we set cutoff = min_dist/2 + epsilon)
        cutoffs = [min_dist/2 + 0.1] * len(atoms)
        nl = NeighborList(cutoffs, skin=0.0, sorted=True, self_interaction=False, bothways=True)
        nl.update(atoms)
        
        for i in range(len(atoms)):
            if i in indices_to_remove:
                continue
            
            indices, offsets = nl.get_neighbors(i)
            for j in indices:
                # Avoid double checking
                if j in indices_to_remove:
                    continue
                
                # Check actual distance
                dist = atoms.get_distance(i, j, mic=True)
                if dist < min_dist:
                    # Remove one of them. Deterministic choice: higher index.
                    if j > i:
                        indices_to_remove.add(j)
                    else:
                        indices_to_remove.add(i)
        
        if indices_to_remove:
            del atoms[sorted(list(indices_to_remove), reverse=True)]
    
    return atoms
