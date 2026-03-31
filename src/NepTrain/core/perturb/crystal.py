import logging
import numpy as np
from ase import Atoms
from ase.build import cut, make_supercell

# Set up module logger
logger = logging.getLogger(__name__)

def find_primitive_period(atoms: Atoms, direction: list, max_denominator: int = 12, tol: float = 1e-4) -> list:
    """
    Finds the shortest periodic lattice vector parallel to the given direction vector.
    
    Args:
        atoms: The crystal structure (e.g. conventional cell).
        direction: Vector [u, v, w] in terms of the input cell basis (usually integers).
        max_denominator: Maximum integer n to check for 1/n fractions.
        tol: Tolerance for position matching.
        
    Returns:
        The shortest vector (list) parallel to direction that is a lattice vector.
        e.g., [1, 1, -2] -> [0.5, 0.5, -1.0] for FCC.
    """
    # Convert direction to numpy array
    direction = np.array(direction, dtype=float)
    
    # Get scaled positions of all atoms in the unit cell
    scaled_positions = atoms.get_scaled_positions()
    
    # We want to find the smallest k = 1/n such that k * direction is a valid translation.
    # A valid translation T (in fractional coords) must map every atom to an equivalent position.
    # For simple lattices (mono-atomic basis), T must connect any atom to any other atom (modulo 1).
    # For multi-atomic bases, T must connect sublattice A to sublattice A.
    # Let's assume we just need T to be a Bravais lattice vector.
    # A robust check: T is a lattice vector if (positions + T) % 1 overlaps with positions.
    
    # Optimization: Check if T maps atom 0 to ANY atom in the cell (of same species).
    # If so, and if it's a true lattice vector, it should work for all.
    # We'll assume the input 'atoms' has valid PBC and symmetry.
    
    # Get symbol of first atom
    first_symbol = atoms[0].symbol
    # Get indices of all atoms with same symbol
    same_species_indices = [i for i, atom in enumerate(atoms) if atom.symbol == first_symbol]
    target_positions = scaled_positions[same_species_indices]
    ref_pos = scaled_positions[0]
    
    # Calculate all difference vectors from ref_pos to other same-species atoms
    # diffs = target_positions - ref_pos
    # We also need to consider periodic images, so we look at (diffs) % 1.0
    # Actually, T % 1.0 must be equal to one of (target_positions - ref_pos) % 1.0
    
    # Pre-calculate possible fractional shifts (modulo 1)
    # diffs_mod = (target_positions - ref_pos) % 1.0
    # But this is sensitive to wrapping. 
    # Better: calculate minimum image distance between (ref_pos + T) and all target_positions.
    
    best_n = 1
    
    # Check n from max down to 1 (to find smallest vector, i.e. largest n)
    # Wait, we want the shortest vector, so we want the largest n such that 1/n is valid.
    for n in range(max_denominator, 0, -1):
        k = 1.0 / n
        test_vec = direction * k
        
        # Check if test_vec is a lattice vector
        # Shift atom 0 by test_vec
        shifted_pos = ref_pos + test_vec
        
        # Check distance to all same-species atoms (with MIC)
        # We need to find IF there is ANY atom j such that dist(shifted_pos, atom_j) ~ 0
        
        dists = []
        for target in target_positions:
            # fractional difference
            d = shifted_pos - target
            # wrap to [-0.5, 0.5]
            d -= np.round(d)
            # Cartesian distance check? Or just fractional check?
            # Fractional check is fine if we use a small tolerance.
            # But converting to Cartesian is safer for 'tol' in Angstroms.
            # Let's just use fractional distance squared
            d2 = np.sum(d**2)
            dists.append(d2)
            
        min_dist_sq = np.min(dists)
        
        if min_dist_sq < tol**2:
            # Found a match!
            best_n = n
            break # Since we started from largest n, this is the shortest vector.
            
    return (direction / best_n).tolist()

def create_oriented_supercell(atoms: Atoms, directions: list, size: tuple = (1, 1, 1), tolerance: float = 1e-3, minimize_basis: bool = False) -> Atoms:
    """
    Creates a supercell with specific crystallographic orientations.
    
    This function uses the input atoms (usually a conventional unit cell) and
    cuts a new cell defined by the given directions (integer linear combinations
    of the input basis vectors).
    
    Args:
        atoms: Input unit cell (e.g., conventional cubic cell).
        directions: List of 3 integer vectors [[h1,k1,l1], [h2,k2,l2], [h3,k3,l3]].
                    These define the new a, b, c axes in terms of the input cell vectors.
        size: Supercell size (nx, ny, nz) along the new axes.
        tolerance: Tolerance for checking orthogonality (optional).
        minimize_basis: If True, attempts to reduce the basis vectors to their 
                        shortest periodic length (primitive lattice vectors).
                        Crucial for minimizing system size.
        
    Returns:
        The oriented supercell.
    """
    # Validate directions
    if len(directions) != 3:
        raise ValueError("Must provide exactly 3 direction vectors.")
    
    # Copy directions to avoid modifying input
    final_directions = [np.array(d, dtype=float) for d in directions]
    
    if minimize_basis:
        # Attempt to reduce each direction vector
        final_directions = [find_primitive_period(atoms, d) for d in directions]
        # print(f"Minimized basis: {directions} -> {final_directions}")
    
    # Check if directions form a valid basis (non-zero determinant)
    matrix = np.array(final_directions)
    det = np.linalg.det(matrix)
    if abs(det) < 1e-5:
        raise ValueError("Direction vectors are linearly dependent (determinant is zero).")
    
    # Check for right-handedness?
    # ASE 'cut' might handle left-handed, but physics usually prefers right-handed.
    if det < 0:
        logger.warning("The provided basis is left-handed. The resulting cell volume will be negative/inverted.")
    
    # Check orthogonality (optional but recommended for simulation boxes)
    # Note: Orthogonality depends on the input metric tensor.
    # If input is cubic, then dot product of indices is 0 -> orthogonal.
    # We assume the user knows what they are doing, but we can warn.
    # cell = atoms.get_cell()
    # v1 = np.dot(directions[0], cell)
    # v2 = np.dot(directions[1], cell)
    # v3 = np.dot(directions[2], cell)
    # ... (orthogonality check) ...
    
    # Create the new unit cell
    # ASE's cut function: a, b, c are the new basis vectors in terms of old ones.
    # This perfectly matches our 'directions' argument.
    # Note: 'cut' might not handle the 'origo' shift if not needed, defaults to (0,0,0).
    try:
        new_atoms = cut(atoms, a=final_directions[0], b=final_directions[1], c=final_directions[2])
    except Exception as e:
        # Fallback or re-raise
        raise RuntimeError(f"ASE cut failed: {e}. Check if directions define a valid cell.")
    
    # Check if the new cell is right-handed (positive determinant)
    if new_atoms.cell.volume < 0:
        # If ASE returns negative volume, it means the basis is left-handed.
        # We can flip the 3rd axis to make it right-handed.
        # Or swap 1 and 2.
        # Let's try to enforce positive volume by flipping all axes if needed?
        # No, just swap a and b.
        # But we want to respect the user's direction assignment (a=dir[0]).
        # If the user gave a left-handed basis, they get a left-handed cell.
        # We just warn.
        pass

    # Apply supercell repetition
    if size != (1, 1, 1):
        # We use a diagonal matrix because we are repeating along the NEW axes
        new_atoms = make_supercell(new_atoms, np.diag(size))
        
    return new_atoms

def align_cell_to_cartesian(atoms: Atoms, tol: float = 1e-4) -> Atoms:
    """
    Rotates the atoms object so that its cell vectors align with the Cartesian axes.
    Ideally:
      a (vector 0) -> x axis [L, 0, 0]
      b (vector 1) -> xy plane [Lx, Ly, 0]
      c (vector 2) -> z component [Lx, Ly, Lz]
    
    This corresponds to the standard triangular (upper triangular) cell representation.
    
    Args:
        atoms: Input atoms.
        tol: Tolerance.
        
    Returns:
        New rotated Atoms object.
    """
    new_atoms = atoms.copy()
    cell = new_atoms.get_cell()
    
    # We want to rotate the system such that cell[0] is along x, cell[1] in xy.
    # We can use ASE's built-in functionality implicitly, or do it manually.
    
    # Method 1: Use geometry rotation
    # v1 = cell[0]
    # v2 = cell[1]
    # v3 = cell[2]
    
    # Target frame:
    # x_hat aligned with v1
    # y_hat aligned with component of v2 perpendicular to v1
    # z_hat aligned with v1 x v2 (or whatever completes the basis)
    
    # Calculate current basis vectors
    v1 = cell[0]
    v2 = cell[1]
    
    # Normalize v1
    x_hat_new = v1 / np.linalg.norm(v1)
    
    # Find y_hat_new
    # Project v2 onto v1
    proj = np.dot(v2, x_hat_new) * x_hat_new
    perp = v2 - proj
    if np.linalg.norm(perp) < tol:
        # Collinear vectors? Should not happen for valid cell
        raise ValueError("Cell vectors a and b are collinear.")
    y_hat_new = perp / np.linalg.norm(perp)
    
    # Find z_hat_new
    z_hat_new = np.cross(x_hat_new, y_hat_new)
    
    # Construct rotation matrix R from Old to New(Cartesian)
    # The columns of the inverse rotation matrix are the new basis vectors expressed in the old basis?
    # No.
    # We have the New Basis Vectors expressed in the Old (Cartesian) frame:
    # e1' = x_hat_new
    # e2' = y_hat_new
    # e3' = z_hat_new
    #
    # We want to map e1' -> (1,0,0), e2' -> (0,1,0), e3' -> (0,0,1).
    #
    # Let M = [e1', e2', e3'] (3x3 matrix with columns as the new basis vectors)
    # R * M = I
    # So R = M_inverse = M_transpose (since M is orthogonal).
    
    M = np.column_stack([x_hat_new, y_hat_new, z_hat_new])
    R = M.T
    
    # Apply rotation
    # Positions: p_new = R * p_old
    # Cell: C_new = C_old * R.T ? Or just rotate the vectors?
    # atoms.rotate uses a rotation matrix? No, it uses Euler angles or vector/vector.
    # We can use set_cell and set_positions.
    
    pos = new_atoms.get_positions()
    new_pos = np.dot(pos, R.T)
    new_cell = np.dot(cell, R.T)
    
    # Clean up small values
    new_cell[np.abs(new_cell) < tol] = 0.0
    
    new_atoms.set_cell(new_cell)
    new_atoms.set_positions(new_pos)
    
    return new_atoms

def get_burgers_vector(atoms: Atoms, direction: list) -> float:
    """
    Calculates the magnitude of a Burgers vector in a given direction.
    
    Args:
        atoms: Input unit cell.
        direction: Integer vector [u, v, w] in terms of lattice basis.
        
    Returns:
        Magnitude of the vector.
    """
    cell = atoms.get_cell()
    # Vector in Cartesian coordinates
    v = np.dot(direction, cell)
    return np.linalg.norm(v)
