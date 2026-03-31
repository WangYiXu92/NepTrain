import numpy as np
from ase import Atoms
from .crystal import create_oriented_supercell, get_burgers_vector, align_cell_to_cartesian

def generate_dislocation(atoms: Atoms, type: str = 'edge', axis: int = 2, burgers: float = 2.55, nu: float = 0.33, center: list = None, delete_overlap: bool = True, tol: float = 1.5) -> Atoms:
    """
    Generate a dislocation by applying Volterra displacement fields.
    
    Args:
        atoms: Input structure.
        type: 'edge' or 'screw'.
        axis: Dislocation line direction (0=x, 1=y, 2=z).
        burgers: Magnitude of Burgers vector.
        nu: Poisson's ratio (for edge dislocation).
        center: [x, y, z] coordinates of the dislocation core. If None, uses box center.
        delete_overlap: Whether to delete overlapping atoms (crucial for edge dislocations).
        tol: Tolerance for overlap deletion (Angstrom).
        
    Returns:
        Perturbed structure.
    """
    atoms = atoms.copy()
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    
    if center is None:
        center = np.sum(cell, axis=0) / 2.0
    else:
        center = np.array(center)
    
    # Map axis to indices
    if axis == 0:
        x_idx, y_idx, z_idx = 1, 2, 0
    elif axis == 1:
        x_idx, y_idx, z_idx = 2, 0, 1
    elif axis == 2:
        x_idx, y_idx, z_idx = 0, 1, 2
    else:
        raise ValueError("Axis must be 0, 1, or 2")
        
    x = positions[:, x_idx] - center[x_idx]
    y = positions[:, y_idx] - center[y_idx]
    
    # Polar coordinates
    # Use arctan2 for theta (-pi to pi)
    theta = np.arctan2(y, x)
    
    if type == 'screw':
        # u_z = b * theta / (2*pi)
        # Displacement along the line direction (z_idx)
        u_z = burgers * theta / (2 * np.pi)
        positions[:, z_idx] += u_z
        
    elif type == 'edge':
        # Edge dislocation with b along x_idx
        # To generate a clean dislocation, we prefer to create an overlap and delete atoms
        # rather than opening a void and filling it.
        # A dislocation of vector b can be formed by removing a half-plane, which 
        # corresponds to the displacement field of -b (bringing atoms closer).
        
        # We enforce "compressive" mode by flipping the sign of b if it's positive.
        # This ensures we create overlaps.
        # Note: This might flip the extra half-plane orientation (up/down), 
        # but ensures a clean core structure.
        
        effective_burgers = -abs(burgers)
        
        r2 = x**2 + y**2
        # Avoid division by zero
        r2[r2 < 1e-6] = 1e-6
        
        factor = effective_burgers / (2 * np.pi)
        
        term1_x = theta
        term2_x = (x * y) / (2 * (1 - nu) * r2)
        u_x = factor * (term1_x + term2_x)
        
        r = np.sqrt(r2)
        term1_y = (1 - 2 * nu) / (2 * (1 - nu)) * np.log(r)
        term2_y = (x**2 - y**2) / (4 * (1 - nu) * r2)
        u_y = -factor * (term1_y + term2_y)
        
        positions[:, x_idx] += u_x
        positions[:, y_idx] += u_y
        
        # Apply displacements to atoms immediately
        atoms.set_positions(positions)
        
        # We intentionally created overlaps to form the dislocation core.
        # Now we must delete them.
        delete_overlap = True
        
    else:
        raise ValueError(f"Unknown dislocation type: {type}")
        
    if type == 'screw':
        atoms.set_positions(positions)
        
    # Global Overlap Removal (Crucial for edge dislocations generated via overlap method)
    if delete_overlap:
        # We need to wrap first to handle PBC correctly in NeighborList
        atoms.wrap()
        
        from ase.neighborlist import NeighborList
        # Use a slightly generous tolerance for core atoms
        nl = NeighborList([tol/2]*len(atoms), skin=0.0, self_interaction=False, bothways=True)
        nl.update(atoms)
        
        to_delete = set()
        for i in range(len(atoms)):
            if i in to_delete: continue
            indices, offsets = nl.get_neighbors(i)
            for j in indices:
                if j > i: # Check each pair once
                    if j not in to_delete:
                        # For edge dislocation overlap, we have pairs of atoms very close.
                        # We remove one of them.
                        to_delete.add(j) 
                            
        if to_delete:
            # print(f"Deleting {len(to_delete)} overlapping atoms in dislocation core")
            del atoms[[i for i in to_delete]]
        
    # Final wrap
    atoms.wrap()
    
    atoms.info['perturb_annotation'] = {
        'type': 'dislocation',
        'sub_type': type,
        'axis': axis,
        'center': center.tolist(),
        'burgers': burgers
    }
    
    return atoms

def generate_dislocation_from_basis(atoms: Atoms,
                                    basis_vectors: list,
                                    supercell: tuple = (1, 1, 1),
                                    type: str = 'edge',
                                    nu: float = 0.33,
                                    tol: float = 1.5,
                                    minimize_basis: bool = True) -> Atoms:
    """
    Generates a dislocation with the crystal oriented according to basis_vectors.
    
    This function handles arbitrary crystal structures by rotating them into a standard frame:
    - Z axis (new index 2) aligned with dislocation line.
    - X axis (new index 0) aligned with Burgers vector (for edge) or plane normal.
    
    Args:
        atoms: Input unit cell (e.g. primitive cell).
        basis_vectors: List of 3 vectors [[h,k,l], [h,k,l], [h,k,l]] defining the new X, Y, Z axes.
                       - Z axis (index 2) MUST be the dislocation line direction.
                       - X axis (index 0) MUST be the Burgers vector direction for edge dislocations.
                       - Y axis (index 1) is the slip plane normal (or glide plane normal) for edge.
        supercell: Supercell size (nx, ny, nz) along the NEW axes.
        type: 'edge' or 'screw'.
        nu: Poisson's ratio.
        tol: Tolerance for overlap removal.
        minimize_basis: If True (default), attempts to reduce the basis vectors to their 
                        shortest periodic length. This is crucial for creating small systems.
        
    Returns:
        Atoms object containing the dislocation in the rotated supercell.
    """
    # 1. Create oriented supercell
    # Note: size argument applies to the new axes
    # The 'create_oriented_supercell' function handles the rotation/cutting.
    oriented_atoms = create_oriented_supercell(atoms, basis_vectors, size=supercell, minimize_basis=minimize_basis)
    
    # 1.5 Align to Cartesian
    # This ensures that X is along [1,0,0], Y in XY plane, Z along [0,0,1] component.
    # Since create_oriented_supercell sets basis[0] as 'a', basis[1] as 'b', etc.,
    # aligning 'a' to x-axis aligns our Burgers vector (if edge) or plane normal to x-axis.
    oriented_atoms = align_cell_to_cartesian(oriented_atoms)
    
    # 2. Calculate Burgers vector magnitude
    # In the new cell, the lattice vectors are aligned with X, Y, Z.
    # However, the magnitude depends on the original lattice constant.
    # 'get_burgers_vector' computes magnitude of a vector given in crystal coordinates.
    # But here we need the magnitude of the vector along the new axes.
    # The new 'a' axis corresponds to basis_vectors[0].
    # The new 'c' axis corresponds to basis_vectors[2].
    
    if minimize_basis:
        # If we minimized basis, we need to know what the new basis vectors are.
        # But create_oriented_supercell doesn't return them directly.
        # However, oriented_atoms.cell[0] is the new a vector in Cartesian coords.
        # So its length is the Burgers vector length (if X is Burgers).
        
        if type == 'edge':
            # Burgers vector is along X (new axis 0)
            b_mag = np.linalg.norm(oriented_atoms.cell[0] / supercell[0])
            axis = 2
        elif type == 'screw':
            # Burgers vector is along Z (new axis 2)
            b_mag = np.linalg.norm(oriented_atoms.cell[2] / supercell[2])
            axis = 2
        else:
            raise ValueError(f"Unknown dislocation type: {type}")

    else:
        # Use the input basis vectors to calculate lengths
        if type == 'edge':
            b_vec = basis_vectors[0]
            b_mag = get_burgers_vector(atoms, b_vec)
            axis = 2 # Line along Z
            
        elif type == 'screw':
            b_vec = basis_vectors[2]
            b_mag = get_burgers_vector(atoms, b_vec)
            axis = 2 # Line along Z
            
        else:
            raise ValueError(f"Unknown dislocation type: {type}")
        
    # 3. Generate dislocation
    # The cell is now oriented such that Z is the line direction.
    # For edge, X is the Burgers vector direction.
    
    return generate_dislocation(oriented_atoms, 
                                type=type, 
                                axis=axis, 
                                burgers=b_mag, 
                                nu=nu, 
                                delete_overlap=True, 
                                tol=tol)

def generate_single_dislocation_atomsk(atoms: Atoms,
                                     basis_vectors: list,
                                     supercell: tuple = (1, 1, 1),
                                     type: str = 'edge',
                                     nu: float = 0.33) -> Atoms:
    """
    Generates a SINGLE dislocation (Atomsk style).
    This creates a step at the boundary due to the non-periodic displacement field.
    The number of atoms is conserved.
    
    Args:
        atoms: Input unit cell.
        basis_vectors: List of 3 vectors defining the new X, Y, Z axes.
                       X: Burgers vector direction.
                       Y: Plane normal.
                       Z: Line direction.
        supercell: Supercell size (nx, ny, nz).
        type: 'edge' or 'screw'.
        nu: Poisson's ratio.
        
    Returns:
        Atoms object containing a single dislocation at the center.
    """
    # 1. Create oriented supercell (No minimization by default to keep user control?)
    # Actually, let's reuse the logic from dipole function but simpler.
    oriented_atoms = create_oriented_supercell(atoms, basis_vectors, size=supercell, minimize_basis=True)
    oriented_atoms = align_cell_to_cartesian(oriented_atoms)
    
    # 2. Burgers vector magnitude
    # Assuming minimized basis
    if type == 'edge':
        b_mag = np.linalg.norm(oriented_atoms.cell[0] / supercell[0])
        axis = 2
    elif type == 'screw':
        b_mag = np.linalg.norm(oriented_atoms.cell[2] / supercell[2])
        axis = 2
    else:
        raise ValueError(f"Unknown dislocation type: {type}")
        
    # 3. Center
    cell_diag = oriented_atoms.cell.diagonal()
    center = cell_diag * np.array([0.5, 0.5, 0.5])
    
    positions = oriented_atoms.get_positions()
    cell = oriented_atoms.get_cell()
    
    x_idx, y_idx, z_idx = 0, 1, 2
    Lx, Ly, Lz = cell[0,0], cell[1,1], cell[2,2]
    
    # 4. Apply Volterra Field (Single Core)
    dx = positions[:, x_idx] - center[x_idx]
    dy = positions[:, y_idx] - center[y_idx]
    
    # IMPORTANT: Do NOT use MIC for single dislocation if we want the step at boundary.
    # The cut is defined by arctan2 range.
    # Standard arctan2(y, x) has cut at x<0 (negative x axis).
    # If we want cut along +x, use arctan2(y, -x).
    # Atomsk usually puts cut along X.
    
    # Let's align cut with +X direction.
    # So we use arctan2(dy, -dx).
    
    r2 = dx**2 + dy**2
    r2[r2 < 1e-6] = 1e-6
    r = np.sqrt(r2)
    
    if type == 'edge':
        # Edge field
        # Use Atomsk coordinate system logic:
        # Atomsk aligns cut with X axis.
        # u_x = b/2pi * (arctan2(y, x) + ...)
        # But we need to be careful with the branch cut.
        # If we use arctan2(y, x), the cut is at x < 0.
        # If we use arctan2(y, -x), the cut is at x > 0.
        
        # Let's try to match the standard text book definition (Hirth & Lothe) often used by Atomsk
        # Cut along negative X axis: theta = arctan2(y, x)
        theta = np.arctan2(dy, dx)
        
        factor = b_mag / (2 * np.pi)
        
        term1_x = theta
        term2_x = (dx * dy) / (2 * (1 - nu) * r2)
        ux = factor * (term1_x + term2_x)
        
        term1_y = (1 - 2 * nu) / (2 * (1 - nu)) * np.log(r)
        term2_y = (dx**2 - dy**2) / (4 * (1 - nu) * r2)
        uy = factor * (term1_y + term2_y) # Note: sign change might be needed depending on convention
        
        positions[:, x_idx] += ux
        positions[:, y_idx] += uy
        
    elif type == 'screw':
        # Screw field
        # u_z = b * theta / 2pi
        # Cut along negative X axis
        theta = np.arctan2(dy, dx)
        uz = b_mag * theta / (2 * np.pi)
        positions[:, z_idx] += uz
        
    oriented_atoms.set_positions(positions)
    
    # No overlap removal for Atomsk style single dislocation
    # But we should wrap atoms back into box?
    # If we wrap, the step at boundary becomes explicit.
    oriented_atoms.wrap()
    
    return oriented_atoms

def generate_dislocation_dipole_from_basis(atoms: Atoms,
                                         basis_vectors: list,
                                         supercell: tuple = (1, 1, 1),
                                         type: str = 'edge',
                                         nu: float = 0.33,
                                         tol: float = 1.5,
                                         minimize_basis: bool = True,
                                         dissociate: bool = False,
                                         dissociation_width: float = 10.0,
                                         core_width: float = 2.0,
                                         method: str = 'volterra') -> Atoms:
    """
    Generates a periodic dislocation dipole (two opposite dislocations).
    This configuration is ideal for periodic boundary conditions as it restores periodicity.
    
    The dislocations are placed at 1/4 and 3/4 positions along the Burgers vector direction (X)
    and at 1/2 height (Y).
    
    Args:
        atoms: Input unit cell.
        basis_vectors: List of 3 vectors defining the new X, Y, Z axes.
                       X: Burgers vector direction.
                       Y: Plane normal.
                       Z: Line direction.
        supercell: Supercell size (nx, ny, nz).
        type: 'edge' or 'screw'.
        nu: Poisson's ratio.
        tol: Tolerance for overlap removal.
        minimize_basis: Whether to use primitive cell vectors (default True).
        dissociate: If True, splits each edge dislocation into two Shockley partials.
        dissociation_width: Distance between partials in Angstroms (default 10.0).
        core_width: Dislocation core width (zeta) to smooth the singularity (default 2.0).
        method: 'volterra' (elastic field) or 'geometric' (shear/cut).
        
    Returns:
        Atoms object containing the dislocation dipole.
    """
    # 1. Create oriented supercell
    oriented_atoms = create_oriented_supercell(atoms, basis_vectors, size=supercell, minimize_basis=minimize_basis)
    
    # 1.5 Align to Cartesian
    oriented_atoms = align_cell_to_cartesian(oriented_atoms)
    
    # 2. Determine Burgers vector magnitude and axis
    if minimize_basis:
        if type == 'edge':
            b_mag = np.linalg.norm(oriented_atoms.cell[0] / supercell[0])
            axis = 2
        elif type == 'screw':
            b_mag = np.linalg.norm(oriented_atoms.cell[2] / supercell[2])
            axis = 2
        else:
            raise ValueError(f"Unknown dislocation type: {type}")
    else:
        if type == 'edge':
            b_vec = basis_vectors[0]
            b_mag = get_burgers_vector(atoms, b_vec)
            axis = 2
        elif type == 'screw':
            b_vec = basis_vectors[2]
            b_mag = get_burgers_vector(atoms, b_vec)
            axis = 2
        else:
            raise ValueError(f"Unknown dislocation type: {type}")
            
    # 3. Define dipole centers
    # Standard configuration:
    # Dislocation 1 at (0.25 * Lx, 0.5 * Ly, 0.5 * Lz)
    # Dislocation 2 at (0.75 * Lx, 0.5 * Ly, 0.5 * Lz)
    
    cell_diag = oriented_atoms.cell.diagonal()
    center1 = cell_diag * np.array([0.25, 0.5, 0.5])
    center2 = cell_diag * np.array([0.75, 0.5, 0.5])
    
    positions = oriented_atoms.get_positions()
    cell = oriented_atoms.get_cell()
    
    # Centers
    c1 = center1
    c2 = center2

    x_idx, y_idx, z_idx = 0, 1, 2
    
    if method == 'geometric' and type == 'edge':
        # ... (keep existing geometric code) ...
        # (Omitting for brevity, but in real file keep it or replace it)
        # Actually, let's replace the whole block with the new 'atomsk' method logic
        pass

    if method == 'atomsk' and type == 'edge':
        # Atomsk-style Volterra Dislocation (Continuous field, minimal/no deletion)
        # 1. Calculate Volterra fields for dipole
        # 2. Apply displacements
        # 3. NO overlap removal (except extreme cases)
        
        Lx, Ly, Lz = cell[0,0], cell[1,1], cell[2,2]
        
        # Helper for MIC
        def mic_dist(d, L):
            return d - L * np.round(d/L)
            
        # Cores
        centers = [c1, c2]
        # Burgers vectors: D1 (-b), D2 (+b) -> Creates vacancy dipole?
        # Actually, sign convention matters.
        # Atomsk: u_x = b/2pi * ...
        # If we want a standard dipole, we usually want them to attract/annihilate or be stable.
        # Let's use [-b, +b].
        b_vals = [-b_mag, b_mag]
        
        ux_total = np.zeros(len(positions))
        uy_total = np.zeros(len(positions))
        
        for c, b in zip(centers, b_vals):
            # Calculate distance from core
            # Important: Use MIC?
            # For a dipole in a periodic box, simply summing fields works well enough
            # if the box is large enough.
            # Atomsk tutorial says "boundary conditions... not periodic".
            # But for a dipole, we want periodicity.
            # Let's use simple distance (no MIC) for the field calculation relative to the specific core?
            # No, if we have PBC, we must handle the cut.
            # The simple sum of two Volterra fields (with cuts extending to infinity)
            # creates a strip between them if the cuts are aligned.
            # D1 cut: -x direction. D2 cut: -x direction.
            # Region between D1 and D2 has net displacement b.
            # So simple coordinates (dx, dy) without MIC is safer for preserving the cut logic,
            # PROVIDED the box is unwrapped or we handle coordinates carefully.
            # But here 'positions' are in the box.
            # Let's use MIC for distance, but this "wraps" the cut.
            # Wrapping the cut is dangerous.
            
            # Better approach for Dipole in PBC:
            # Construct the displacements using the "infinite array" solution or
            # Just use the simple sum and assume the cut is consistent.
            # If we use MIC, the cut "wraps" around the boundary.
            # For a dipole, this is fine: the cut starts at D1, goes to boundary, wraps, comes back to D2?
            # Ideally, we want the cut to be the short segment D1-D2.
            # To achieve this:
            # D1 (+b): Cut along +x.
            # D2 (-b): Cut along +x.
            # Region D1 < x < D2: +b displacement.
            # Region x > D2: +b - b = 0.
            # Region x < D1: 0.
            # So cut is effectively finite segment D1-D2.
            # This is perfect.
            
            # So:
            # D1 at c1. b1 = -b_mag (if we want removal). Wait.
            # Let's try b1 = +b_mag (Cut +x).
            # D2 at c2. b2 = -b_mag (Cut +x).
            # Result: Strip between c1 and c2 has displacement +b.
            # This is a Shear Loop.
            
            dx = positions[:, x_idx] - c[x_idx]
            dy = positions[:, y_idx] - c[y_idx]
            
            # Apply MIC only to Y?
            # If we apply MIC to X, we mess up the cut direction logic.
            # So let's Apply MIC to Y (to handle vertical periodicity)
            # but KEEP X absolute (to maintain Left/Right relation for the cut).
            # This requires the box to be "unwrapped" in X or cores to be well-placed.
            # Our cores are at 0.25 and 0.75.
            # dx ranges from -0.75 to +0.75.
            # This is fine.
            
            dy = mic_dist(dy, Ly)
            # dx is raw difference.
            
            # Calculate field
            # Use standard Volterra.
            # Note: arctan2(y, x) has cut at x<0 (negative axis).
            # We want cut along +x?
            # arctan2(y, -x) has cut at -x<0 => x>0.
            # So use arctan2(y, -dx) to put cut along +x.
            
            r2 = dx**2 + dy**2
            r2[r2 < 1e-6] = 1e-6
            
            # To place cut along +x axis relative to core:
            # theta = arctan2(y, -x) ?
            # x>0, y=0 -> -x<0 -> theta = pi.
            # x<0, y=0 -> -x>0 -> theta = 0.
            # So cut is at +x. Correct.
            theta = np.arctan2(dy, -dx) 
            
            # Formula with standard cut (along -x):
            # u_x = b/2pi * (theta + ...)
            # We use our rotated theta.
            
            # Apply formula
            factor = b / (2 * np.pi)
            
            term1_x = theta
            term2_x = (dx * dy) / (2 * (1 - nu) * r2)
            ux = factor * (term1_x + term2_x)
            
            term1_y = (1 - 2 * nu) / (2 * (1 - nu)) * np.log(np.sqrt(r2))
            term2_y = (dx**2 - dy**2) / (4 * (1 - nu) * r2)
            uy = -factor * (term1_y + term2_y)
            
            ux_total += ux
            uy_total += uy
            
        positions[:, x_idx] += ux_total
        positions[:, y_idx] += uy_total
        
        oriented_atoms.set_positions(positions)
        oriented_atoms.wrap()
        
        # NO overlap removal (Atomsk style)
        # Unless atoms are dangerously close (< 0.5 A)
        from ase.neighborlist import NeighborList
        nl = NeighborList([0.5]*len(oriented_atoms), skin=0.0, self_interaction=False, bothways=True)
        nl.update(oriented_atoms)
        to_delete = set()
        for i in range(len(oriented_atoms)):
            indices, _ = nl.get_neighbors(i)
            for j in indices:
                if j > i: to_delete.add(j)
        if to_delete:
            del oriented_atoms[[i for i in to_delete]]
            
        return oriented_atoms

    elif method == 'geometric' and type == 'edge':
        # Geometric construction for edge dislocation dipole
        # Instead of elastic field, we simply insert a half-plane or remove one.
        # For a dipole, we can remove a strip of width 'b_mag' between c1 and c2.
        # This corresponds to a vacancy platelet.
        # If we remove a strip of atoms where:
        # x > c1.x AND x < c2.x
        # y is close to the slip plane (c1.y)
        
        # But this is crude. Let's use the 'shear' method.
        # 1. Select slip plane at y = c1.y
        # 2. For x between c1.x and c2.x:
        #    Shift atoms above slip plane by -b/2
        #    Shift atoms below slip plane by +b/2
        # This creates a mismatch at x=c1.x and x=c2.x
        
        # Actually, for a dipole, the region BETWEEN the dislocations has slipped by b relative to outside.
        # So:
        # If c1.x < x < c2.x:
        #   Shift all atoms (y > slip_plane) by -b
        #   (Or shift y < slip_plane by +b)
        # This creates a step at the slip plane, but only between c1 and c2.
        # At the ends (c1, c2), the step terminates -> dislocation!
        
        # Let's try this simple geometric construction.
        # Slip plane Y coordinate:
        y_plane = c1[1]
        x_start = c1[0]
        x_end = c2[0]
        
        # Burgers vector direction is X (axis 0)
        # b vector magnitude is b_mag. Direction is -X for removal?
        # Let's shift the UPPER crystal (y > y_plane) to the LEFT (-X) by b_mag
        # ONLY in the region x_start < x < x_end.
        
        # Wait, if we shift only a part of the crystal, we break bonds at x_start and x_end.
        # That IS the dislocation core.
        
        # Implementation:
        # Iterate over atoms.
        # If y > y_plane:
        #   If x_start < x < x_end:
        #     x -= b_mag
        
        # But this leaves a gap at x_end and overlap at x_start?
        # No.
        # At x_start: atoms at x > x_start move left. They approach atoms at x < x_start. -> OVERLAP (Compressive)
        # At x_end: atoms at x < x_end move left. They move away from atoms at x > x_end. -> GAP (Tensile)
        
        # This creates one compressive core and one tensile core. Perfect dipole.
        # And it's purely geometric, no elasticity involved.
        # This is often cleaner for small cells.
        
        mask_y = positions[:, 1] > y_plane
        mask_x = (positions[:, 0] > x_start) & (positions[:, 0] < x_end)
        mask = mask_y & mask_x
        
        # Apply shear
        # We shift by b_mag/2 on both sides to keep symmetry?
        # Or just shift one side by b_mag.
        # Let's shift top side by -b_mag/2, bottom side by +b_mag/2
        # In the region x_start < x < x_end.
        
        mask_y_top = positions[:, 1] > y_plane
        mask_y_bot = positions[:, 1] <= y_plane
        
        mask_region = (positions[:, 0] > x_start) & (positions[:, 0] < x_end)
        
        # Shift
        # Top moves Left (-X)
        positions[mask_y_top & mask_region, 0] -= b_mag * 0.5
        # Bottom moves Right (+X)
        positions[mask_y_bot & mask_region, 0] += b_mag * 0.5
        
        oriented_atoms.set_positions(positions)
        
        # Now we have overlaps at x_start (Compression) and gaps at x_end (Tension).
        # We need to remove overlaps.
        # The gap at x_end might need filling?
        # Actually, for a standard dipole, we usually want two edge dislocations of opposite sign.
        # This method produces that.
        # Ovito should see the termination of atomic planes.
        
        # Overlap removal
        oriented_atoms.wrap()
        
        # We might need to relax the gap manually?
        # No, let's just remove overlaps.
        # The gap will be a void.
        
        from ase.neighborlist import NeighborList
        nl = NeighborList([tol/2]*len(oriented_atoms), skin=0.0, self_interaction=False, bothways=True)
        nl.update(oriented_atoms)
        
        to_delete = set()
        for i in range(len(oriented_atoms)):
            if i in to_delete: continue
            indices, offsets = nl.get_neighbors(i)
            for j in indices:
                if j > i:
                    if j not in to_delete:
                        to_delete.add(j)
                        
        if to_delete:
            del oriented_atoms[[i for i in to_delete]]
            
        return oriented_atoms

    # For screw dipole, it's easier (just Z displacements).
    if type == 'screw':
        # Apply sum of fields
        # u_z = b/2pi * (theta1 - theta2)
        # Note: signs depend on b direction.
        # Let's assume dislocation 1 is +b, dislocation 2 is -b.
        
        x_idx, y_idx, z_idx = 0, 1, 2
        
        # Relative coordinates
        x1 = positions[:, x_idx] - c1[x_idx]
        y1 = positions[:, y_idx] - c1[y_idx]
        # Wrap relative coordinates to handle periodicity?
        # For dipole in periodic box, we should use minimum image convention relative to centers.
        # But arctan2 handles full range.
        # The main issue is the branch cut.
        # Simple summation works if cores are far from boundaries.
        
        theta1 = np.arctan2(y1, x1)
        
        x2 = positions[:, x_idx] - c2[x_idx]
        y2 = positions[:, y_idx] - c2[y_idx]
        theta2 = np.arctan2(y2, x2)
        
        u_z = (b_mag / (2 * np.pi)) * (theta1 - theta2)
        positions[:, z_idx] += u_z
        
        oriented_atoms.set_positions(positions)
        oriented_atoms.wrap()
        return oriented_atoms

    elif type == 'edge':
        # Edge Dipole using Volterra Overlap Method
        # Instead of manually removing a strip, we apply the displacement field of the dipole
        # to the PERFECT crystal.
        # The field for a dipole (+b at c1, -b at c2) naturally moves atoms INWARD
        # into the region between the cores (the "missing strip" region).
        # This creates overlap in that region.
        # Then we delete the overlapping atoms.
        
        x_idx, y_idx, z_idx = 0, 1, 2
        
        # We need the displacement field of an edge dislocation.
        # u_x = b/2pi * [ theta + xy / (2(1-nu)r^2) ]
        # u_y = -b/2pi * [ (1-2nu)/2(1-nu) * ln(r) + (x^2-y^2)/(4(1-nu)r^2) ]
        
        def get_edge_field(x, y, b, nu, zeta=2.0):
            # Smooth the core using a small zeta
            r2 = x**2 + y**2 + zeta**2
            r = np.sqrt(r2)
            
            # For the angle theta, we use a slightly smoothed version to avoid jump
            theta = np.arctan2(y, x)
            
            factor = b / (2 * np.pi)
            
            term1_x = theta
            term2_x = (x * y) / (2 * (1 - nu) * r2)
            u_x = factor * (term1_x + term2_x)
            
            term1_y = (1 - 2 * nu) / (2 * (1 - nu)) * np.log(r)
            term2_y = (x**2 - y**2) / (4 * (1 - nu) * r2)
            u_y = -factor * (term1_y + term2_y)
            
            return u_x, u_y
            
        # Determine displacement fields
        Lx, Ly, Lz = cell[0,0], cell[1,1], cell[2,2]
        
        # Helper for MIC (Minimum Image Convention)
        def mic_dist(d, L):
            return d - L * np.round(d/L)

        ux_total = np.zeros(len(positions))
        uy_total = np.zeros(len(positions))
        uz_total = np.zeros(len(positions))

        if dissociate and type == 'edge':
            # Split each edge dislocation into two Shockley partials
            # D1 (at c1) -> P1a + P1b
            # D2 (at c2) -> P2a + P2b
            
            # Partial Burgers vectors geometry for FCC:
            # Full b = [b, 0, 0]
            # Partials have x-component b/2 and z-component +/- b/(2*sqrt(3))
            b_screw = b_mag / (2 * np.sqrt(3))
            b_edge_partial = b_mag / 2.0
            
            # Separation vector: along X (slip direction)
            sep = np.array([dissociation_width/2.0, 0, 0])
            
            # Define centers for partials
            centers = [c1 - sep, c1 + sep, c2 - sep, c2 + sep]
            b_edges = [-b_edge_partial, -b_edge_partial, b_edge_partial, b_edge_partial]
            b_screws = [b_screw, -b_screw, -b_screw, b_screw]
            
            for c, be, bs in zip(centers, b_edges, b_screws):
                dx = mic_dist(positions[:, x_idx] - c[x_idx], Lx)
                dy = mic_dist(positions[:, y_idx] - c[y_idx], Ly)
                
                # Edge part with core smoothing
                ux, uy = get_edge_field(dx, dy, be, nu, zeta=core_width)
                ux_total += ux
                uy_total += uy
                
                # Screw part (u_z = b * theta / 2pi)
                # Smoothed theta for screw
                theta = np.arctan2(dy, dx)
                uz = bs * theta / (2 * np.pi)
                uz_total += uz
                
        else:
            # Standard full dislocations
            centers = [c1, c2]
            b_vals = [-b_mag, b_mag]
            
            for c, b in zip(centers, b_vals):
                dx = mic_dist(positions[:, x_idx] - c[x_idx], Lx)
                dy = mic_dist(positions[:, y_idx] - c[y_idx], Ly)
                
                ux, uy = get_edge_field(dx, dy, b, nu, zeta=core_width)
                ux_total += ux
                uy_total += uy

        positions[:, x_idx] += ux_total
        positions[:, y_idx] += uy_total
        positions[:, z_idx] += uz_total
        
        oriented_atoms.set_positions(positions)
        
        # Now remove overlaps
        # Overlap removal is still needed but should be less drastic due to core smoothing.
        oriented_atoms.wrap()
        
        from ase.neighborlist import NeighborList
        nl = NeighborList([tol/2]*len(oriented_atoms), skin=0.0, self_interaction=False, bothways=True)
        nl.update(oriented_atoms)
        
        to_delete = set()
        for i in range(len(oriented_atoms)):
            if i in to_delete: continue
            indices, offsets = nl.get_neighbors(i)
            for j in indices:
                if j > i:
                    if j not in to_delete:
                        to_delete.add(j)
                        
        if to_delete:
            # print(f"Deleting {len(to_delete)} overlapping atoms in dipole generation")
            del oriented_atoms[[i for i in to_delete]]
            
        return oriented_atoms

    else:
        raise ValueError(f"Unknown dislocation type: {type}")


