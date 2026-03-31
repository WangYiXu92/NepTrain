import numpy as np
from ase import Atoms
from ase.build import make_supercell, surface
from NepTrain.core.perturb import csl_core

class InterfaceOptimizer:
    """
    Optimizes the interface between two grains by finding the best Rigid Body Translation (RBT)
    and vertical separation to maximize physical realism.
    """
    def __init__(self, grain_a: Atoms, grain_b: Atoms):
        self.grain_a = grain_a.copy()
        self.grain_b = grain_b.copy()

    def optimize(self, grid_size=8, target_dist=2.0, vacuum=0.0, translation=None, center_at_half_z=True):
        """
        Performs the optimization.
        
        Args:
            grid_size: Number of grid points for RBT search (NxN).
            target_dist: Target minimum distance between atoms at interface.
            vacuum: Additional vacuum to insert.
            translation: Optional fixed translation vector (Cartesian [x, y] or [x, y, z]) to apply to Grain B.
                        If provided, grid search is skipped.
            center_at_half_z: If True, adjust the final Z coordinates so the interface is at the center of the box.
            
        Returns:
            Merged Atoms object.
        """
        # 1. Setup Surface Atoms for fast calculation
        z_a = self.grain_a.positions[:, 2]
        max_z_a = np.max(z_a)
        min_z_a = np.min(z_a)
        
        # Initial vertical alignment: B starts just above A
        z_b = self.grain_b.positions[:, 2]
        min_z_b = np.min(z_b)
        
        # Shift B so its bottom is at 0 (relative)
        self.grain_b.positions[:, 2] -= min_z_b
        
        # Select interaction region (atoms within 4A of interface)
        cutoff = 4.0
        indices_a = np.where(z_a > max_z_a - cutoff)[0]
        indices_b = np.where(self.grain_b.positions[:, 2] < cutoff)[0]
        
        if len(indices_a) == 0 or len(indices_b) == 0:
            return self._stack_simple(target_dist, vacuum, center_at_half_z=center_at_half_z)

        # Get positions
        pos_a = self.grain_a.positions[indices_a]
        pos_b = self.grain_b.positions[indices_b] 
        
        # Create periodic images of A (3x3 grid) to ensure correct PBC checks
        cell = self.grain_a.get_cell()
        a_vec = cell[0]
        b_vec = cell[1]
        
        images_pos_a = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                shift = i * a_vec + j * b_vec
                img = pos_a.copy()
                img += shift
                images_pos_a.append(img)
        
        pos_a_expanded = np.vstack(images_pos_a)
        
        # 2. RBT Search or Fixed Translation
        best_shift = np.zeros(2)
        min_required_z = 1e9 
        
        pos_a_xy = pos_a_expanded[:, :2]
        pos_b_xy = pos_b[:, :2]
        
        target2 = target_dist**2
        
        if translation is not None:
            # Use fixed translation
            trans_arr = np.array(translation)
            shift_vec = trans_arr[:2] # Ignore Z component of translation for now, we optimize Z
            
            # Check Z requirement for this shift
            shift_xy = shift_vec
            
            diff = pos_a_xy[:, np.newaxis, :] - (pos_b_xy[np.newaxis, :, :] + shift_xy)
            d2_xy = np.sum(diff**2, axis=2) 
            
            term_sq = target2 - d2_xy
            mask = term_sq > 0
            
            if not np.any(mask):
                current_max_z_a = np.max(pos_a_expanded[:, 2])
                min_required_z = current_max_z_a - 2.0
            else:
                required_dz_pair = np.sqrt(term_sq[mask])
                z_diff = pos_a_expanded[:, 2:3] - pos_b[:, 2:3].T
                relevant_z_diff = z_diff[mask]
                constraints = relevant_z_diff + required_dz_pair
                min_required_z = np.max(constraints)
            
            best_shift = shift_vec
            
        else:
            # Grid Search
            u_vals = np.linspace(0, 1, grid_size, endpoint=False)
            v_vals = np.linspace(0, 1, grid_size, endpoint=False)
            
            for u in u_vals:
                for v in v_vals:
                    shift_vec = u * a_vec + v * b_vec
                    shift_xy = shift_vec[:2]
                    
                    # Vectorized distance calculation
                    diff = pos_a_xy[:, np.newaxis, :] - (pos_b_xy[np.newaxis, :, :] + shift_xy)
                    d2_xy = np.sum(diff**2, axis=2) 
                    
                    term_sq = target2 - d2_xy
                    mask = term_sq > 0
                    
                    if not np.any(mask):
                        # No atoms laterally overlapping within target_dist.
                        # We can potentially interpenetrate, but bounded by cutoff.
                        # We limit the shift to avoid smashing into unchecked atoms.
                        # z_diff approx max_z_a. We allow max 2.0A overlap.
                        # max_z_a is roughly max(pos_a_expanded[:, 2]).
                        # min_z_b is 0.
                        # So shift_z should be >= max_z_a - 2.0
                        current_max_z_a = np.max(pos_a_expanded[:, 2])
                        req_dz = current_max_z_a - 2.0
                    else:
                        required_dz_pair = np.sqrt(term_sq[mask])
                        z_diff = pos_a_expanded[:, 2:3] - pos_b[:, 2:3].T
                        relevant_z_diff = z_diff[mask]
                        constraints = relevant_z_diff + required_dz_pair
                        req_dz = np.max(constraints)
                    
                    if req_dz < min_required_z:
                        min_required_z = req_dz
                        best_shift = shift_vec

        # 3. Apply Best Shift
        # Ensure best_shift is 3D
        if len(best_shift) == 2:
            shift_3d = np.array([best_shift[0], best_shift[1], 0.0])
        else:
            shift_3d = np.array(best_shift)
            shift_3d[2] = 0.0 # Only apply XY shift here, Z is handled separately
            
        self.grain_b.translate(shift_3d)
        self.grain_b.translate([0, 0, min_required_z])
        
        if vacuum > 0:
            self.grain_b.translate([0, 0, vacuum])
            
        return self._merge(center_at_half_z=center_at_half_z)

    def _stack_simple(self, dist, vacuum, center_at_half_z=True):
        z_a = np.max(self.grain_a.positions[:, 2])
        z_b_min = np.min(self.grain_b.positions[:, 2])
        shift = z_a - z_b_min + dist + vacuum
        self.grain_b.translate([0, 0, shift])
        return self._merge(center_at_half_z=center_at_half_z)

    def _merge(self, center_at_half_z=True):
        """
        Merge grains and optionally center interface.
        """
        combined = self.grain_a + self.grain_b
        
        # Current Z range
        pos = combined.get_positions()
        z_coords = pos[:, 2]
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        z_range = z_max - z_min
        
        # Interface Z is roughly where Grain A ends and Grain B begins
        # Since we stacked B on top of A, interface is around max_z_a of grain A
        # Recalculate max_z_a because grain_a might have been modified? 
        # No, self.grain_a is not modified by translate(shift_3d) which is on self.grain_b.
        # But grain_a was centered in Z before.
        
        max_z_a = np.max(self.grain_a.positions[:, 2])
        min_z_b = np.min(self.grain_b.positions[:, 2])
        interface_z = (max_z_a + min_z_b) / 2.0
        
        # Set new cell
        new_cell = self.grain_a.get_cell()
        
        # For slab model with vacuum, we typically add 10A vacuum
        # But user might want specific box size.
        # Here we just ensure enough vacuum.
        vacuum_padding = 10.0
        box_z = z_range + vacuum_padding
        new_cell[2, 2] = box_z
        combined.set_cell(new_cell)
        
        if center_at_half_z:
            # Shift atoms so interface_z aligns with box_z / 2
            target_z = box_z / 2.0
            shift_z = target_z - interface_z
            
            # Need to update positions manually or use translate
            # translate updates positions but not cell
            combined.translate([0, 0, shift_z])
            combined.wrap()
            
        combined.pbc = [True, True, False] # Typically slab is non-periodic in Z
        return combined


class InterfaceBuilder:
    """
    Constructs Grain Boundaries and Twin boundaries using CSL theory.
    Uses pure ASE/NumPy logic without pymatgen.
    """
    
    @staticmethod
    def ensure_conventional_cubic(atoms: Atoms) -> Atoms:
        """
        Checks if the input atoms object is a primitive FCC or BCC cell
        and converts it to the conventional cubic cell if necessary.
        This is important because CSL generation assumes cubic axes.
        
        Args:
            atoms: Input structure.
            
        Returns:
            Atoms object (converted or original).
        """
        cell = atoms.get_cell()
        angles = cell.angles()
        
        # If already orthogonal (90 degrees), assume it's fine
        if np.allclose(angles, 90.0):
            return atoms
            
        # Check for Primitive FCC (1 atom, 60 degree angles)
        # Note: ASE bulk('Al', 'fcc') gives 60 degree angles
        if len(atoms) == 1 and np.allclose(angles, 60.0):
            print("Detected Primitive FCC cell. Converting to Conventional Cubic...")
            # Transformation: new_v1 = -v1+v2+v3, etc.
            M = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
            return make_supercell(atoms, M)
            
        # Check for Primitive BCC (1 atom, ~109.47 degree angles)
        if len(atoms) == 1 and np.isclose(angles[0], 109.47122, atol=0.1):
            print("Detected Primitive BCC cell. Converting to Conventional Cubic...")
            # Transformation: new_v1 = v2+v3, etc.
            M = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
            return make_supercell(atoms, M)
            
        # If not detected as simple primitive, return as is but warn
        print("Warning: Input cell is not orthogonal (cubic). CSL generation might be incorrect if indices are for cubic.")
        return atoms

    @staticmethod
    def build_oriented_supercell(atoms: Atoms, u=[1,0,0], v=[0,1,0], w=[0,0,1], supercell=[1,1,1]):
        """
        Builds a supercell oriented along specified crystal directions.
        This is similar to AtomSK's '--orient' mode.
        
        Args:
            atoms: Input primitive cell
            u, v, w: Integer Miller indices for new X, Y, Z axes
            supercell: Multipliers for the new cell [nx, ny, nz]
            
        Returns:
            Atoms object of the oriented supercell.
        """
        # 1. Validate inputs are integers
        u = np.array(u, dtype=int)
        v = np.array(v, dtype=int)
        w = np.array(w, dtype=int)
        
        # 2. Construct Transformation Matrix M
        # M columns are the new lattice vectors expressed in terms of old lattice vectors
        # For cubic, if u=[1,1,0], then new_a = 1*old_a + 1*old_b + 0*old_c
        # So M = [u, v, w].T
        
        M = np.vstack([u, v, w]).T
        
        # Check if M is singular (vectors must be linearly independent)
        if np.isclose(np.linalg.det(M), 0):
            raise ValueError(f"Directions {u}, {v}, {w} are not linearly independent.")
            
        # 3. Create Supercell
        # We use make_supercell with the general matrix M
        # ASE's make_supercell P matrix is such that new_cell = P @ old_cell? 
        # No, ASE uses P such that new_cell vectors are rows of P @ old_cell.
        # Wait, make_supercell(atoms, P) -> new cell is P @ atoms.cell
        # Our M has u, v, w as columns. So M.T has them as rows.
        # So P = M.T
        
        P = M.T
        oriented_cell = make_supercell(atoms, P)
        
        # 4. Enlarge if requested
        if np.any(np.array(supercell) > 1):
            oriented_cell = make_supercell(oriented_cell, np.diag(supercell))
            
        return oriented_cell

    @staticmethod
    def build_twist_boundary(atoms: Atoms, sigma: int, axis=[0,0,1], target_dist=1.8, translation=None, translation_frac=None, supercell_factor=2, preserve_cell=False):
        """
        Builds a Twist Grain Boundary.
        
        Args:
            atoms: Input structure
            sigma: Sigma value for CSL
            axis: Rotation axis
            target_dist: Minimum distance at interface
            translation: Optional Cartesian translation
            translation_frac: Optional fractional translation [u, v]
            supercell_factor: Factor to enlarge the GB plane (default 2x2)
            preserve_cell: If True, preserve original cell dimensions (default False)
        """
        # Store original cell if needed
        original_cell = atoms.get_cell()
        
        # 1. Get Rotation Angle
        angle = csl_core.get_csl_angle(sigma, axis)
        if angle is None:
            raise ValueError(f"No CSL angle found for sigma={sigma}, axis={axis}")
            
        # 2. Find CSL Basis Vectors
        M = csl_core.find_csl_vectors(sigma, axis, angle)
        if M is None:
            raise ValueError(f"Could not find CSL vectors for sigma={sigma}")
            
        # 3. Create Grain A (Supercell)
        grain_a = make_supercell(atoms, M)
        
        # 4. Enlarge in-plane for more reasonable structure
        if supercell_factor > 1:
            scale_matrix = np.diag([supercell_factor, supercell_factor, 1])
            grain_a = make_supercell(grain_a, scale_matrix)
        
        # 5. Create Grain B (Rotated)
        grain_b = grain_a.copy()
        
        # Rotate atoms around the axis
        grain_b.rotate(angle, axis, center=(0,0,0), rotate_cell=False)
        grain_b.wrap()
        
        # 6. Cleanup Grain B: Remove overlapping atoms
        from ase.geometry import get_distances
        tol = 0.1
        
        pos_b = grain_b.get_positions()
        
        keep_mask = np.ones(len(grain_b), dtype=bool)
        dists = grain_b.get_all_distances(mic=True)
        np.fill_diagonal(dists, 10.0)
        
        pairs = np.argwhere(dists < tol)
        
        for i, j in pairs:
            if keep_mask[i] and keep_mask[j]:
                if i < j:
                    keep_mask[j] = False
                    
        if np.sum(keep_mask) < len(grain_b):
            grain_b = grain_b[keep_mask]
        
        # 7. Orient along Z axis
        axis_vec = np.array(axis, dtype=float)
        axis_vec /= np.linalg.norm(axis_vec)
        
        if not np.allclose(axis_vec, [0,0,1]):
            grain_a.rotate(axis, 'z', rotate_cell=True)
            grain_b.rotate(axis, 'z', rotate_cell=True)
            
        # Center in Z to ensure good overlap check
        grain_a.center(axis=2)
        grain_b.center(axis=2)
        
        # Calculate Cartesian translation if fractional provided
        if translation is None and translation_frac is not None:
            cell = grain_a.get_cell()
            v1 = cell[0]
            v2 = cell[1]
            u, v = translation_frac[0], translation_frac[1]
            translation = u * v1 + v * v2
        
        optimizer = InterfaceOptimizer(grain_a, grain_b)
        combined = optimizer.optimize(target_dist=target_dist, vacuum=10.0, translation=translation, center_at_half_z=True)
        
        # 8. Preserve original cell if requested
        if preserve_cell:
            orig_cell = original_cell
            new_cell = combined.get_cell()
            
            # Calculate required Z size for the atoms
            z_positions = combined.get_positions()
            z_range = np.max(z_positions[:, 2]) - np.min(z_positions[:, 2])
            z_margin = 10.0
            
            # Set new cell: keep XY from original, Z just fits atoms + margin
            new_cell[0] = orig_cell[0]
            new_cell[1] = orig_cell[1]
            new_cell[2, 2] = z_range + z_margin
            
            combined.set_cell(new_cell)
            combined.wrap()
        
        return combined

    @staticmethod
    def build_mirror_twin(atoms: Atoms, plane=[1,1,1], target_dist=1.8, translation=None, translation_frac=None, layers=10, preserve_cell=False):
        """
        Builds a Twin boundary using the Mirror Method.
        
        Args:
            atoms: Input structure (must be bulk)
            plane: Miller indices for twin plane (e.g., [1,1,1])
            target_dist: Minimum distance at interface
            translation: Optional Cartesian translation
            translation_frac: Optional fractional translation [u, v]
            layers: Number of layers for each grain (thickness)
            preserve_cell: If True, preserve original cell dimensions (default False)
        """
        # Store original cell if needed
        original_cell = atoms.get_cell()
        
        # 1. Create Oriented Slab
        indices = tuple(np.array(plane, dtype=int))
        
        # Use surface to get the correct orientation
        grain_a = surface(atoms, indices, layers=layers, vacuum=0.0)
        
        # Center the slab in XY plane but keep Z orientation
        grain_a.center(axis=2)
        
        # Get cell parameters
        cell = grain_a.get_cell()
        a_vec = cell[0]
        b_vec = cell[1]
        c_vec = cell[2]
        
        # 2. Create Twin (Grain B) by proper mirroring
        grain_b = grain_a.copy()
        
        # Get positions and find the interface plane
        pos_a = grain_a.get_positions()
        z_coords = pos_a[:, 2]
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        z_mid = (z_min + z_max) / 2
        
        # Reflect across the twin plane
        pos_b = grain_b.get_positions()
        pos_b[:, 2] = 2 * z_mid - pos_b[:, 2]
        grain_b.set_positions(pos_b)
        
        # Wrap atoms back into the cell
        grain_b.wrap()
        
        # Shift grain B up to create proper separation
        grain_b.translate([0, 0, target_dist])
        
        # Calculate Cartesian translation if fractional provided
        if translation is None and translation_frac is not None:
            u, v = translation_frac[0], translation_frac[1]
            translation = u * a_vec + v * b_vec
            translation = np.array([translation[0], translation[1], 0.0])
        
        # 3. Optimize interface
        optimizer = InterfaceOptimizer(grain_a, grain_b)
        combined = optimizer.optimize(target_dist=target_dist, vacuum=10.0, translation=translation, center_at_half_z=True)
        
        # 4. Preserve original cell if requested
        if preserve_cell:
            # Scale the combined structure to match original cell in XY, keep Z reasonable
            orig_cell = original_cell
            new_cell = combined.get_cell()
            
            # Calculate required Z size for the atoms
            z_positions = combined.get_positions()
            z_range = np.max(z_positions[:, 2]) - np.min(z_positions[:, 2])
            z_margin = 10.0  # vacuum margin
            
            # Set new cell: keep XY from original, Z just fits atoms + margin
            new_cell[0] = orig_cell[0]
            new_cell[1] = orig_cell[1]
            new_cell[2, 2] = z_range + z_margin
            
            combined.set_cell(new_cell)
            # Re-wrap atoms to new cell
            combined.wrap()
        
        return combined
