
import numpy as np
from ase import Atoms
from ase.neighborlist import natural_cutoffs, NeighborList
from scipy import sparse
from scipy.spatial.transform import Rotation

def parse_rigid_list_string(s):
    """
    Parse a string definition of rigid bodies.
    Format: "body1_indices;body2_indices;..."
    Indices can be comma-separated integers or ranges (start-end).
    Example: "0-3;4-7" -> [[0,1,2,3], [4,5,6,7]]
             "0,1,2;3,4,5" -> [[0,1,2], [3,4,5]]
    """
    if not s:
        return []
        
    bodies = []
    # Split by semicolon for distinct bodies
    parts = s.split(';')
    
    for part in parts:
        if not part.strip():
            continue
        indices = []
        # Split by comma for indices/ranges within a body
        subparts = part.split(',')
        for sub in subparts:
            sub = sub.strip()
            if not sub:
                continue
            if '-' in sub:
                try:
                    start, end = map(int, sub.split('-'))
                    indices.extend(range(start, end + 1))
                except ValueError:
                    print(f"Warning: Invalid range format '{sub}' in rigid list")
            else:
                try:
                    indices.append(int(sub))
                except ValueError:
                    print(f"Warning: Invalid index '{sub}' in rigid list")
        if indices:
            bodies.append(sorted(list(set(indices))))
            
    return bodies

class RigidBodyManager:
    """
    Manages rigid body definitions and perturbations for an Atoms object.
    """
    def __init__(self, atoms: Atoms):
        self.atoms = atoms
        # rigid_id: -1 for loose atoms, >=0 for rigid bodies
        if 'rigid_id' in atoms.arrays:
            self.ids = atoms.arrays['rigid_id']
        else:
            self.ids = np.full(len(atoms), -1, dtype=int)
            
    def detect_auto(self, mult=1.2):
        """Auto-detect molecules as rigid bodies based on connectivity."""
        cutoffs = natural_cutoffs(self.atoms, mult=mult)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(self.atoms)
        adj = nl.get_connectivity_matrix()
        n_components, component_labels = sparse.csgraph.connected_components(adj, directed=False)
        
        # Reset IDs
        self.ids[:] = -1
        
        # Assign IDs only to clusters with size > 1
        for i in range(n_components):
            indices = np.where(component_labels == i)[0]
            if len(indices) > 1:
                self.ids[indices] = i
                
        self.atoms.set_array('rigid_id', self.ids)
        return n_components

    def detect_by_composition(self, allowed_formulas, mult=1.2):
        """
        Detect rigid bodies by connectivity and filter by chemical composition.
        allowed_formulas: list of strings, e.g. ['H2O', 'CO2']
        Only clusters matching one of the allowed formulas (Hill notation) are marked rigid.
        """
        # Normalize allowed formulas to Hill notation
        normalized_allowed = set()
        for f in allowed_formulas:
            try:
                # Create dummy atoms to normalize formula string
                # Note: Atoms(str) works if ase version supports it, otherwise use Symbols
                # Safer: use ase.Symbols if available, or just try Atoms
                temp = Atoms(f)
                normalized_allowed.add(temp.get_chemical_formula(mode='hill'))
            except Exception:
                # Fallback: just add the raw string
                normalized_allowed.add(f)

        # 1. Detect all components
        cutoffs = natural_cutoffs(self.atoms, mult=mult)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(self.atoms)
        adj = nl.get_connectivity_matrix()
        n_components, component_labels = sparse.csgraph.connected_components(adj, directed=False)
        
        # Reset IDs
        self.ids[:] = -1
        current_body_id = 0
        
        for i in range(n_components):
            indices = np.where(component_labels == i)[0]
            if len(indices) <= 1:
                continue
                
            # Get formula of this cluster
            cluster = self.atoms[indices]
            # Get Hill formula
            cluster_formula = cluster.get_chemical_formula(mode='hill')
            
            if cluster_formula in normalized_allowed:
                self.ids[indices] = current_body_id
                current_body_id += 1
                
        self.atoms.set_array('rigid_id', self.ids)
        return current_body_id

    def set_manual(self, indices_list):
        """
        Manually set rigid bodies.
        indices_list: list of lists, e.g., [[0,1], [2,3,4]]
        """
        self.ids[:] = -1
        for i, indices in enumerate(indices_list):
            self.ids[indices] = i
        self.atoms.set_array('rigid_id', self.ids)

    def get_bodies(self):
        """Return dictionary of {body_id: [indices]} and list of loose atom indices."""
        unique_ids = np.unique(self.ids)
        bodies = {}
        loose = []
        for uid in unique_ids:
            indices = np.where(self.ids == uid)[0]
            if uid == -1:
                loose = indices
            else:
                bodies[uid] = indices
        return bodies, loose

    def perturb_inter(self, 
                      strain_tensor=None, 
                      disp_rng=None, 
                      rot_rng=None, 
                      min_distance=0.1,
                      cell_pert_fraction=0.03,
                      rotation_max_deg=15.0,
                      rng_values=None):
        """
        Inter-molecular perturbation:
        1. Deform cell.
        2. Move COMs affine to cell deformation.
        3. Add random displacement to COMs.
        4. Add random rotation to bodies (limited by rotation_max_deg).
        5. Loose atoms move affine + random displacement.
        
        rng_values: (Optional) 1D numpy array of uniform random numbers [0, 1].
                    Expected layout: 
                    [loose_atoms_disp (3*n_loose), 
                     body_1_com_disp (3), body_1_rot (3), 
                     body_2_com_disp (3), body_2_rot (3), ...]
        """
        atoms = self.atoms.copy()
        bodies, loose = self.get_bodies()
        
        # Pointer for rng_values
        rng_idx = 0
        
        # 1. Cell Deformation
        old_cell = atoms.get_cell()
        if strain_tensor is None:
            strains = np.random.uniform(-cell_pert_fraction, cell_pert_fraction, (3, 3))
            deformation = np.eye(3) + strains
        else:
            # strain_tensor assumed to be diagonal terms or full tensor?
            # If 3 elements -> diagonal. If 9 -> full.
            # Assuming consistency with run.py which passes 3 elements for diagonal in generate_strained_structure
            # But let's support general case.
            if strain_tensor.size == 3:
                 deformation = np.eye(3)
                 np.fill_diagonal(deformation, 1 + strain_tensor)
            else:
                 deformation = np.eye(3) + strain_tensor.reshape(3, 3)

        # New cell
        # ASE uses row vectors: v_new = v_old @ F (where F is deformation tensor applied to rows)
        # Actually usually F = I + epsilon. v_new = (I+eps) @ v_old.
        # Let's stick to simple scaling for now to match generate_strained_structure logic
        # which does: new_cell[i] = cell[i] * (1 + strains[i])
        
        # If we want full affine transform:
        # F = deformation
        # new_cell = old_cell @ F.T (if F acts on column vectors) or @ F (if F acts on rows)
        # Let's use: new_pos = pos @ F.T
        
        # Consistent with generate_deformed_structure in run.py (commented out one):
        # new_cell = np.dot(cell, deformation.T)
        
        # Consistent with generate_strained_structure (diagonal only):
        if strain_tensor is None or strain_tensor.size == 3:
             # Construct diagonal F
             strains = np.diag(deformation) - 1
             new_cell = np.zeros_like(old_cell)
             for i in range(3):
                 new_cell[i] = old_cell[i] * (1 + strains[i])
             
             # F matrix for positions (diagonal)
             F = np.diag(1 + strains)
        else:
             # Full tensor
             new_cell = old_cell @ deformation.T
             F = deformation

        atoms.set_cell(new_cell, scale_atoms=False) # We handle positions manually

        positions = atoms.get_positions()
        masses = atoms.get_masses()
        
        # Process Loose Atoms
        if len(loose) > 0:
            pos_loose = positions[loose]
            # Affine transform
            pos_loose = pos_loose @ F.T
            # Random displacement
            noise = np.zeros_like(pos_loose)
            
            if rng_values is not None:
                n_needed = 3 * len(loose)
                if len(rng_values) < rng_idx + n_needed:
                    raise ValueError(f"Not enough RNG values for loose atoms. Needed {n_needed} more.")
                
                vals = rng_values[rng_idx : rng_idx + n_needed].reshape(len(loose), 3)
                rng_idx += n_needed
                noise = vals * (2 * min_distance) - min_distance
            elif disp_rng is not None:
                # Use provided RNG (legacy)
                # Not fully implemented in legacy, fallback to random
                noise = np.random.uniform(-min_distance, min_distance, pos_loose.shape)
            else:
                noise = np.random.uniform(-min_distance, min_distance, pos_loose.shape)
            
            positions[loose] = pos_loose + noise

        # Process Rigid Bodies (Sorted by ID for determinism)
        sorted_body_ids = sorted(bodies.keys())
        
        for uid in sorted_body_ids:
            indices = bodies[uid]
            pos_body = positions[indices]
            mass_body = masses[indices]
            
            # Calculate COM
            com = np.average(pos_body, axis=0, weights=mass_body)
            
            # 1. Affine transform COM
            new_com = com @ F.T
            
            # 2. Add random displacement to COM
            com_noise = np.zeros(3)
            
            if rng_values is not None:
                if len(rng_values) < rng_idx + 3:
                    raise ValueError("Not enough RNG values for rigid body COM.")
                vals = rng_values[rng_idx : rng_idx + 3]
                rng_idx += 3
                com_noise = vals * (2 * min_distance) - min_distance
            else:
                com_noise = np.random.uniform(-min_distance, min_distance, 3)
                
            new_com += com_noise
            
            # 3. Rotate body
            centered_pos = pos_body - com
            
            # Generate rotation
            axis = np.array([0., 0., 1.])
            angle = 0.0
            
            if rng_values is not None:
                # Need 3 values: 2 for axis direction (uniform sphere), 1 for angle
                if len(rng_values) < rng_idx + 3:
                     raise ValueError("Not enough RNG values for rigid body rotation.")
                vals = rng_values[rng_idx : rng_idx + 3]
                rng_idx += 3
                
                # Uniform sphere sampling using 2 variables
                # z = 2*u - 1, theta = 2*pi*v
                u, v = vals[0], vals[1]
                z = 2 * u - 1
                r = np.sqrt(max(0, 1 - z*z))
                theta_ang = 2 * np.pi * v
                
                axis = np.array([
                    r * np.cos(theta_ang),
                    r * np.sin(theta_ang),
                    z
                ])
                
                # Angle: Map [0, 1] to [-max, max]
                angle = vals[2] * (2 * rotation_max_deg) - rotation_max_deg
                
            else:
                # Random axis
                axis = np.random.normal(size=3)
                axis /= np.linalg.norm(axis)
                # Random angle
                angle = np.random.uniform(-rotation_max_deg, rotation_max_deg)
            
            rot_matrix = Rotation.from_rotvec(axis * np.radians(angle)).as_matrix()
            rotated_pos = centered_pos @ rot_matrix.T
            
            # 4. Place at new COM
            positions[indices] = new_com + rotated_pos
            
        atoms.set_positions(positions)
        
        # Store rigid body info for plotting
        rigid_list = [indices.tolist() for indices in bodies.values()]
        atoms.info['rigid_bodies'] = rigid_list
        if 'perturb_annotation' not in atoms.info:
            atoms.info['perturb_annotation'] = {}
        atoms.info['perturb_annotation']['type'] = 'rigid'
        
        return atoms

    def perturb_intra(self, min_distance=0.1, rng_values=None):
        """
        Intra-molecular perturbation:
        Only perturb atoms WITHIN rigid bodies.
        Keep COM and Cell fixed? Or just restrict noise?
        User said: "perturbation only concerns the atoms within the rigid body"
        Interpretation: Add noise to atoms in bodies. Loose atoms untouched?
        
        rng_values: (Optional) 1D numpy array for Sobol sampling.
        """
        atoms = self.atoms.copy()
        positions = atoms.get_positions()
        bodies, loose = self.get_bodies()
        
        # Sort bodies for determinism
        sorted_uids = sorted(bodies.keys())
        
        rng_idx = 0
        
        for uid in sorted_uids:
            indices = bodies[uid]
            n_atoms = len(indices)
            
            if rng_values is not None:
                n_needed = 3 * n_atoms
                if len(rng_values) < rng_idx + n_needed:
                    raise ValueError(f"Not enough RNG values for intra perturbation. Needed {n_needed} more.")
                
                vals = rng_values[rng_idx : rng_idx + n_needed].reshape(n_atoms, 3)
                rng_idx += n_needed
                noise = vals * (2 * min_distance) - min_distance
            else:
                noise = np.random.uniform(-min_distance, min_distance, (n_atoms, 3))
                
            positions[indices] += noise
            
        atoms.set_positions(positions)
        
        # Store rigid body info for plotting
        rigid_list = [indices.tolist() for indices in bodies.values()]
        atoms.info['rigid_bodies'] = rigid_list
        if 'perturb_annotation' not in atoms.info:
            atoms.info['perturb_annotation'] = {}
        atoms.info['perturb_annotation']['type'] = 'rigid'
        
        return atoms

def generate_rigid_perturbed_structure(atoms: Atoms,
                                       mode='inter',
                                       strain_lim=[-0.03, 0.03],
                                       min_distance=0.1,
                                       strain_tensor=None,
                                       rng_values=None):
    """
    Wrapper for rigid body perturbation.
    """
    manager = RigidBodyManager(atoms)
    # Check if we need auto-detection (if no IDs set)
    # Assuming IDs are set before calling this, or we default to auto if all -1
    if np.all(manager.ids == -1):
        manager.detect_auto()
        
    if mode == 'inter':
        # Need to handle RNG values for Sobol consistency
        # Calculate random strain if not provided
        if strain_tensor is None:
             strains = np.random.uniform(strain_lim[0], strain_lim[1], 3)
        else:
             strains = strain_tensor
             
        return manager.perturb_inter(strain_tensor=strains, 
                                     min_distance=min_distance, 
                                     cell_pert_fraction=strain_lim[1],
                                     rng_values=rng_values) # approx
    
    elif mode == 'intra':
        return manager.perturb_intra(min_distance=min_distance, rng_values=rng_values)
        
    return atoms
