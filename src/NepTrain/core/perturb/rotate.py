
import numpy as np
from ase import Atoms
from ase.neighborlist import natural_cutoffs, NeighborList
from scipy import sparse
from scipy.spatial.transform import Rotation

def get_molecules(atoms: Atoms, mult: float = 1.2):
    """
    Identify molecules in an Atoms object based on bond connectivity.
    
    Args:
        atoms: ASE Atoms object
        mult: Multiplier for natural cutoffs to define bonds
        
    Returns:
        list of lists: Each inner list contains indices of atoms in a molecule
    """
    cutoffs = natural_cutoffs(atoms, mult=mult)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    adj = nl.get_connectivity_matrix()
    n_components, component_labels = sparse.csgraph.connected_components(adj, directed=False)
    
    molecules = []
    for i in range(n_components):
        indices = np.where(component_labels == i)[0]
        molecules.append(indices)
    return molecules

def parse_formula_dict(formula: str):
    """
    Parse chemical formula string into a dictionary of counts.
    Example: "H2O" -> {'H': 2, 'O': 1}
    """
    # Simple parser using ASE's internal or manual
    # ASE's Atoms(formula).symbols can be used to count
    temp_atoms = Atoms(formula)
    syms = temp_atoms.get_chemical_symbols()
    counts = {}
    for s in syms:
        counts[s] = counts.get(s, 0) + 1
    return counts

def rotate_fragments_by_formula(atoms: Atoms, formula: str, mult: float = 1.2, seed: int = None, rng_values: np.ndarray = None):
    """
    Identify fragments matching the given formula and rotate them randomly.
    
    Args:
        atoms: Input ASE Atoms object (will be copied)
        formula: Target chemical formula (e.g., "H2O", "CH4")
        mult: Multiplier for bond cutoffs
        seed: Random seed
        rng_values: (Optional) 1D numpy array of uniform random numbers [0, 1].
                    Need 3 values per fragment.
    
    Returns:
        Atoms: New Atoms object with rotated fragments
    """
    if seed is not None:
        np.random.seed(seed)
        
    new_atoms = atoms.copy()
    
    # Identify molecules
    molecules = get_molecules(new_atoms, mult=mult)
    
    # Target composition
    target_counts = parse_formula_dict(formula)
    
    # Identify fragments to rotate
    fragments_to_rotate = []
    for indices in molecules:
        # Check composition of this fragment
        fragment_syms = [new_atoms[i].symbol for i in indices]
        fragment_counts = {}
        for s in fragment_syms:
            fragment_counts[s] = fragment_counts.get(s, 0) + 1
            
        if fragment_counts == target_counts:
            fragments_to_rotate.append(indices)
            
    if not fragments_to_rotate:
        return new_atoms

    n_frags = len(fragments_to_rotate)
    
    if rng_values is not None:
        n_needed = 3 * n_frags
        if len(rng_values) < n_needed:
             raise ValueError(f"Not enough random values for rotation. Needed {n_needed}, got {len(rng_values)}.")
        
        # Reshape to (n_frags, 3)
        u = rng_values[:n_needed].reshape(n_frags, 3)
        
        # Uniform to Rotation Matrix (Shoemake)
        u1 = u[:, 0]
        u2 = u[:, 1]
        u3 = u[:, 2]
        
        r1 = np.sqrt(1 - u1)
        r2 = np.sqrt(u1)
        
        theta1 = 2 * np.pi * u2
        theta2 = 2 * np.pi * u3
        
        x = r1 * np.sin(theta1)
        y = r1 * np.cos(theta1)
        z = r2 * np.sin(theta2)
        w = r2 * np.cos(theta2)
        
        quats = np.stack([x, y, z, w], axis=1)
        matrices = Rotation.from_quat(quats).as_matrix()
        
    else:
        # Generate random rotations efficiently
        # Use scipy's Rotation.random which samples uniformly from SO(3) (Haar measure)
        # This ensures "equal sampling of the space"
        rots = Rotation.random(n_frags)
        matrices = rots.as_matrix()
    
    positions = new_atoms.get_positions()
    masses = new_atoms.get_masses()
    
    for i, indices in enumerate(fragments_to_rotate):
        fragment_pos = positions[indices]
        fragment_masses = masses[indices]
        
        # Calculate Center of Mass (mass-weighted)
        # Using mass-weighted COM ensures physical rotation and passes checks
        total_mass = np.sum(fragment_masses)
        if total_mass > 0:
            center_of_mass = np.average(fragment_pos, axis=0, weights=fragment_masses)
        else:
            center_of_mass = np.mean(fragment_pos, axis=0)
        
        # Rotate relative to COM
        centered_pos = fragment_pos - center_of_mass
        rotated_pos = centered_pos @ matrices[i].T
        new_pos = rotated_pos + center_of_mass
        
        positions[indices] = new_pos
            
    new_atoms.set_positions(positions)
    return new_atoms
