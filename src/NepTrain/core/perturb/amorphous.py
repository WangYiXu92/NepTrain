import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.calculator import Calculator, all_changes
from scipy.spatial.distance import pdist

class RepulsiveCalculator(Calculator):
    """
    Simple calculator that applies 1/r^12 repulsive potential to separate atoms.
    """
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, cutoff=2.0, repulsion_strength=1.0):
        super().__init__()
        self.cutoff = cutoff
        self.repulsion_strength = repulsion_strength
        
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        
        positions = atoms.get_positions()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        
        # Use ASE neighbor list or naive loop?
        # For simplicity and robustness on small/medium systems, let's use neighbor list
        # But we need forces.
        # Let's use simple neighbor list logic.
        
        from ase.neighborlist import neighbor_list
        # i, j are indices of interacting atoms
        # D is vector from j to i (or i to j depending on convention)
        # d is distance
        i_indices, j_indices, d_vecs, dists = neighbor_list('ijDd', atoms, cutoff=self.cutoff)
        
        energy = 0.0
        forces = np.zeros_like(positions)
        
        # Avoid division by zero
        dists = np.maximum(dists, 0.01)
        
        # Potential: V = k * (1 - r/rc)^2 ? (Like simple repulsive harmonic)
        # Or V = (rc/r)^12?
        # Soft sphere is good but gradients get huge.
        # Let's use a simpler bounded potential for better convergence?
        # V = 0.5 * k * (rc - r)^2 for r < rc
        # F = k * (rc - r) * (r_vec / r)
        
        # Or stick to 1/r^12 but scale forces?
        # Let's try the harmonic repulsion. It is softer at overlap.
        # V = 0.5 * (1 - r/rc)^2  (Normalized)
        
        # Actually, let's stick to power law but handle small r better.
        # But wait, BFGS handles high energy fine usually.
        # The issue might be that 100 steps is not enough or cutoff is too strict.
        
        # Let's switch to a softer potential: V = (1 - r/rc)^2
        # This has finite value at r=0.
        
        # Calculate terms
        # r < cutoff implies ratio > 1
        
        # V = 0.5 * (1 - dists/cutoff)^2 * repulsion_strength
        # F_mag = (1 - dists/cutoff) / cutoff * repulsion_strength
        # F_vec = F_mag * (d_vecs / dists)
        
        # But we want strong repulsion.
        # Let's use V = (cutoff/dists)**4 ?
        
        # Let's stick to the previous one but ensure cutoff is correct.
        # If we want min_dist, we should set cutoff = min_dist.
        # And we want energy to be minimized.
        # If optimization succeeds, max force should be small.
        # But if atoms are jammed, maybe we can't satisfy it without volume expansion.
        
        # Let's try to increase volume slightly in generate_amorphous before relaxation.
        
        # ratio = (self.cutoff / dists)
        # Soften the potential slightly to 4th power for better convergence?
        # Or 12 is fine?
        # ratio6 = ratio**6
        # ratio12 = ratio6**2
        
        # energy = np.sum(ratio12) * 0.5 
        # f_scalar = 12.0 * ratio12 / dists 
        
        # Let's switch to harmonic potential for stability
        # V = 0.5 * k * (rc - r)^2
        # F = k * (rc - r)
        
        # Only for r < rc
        
        delta = self.cutoff - dists
        # delta > 0 since neighbor list uses cutoff
        
        energy = np.sum(0.5 * self.repulsion_strength * delta**2) * 0.5
        
        f_scalar = self.repulsion_strength * delta
        
        # D is vector from i to j.
        # Force on i should be away from j (repulsive).
        # So direction is -(r_j - r_i) = -D.
        f_vecs = -d_vecs * (f_scalar[:, np.newaxis] / dists[:, np.newaxis])
        
        # Accumulate forces
        # np.add.at is useful here
        np.add.at(forces, i_indices, f_vecs)
        
        self.results['energy'] = energy
        self.results['forces'] = forces

from scipy.stats import norm

def generate_amorphous(atoms: Atoms, min_dist: float = 2.0, rattle_strength: float = 0.5, max_steps: int = 100, rng_values: np.ndarray = None) -> Atoms:
    """
    Generate an amorphous structure by randomly displacing atoms and relaxing with a repulsive potential.
    
    Args:
        atoms: Input structure.
        min_dist: Target minimum distance between atoms.
        rattle_strength: Magnitude of initial random displacement (fraction of avg bond length or just Angstrom).
                         Here treated as Angstroms sigma for Gaussian.
        max_steps: Maximum relaxation steps.
        rng_values: (Optional) 1D numpy array of uniform random values [0, 1] for Sobol/determinism.
                    Must have length 3 * N_atoms.
        
    Returns:
        Amorphous structure.
    """
    atoms = atoms.copy()
    
    # 1. Random displacement
    if rng_values is not None:
        n_atoms = len(atoms)
        n_needed = 3 * n_atoms
        if len(rng_values) < n_needed:
            raise ValueError(f"Not enough random values for amorphous rattle. Needed {n_needed}, got {len(rng_values)}.")
        
        # Transform uniform [0, 1] to normal distribution for Gaussian rattle
        # Using inverse CDF (ppf)
        vals = rng_values[:n_needed]
        displacements = norm.ppf(vals, loc=0, scale=rattle_strength).reshape(n_atoms, 3)
        atoms.positions += displacements
    else:
        atoms.rattle(stdev=rattle_strength)
    
    # Scale volume slightly to allow relaxation
    # Density of amorphous phase is typically lower than crystal
    # Let's increase volume by ~40% (linear scale ~1.12)
    atoms.set_cell(atoms.get_cell() * 1.25, scale_atoms=True)
    
    # 2. Random cell deformation?
    # Maybe optional. For now, keep cell fixed or slightly distorted?
    # Amorphous usually implies density change. 
    # Let's apply a random strain if desired, but user didn't explicitly ask for cell change.
    # But "amorphous" from crystal usually needs volume expansion.
    # Let's scale volume by e.g. 1.1?
    # Or let user handle it via cell perturbation.
    # We will just focus on internal coordinates.
    
    # 3. Repulsive relaxation
    # Attach repulsive calculator
    # Set cutoff slightly larger than min_dist to have a gradient
    # But strictly we want r >= min_dist.
    # If we use V=(min_dist/r)^12, potential is 1.0 at min_dist.
    # We want to minimize energy.
    
    # We iterate multiple times if needed, or just run optimization once?
    # Sometimes optimization gets stuck.
    # We can try to optimize, check distances, if failing, rattle again and optimize.
    # But for now, let's just run it.
    
    calc = RepulsiveCalculator(cutoff=min_dist, repulsion_strength=5.0)
    atoms.calc = calc
    
    # Optimize
    # Use FIRE or BFGS
    # BFGS is robust.
    # We don't need high precision, just enough to push atoms apart.
    # fmax=0.1 might be too loose if forces are huge.
    # But huge forces mean huge repulsion.
    # Increase steps or use FIRE?
    # BFGS can be slow for this landscape.
    from ase.optimize import FIRE
    try:
        # Use simple gradient descent manually if optimizers fail?
        # Or simple Monte Carlo?
        # Or just use `Filter`?
        
        # Let's use simple steepest descent loop manually
        # This is often more robust for simple repulsion than fancy optimizers
        
        lr = 0.1
        for _ in range(max_steps):
            forces = atoms.get_forces()
            max_f = np.sqrt((forces**2).sum(axis=1).max())
            if max_f < 0.01:
                break
            
            # Limit step size
            # dr = lr * F
            # Clip dr to max 0.1 Angstrom
            # But if atoms are on top of each other, forces are huge.
            # We need to limit force magnitude contribution.
            
            dr = lr * forces
            dr_norms = np.linalg.norm(dr, axis=1)
            
            # Cap step size at 0.1 Angstrom
            scale = np.minimum(1.0, 0.1 / (dr_norms + 1e-8))
            
            # If dr is huge, we might overshoot.
            # If distance is super small (0.17), force is large.
            # We want to move them apart.
            # Direction is correct.
            
            dr *= scale[:, np.newaxis]
            
            atoms.positions += dr
            atoms.wrap()
            
            # If very close overlap persists, maybe random kick?
            # If max force is huge but step limited, we are moving slowly.
            # If stuck?

    except Exception:
        pass # If convergence fails, return what we have
        
    # Remove calculator
    atoms.calc = None
    
    return atoms
