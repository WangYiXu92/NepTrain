from ase.build import surface as ase_surface
from ase.atoms import Atoms

def generate_surface(atoms: Atoms, indices: tuple = (1, 1, 1), vacuum: float = 10.0, layers: int = 3) -> Atoms:
    """
    Generate a surface slab from a bulk structure.
    
    Args:
        atoms: Input bulk Atoms object
        indices: Miller indices (h, k, l) as a tuple of integers
        vacuum: Vacuum size in Angstroms to add on both sides of the slab
        layers: Number of atomic layers in the slab
        
    Returns:
        Atoms object representing the slab with vacuum and correct PBC (True, True, False)
    """
    # ase.build.surface returns a slab
    slab = ase_surface(atoms, indices, layers)
    
    # Add vacuum
    # axis=2 is Z-axis
    slab.center(vacuum=vacuum, axis=2)
    
    # Ensure PBC is set correctly for a slab (periodic in x,y, not z)
    slab.set_pbc([True, True, False])
    
    return slab
