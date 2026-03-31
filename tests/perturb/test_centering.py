
import sys
import os
import numpy as np
from ase.build import bulk
from ase.io import write

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from NepTrain.core.perturb.grain_boundary import generate_grain_boundary
from NepTrain.core.perturb.twinning import generate_twinning

def test_interface_centering():
    print("Testing Interface Centering at 1/2 Z...")
    
    # 1. Test Grain Boundary
    print("\n--- Testing GB Centering ---")
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    
    gb = generate_grain_boundary(
        atoms, 
        sigma=5, 
        axis=[0,0,1], 
        min_dist=1.8,
        vacuum=10.0 # Add explicit vacuum
    )
    
    cell = gb.get_cell()
    pos = gb.get_positions()
    z_coords = pos[:, 2]
    
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    z_range = z_max - z_min
    z_box = cell[2, 2]
    z_center_atoms = (z_max + z_min) / 2.0
    z_center_box = z_box / 2.0
    
    print(f"GB Box Z: {z_box:.3f}")
    print(f"Atom Z Range: {z_min:.3f} - {z_max:.3f} (Span: {z_range:.3f})")
    print(f"Atom Center Z: {z_center_atoms:.3f}")
    print(f"Box Center Z: {z_center_box:.3f}")
    print(f"Deviation: {abs(z_center_atoms - z_center_box):.3f}")
    
    # Allow small deviation due to atomic discreteness, but interface (gap) should be near center
    # For a symmetric slab, atom center should be box center
    assert abs(z_center_atoms - z_center_box) < 2.0, "GB Atoms not centered in Z"
    
    write("test_centered_gb.xyz", gb)
    
    # 2. Test Twinning
    print("\n--- Testing Twin Centering ---")
    twin = generate_twinning(
        atoms,
        miller_indices=(1, 1, 1),
        min_dist=1.8,
        layers=6,
        vacuum=10.0
    )
    
    cell_t = twin.get_cell()
    pos_t = twin.get_positions()
    z_coords_t = pos_t[:, 2]
    
    z_min_t, z_max_t = np.min(z_coords_t), np.max(z_coords_t)
    z_center_atoms_t = (z_max_t + z_min_t) / 2.0
    z_center_box_t = cell_t[2, 2] / 2.0
    
    print(f"Twin Box Z: {cell_t[2, 2]:.3f}")
    print(f"Atom Center Z: {z_center_atoms_t:.3f}")
    print(f"Box Center Z: {z_center_box_t:.3f}")
    print(f"Deviation: {abs(z_center_atoms_t - z_center_box_t):.3f}")
    
    assert abs(z_center_atoms_t - z_center_box_t) < 1.0, "Twin Atoms not centered in Z"
    
    write("test_centered_twin.xyz", twin)

if __name__ == "__main__":
    test_interface_centering()
