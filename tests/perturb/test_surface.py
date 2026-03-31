import unittest
from ase.build import bulk
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

# Try import, will fail initially
try:
    from NepTrain.core.perturb.surface import generate_surface
except ImportError:
    pass

class TestSurface(unittest.TestCase):
    def test_generate_surface(self):
        try:
            from NepTrain.core.perturb.surface import generate_surface
        except ImportError:
            self.fail("Could not import generate_surface")

        atoms = bulk('Cu', 'fcc', a=3.6)
        # Generate (1,1,1) surface with 10A vacuum and 4 layers
        # Note: ase.build.surface creates a slab. 
        slab = generate_surface(atoms, indices=(1,1,1), vacuum=10.0, layers=4)
        
        # Check PBC: typically slabs are periodic in x,y but not z
        self.assertFalse(slab.pbc[2])
        self.assertTrue(slab.pbc[0])
        self.assertTrue(slab.pbc[1])
        
        # Check atoms count: bulk fcc Cu has 1 atom basis. (111) surface unit cell?
        # ase.build.surface returns a minimal surface cell.
        # For fcc(111), it usually has 1 atom per layer in primitive setting, or more.
        # Let's just check it's not empty and has correct vacuum.
        self.assertTrue(len(slab) > 0)
        
        # Check vacuum
        # The z-height of cell should be > layers thickness + 2*vacuum
        # 4 layers of Cu(111) ~ 4 * 2.08A ~ 8A. Vacuum 10A * 2 = 20A? 
        # ase.atoms.center(vacuum=10) adds 10A on each side (top/bottom) or total? 
        # doc says: "vacuum: float (default: None) If specified, the atomic positions are adjusted 
        # and the unit cell is extended to provide the given amount of vacuum on both sides of the slab."
        # So total vacuum added to cell length is 2*vacuum.
        
        z_length = slab.cell[2,2]
        positions = slab.get_positions()
        z_coords = positions[:, 2]
        thickness = z_coords.max() - z_coords.min()
        
        self.assertTrue(z_length >= thickness + 2 * 10.0 - 0.1) # Tolerance
        
if __name__ == '__main__':
    unittest.main()
