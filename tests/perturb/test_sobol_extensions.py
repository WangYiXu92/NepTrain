import unittest
import numpy as np
from ase import Atoms
from ase.build import molecule
from NepTrain.core.perturb.run import perturb
from NepTrain.core.perturb.rigid import RigidBodyManager

class TestSobolExtensions(unittest.TestCase):
    
    def test_rigid_sobol_determinism(self):
        """Test that rigid body perturbation with Sobol is deterministic."""
        # CO molecule
        atoms = molecule('CO')
        atoms.set_cell([10, 10, 10])
        atoms.center()
        
        # Run 1: Seed 42
        gen1 = perturb(atoms, num=5, rigid=True, rigid_method='auto', 
                       sampler='sobol', seed=42, cell_pert_fraction=0.0,
                       validate_structure=False)
        structs1 = list(gen1)
        
        # Run 2: Seed 42
        gen2 = perturb(atoms, num=5, rigid=True, rigid_method='auto', 
                       sampler='sobol', seed=42, cell_pert_fraction=0.0,
                       validate_structure=False)
        structs2 = list(gen2)
        
        # Run 3: Seed 123
        gen3 = perturb(atoms, num=5, rigid=True, rigid_method='auto', 
                       sampler='sobol', seed=123, cell_pert_fraction=0.0,
                       validate_structure=False)
        structs3 = list(gen3)
        
        # Assertions
        for s1, s2 in zip(structs1, structs2):
            np.testing.assert_array_almost_equal(s1.get_positions(), s2.get_positions(), 
                                                 err_msg="Rigid Sobol should be deterministic")
            
        # Check difference with different seed
        diff = np.abs(structs1[0].get_positions() - structs3[0].get_positions())
        self.assertTrue(np.max(diff) > 1e-4, "Different seeds should produce different results")
        
        # Check that it is indeed rigid
        # CO bond length should be preserved exactly (if intra perturbation is off, which is default)
        d0 = atoms.get_distance(0, 1)
        for s in structs1:
            d = s.get_distance(0, 1)
            self.assertAlmostEqual(d, d0, places=5, msg="Rigid body should preserve bond length")

    def test_magnetic_random_axis_sobol(self):
        """Test that magnetic random axis with Sobol is deterministic."""
        atoms = Atoms('Fe', positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        
        # Run 1: Seed 42
        gen1 = perturb(atoms, num=5, mag_mode='collinear', 
                       mag_kwargs={'axis': 'random'},
                       sampler='sobol', seed=42, cell_pert_fraction=0.0,
                       validate_structure=False)
        structs1 = list(gen1)
        
        # Run 2: Seed 42
        gen2 = perturb(atoms, num=5, mag_mode='collinear', 
                       mag_kwargs={'axis': 'random'},
                       sampler='sobol', seed=42, cell_pert_fraction=0.0,
                       validate_structure=False)
        structs2 = list(gen2)
        
        # Run 3: Seed 123
        gen3 = perturb(atoms, num=5, mag_mode='collinear', 
                       mag_kwargs={'axis': 'random'},
                       sampler='sobol', seed=123, cell_pert_fraction=0.0,
                       validate_structure=False)
        structs3 = list(gen3)
        
        # Assertions
        for i, (s1, s2) in enumerate(zip(structs1, structs2)):
            m1 = s1.get_initial_magnetic_moments()
            m2 = s2.get_initial_magnetic_moments()
            np.testing.assert_array_almost_equal(m1, m2, 
                                                 err_msg=f"Magnetic Sobol should be deterministic (idx {i})")
            
            # Check annotation
            ann1 = s1.info['perturb_annotation']
            ann2 = s2.info['perturb_annotation']
            np.testing.assert_array_almost_equal(ann1['axis'], ann2['axis'],
                                                 err_msg="Axis annotation should match")

        # Check difference with different seed
        m1 = structs1[0].get_initial_magnetic_moments()
        m3 = structs3[0].get_initial_magnetic_moments()
        # They could accidentally match if random numbers align, but unlikely for floats
        self.assertFalse(np.allclose(m1, m3), "Different seeds should produce different magnetic axes")
        
        # Check that axis varies within the sequence (Sobol should explore space)
        # Note: Axis is global for 'collinear' mode?
        # Wait, apply_magnetic_perturbation is called PER structure in the loop in run.py.
        # So each structure gets a NEW random axis if axis='random'.
        # Let's verify this.
        
        axis1 = structs1[0].info['perturb_annotation']['axis']
        axis2 = structs1[1].info['perturb_annotation']['axis']
        self.assertFalse(np.allclose(axis1, axis2), "Axis should vary between samples in Sobol sequence")

if __name__ == '__main__':
    unittest.main()
