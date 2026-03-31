import unittest
import numpy as np
from ase.build import bulk
from NepTrain.core.perturb.run import perturb

class TestSobolCombined(unittest.TestCase):
    def setUp(self):
        # 4x4x4 Fe BCC
        self.atoms = bulk('Fe', 'bcc', a=2.87, cubic=True) * (4, 4, 4)
        
    def test_combined_dislocation_magnetic(self):
        """
        Test that combining Dislocation (structural) and Magnetic Random Axis (property)
        works correctly with Sobol sampling.
        
        This validates that the dimension offset logic in run.py is correct:
        - Dislocation consumes dimensions for center (2) + type (1)
        - Magnetic consumes dimensions for random axis (2)
        """
        # Configuration
        # Dislocation: type='random' -> needs 3 dims (2 center + 1 type)
        # Magnetic: axis='random' -> needs 2 dims (theta, phi)
        # Plus standard cell (9) + disp (3*N) + mag_noise (N) + mag_collinear (N)
        
        kwargs = {
            'dislocation': True,
            'dislocation_type': 'random', # Sobol used for type selection
            'dislocation_axis': [0, 0, 1],
            'dislocation_burgers': [0, 0, 1], # Dummy
            
            'mag_mode': 'random_collinear',
            'mag_kwargs': {'axis': 'random'}, # Sobol used for axis generation
            'mag_noise': 0.0, # Disable noise to focus on axis determinism
            
            'sampler': 'sobol',
            'cell_pert_fraction': 0.0, # Disable cell strain
            'min_distance': 0.1,
            'validate_structure': False,
            'num': 5
        }
        
        # Run 1: Seed 123
        gen1 = perturb(self.atoms.copy(), seed=123, **kwargs)
        results1 = list(gen1)
        
        # Run 2: Seed 123 (Should be identical)
        gen2 = perturb(self.atoms.copy(), seed=123, **kwargs)
        results2 = list(gen2)
        
        # Run 3: Seed 999 (Should be different)
        gen3 = perturb(self.atoms.copy(), seed=999, **kwargs)
        results3 = list(gen3)
        
        # Verify Determinism (Run 1 vs Run 2)
        for i, (s1, s2) in enumerate(zip(results1, results2)):
            # Check Positions (influenced by dislocation)
            np.testing.assert_array_almost_equal(
                s1.positions, s2.positions, 
                err_msg=f"Positions mismatch at step {i} for same seed"
            )
            
            # Check Magnetic Moments (influenced by random axis)
            m1 = s1.get_initial_magnetic_moments()
            m2 = s2.get_initial_magnetic_moments()
            np.testing.assert_array_almost_equal(
                m1, m2,
                err_msg=f"Magnetic moments mismatch at step {i} for same seed"
            )
            
            # Check Metadata (perturb_annotation)
            # Check dislocation type
            d_type1 = s1.info['perturb_annotation']['metadata']['dislocation_type']
            d_type2 = s2.info['perturb_annotation']['metadata']['dislocation_type']
            self.assertEqual(d_type1, d_type2, f"Dislocation type mismatch at step {i}")
            
            # Check magnetic axis is implicit in moments, but we can verify consistency
            # If moments are same, axis was same.
            
        # Verify Difference (Run 1 vs Run 3)
        # It is statistically possible but unlikely that they are identical.
        # We check the first structure.
        s1_0 = results1[0]
        s3_0 = results3[0]
        
        # Positions should differ (different dislocation center/type)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_almost_equal(s1_0.positions, s3_0.positions)
            
        # Magnetic moments should differ (different random axis)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_almost_equal(
                s1_0.get_initial_magnetic_moments(), 
                s3_0.get_initial_magnetic_moments()
            )

if __name__ == '__main__':
    unittest.main()
