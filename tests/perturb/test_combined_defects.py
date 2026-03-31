import unittest
import numpy as np
from ase.build import bulk
from NepTrain.core.perturb.run import perturb

class TestCombinedDefects(unittest.TestCase):
    def setUp(self):
        # 4x4x4 BCC Fe supercell
        self.atoms = bulk('Fe', 'bcc', a=2.87, cubic=True) * (4, 4, 4)
        
    def test_dislocation_magnetic_random_axis(self):
        """
        Test combined Dislocation + Magnetic Random Axis perturbation using Sobol.
        """
        num_samples = 5
        
        # Configuration
        # Dislocation: Edge dislocation
        # Magnetic: Random axis collinear (axis='random')
        
        generator = perturb(
            self.atoms,
            num=num_samples,
            sampler='sobol',
            # Dislocation settings
            dislocation=True,
            dislocation_type='edge',
            dislocation_axis=2, # Z axis
            cell_pert_fraction=0.01, # Small strain
            
            # Magnetic settings
            mag_mode='random_collinear',
            mag_noise=0.1,
            mag_kwargs={'axis': 'random'},
            
            # General settings
            min_distance=0.1,
            skip_normal=True, # Focus on defects
            yield_atoms=True,
            validate_structure=False # Avoid adjust_reasonable rejections for test stability
        )
        
        results = list(generator)
        self.assertEqual(len(results), num_samples)
        
        for i, struct in enumerate(results):
            # Check Config_type
            config_type = struct.info.get('Config_type', '')
            self.assertIn('disloc_edge', config_type)
            self.assertIn('mag_random_collinear', config_type)
            
            # Check Annotation
            info = struct.info.get('perturb_annotation')
            # Annotation usually keeps the last major topological defect type, 
            # or maybe we need to check if multiple annotations are preserved?
            # Current run.py overwrites 'perturb_annotation' for each defect type sequentially.
            # So dislocation will be overwritten by subsequent defects if any?
            # Order in run.py: Surface -> GB -> Dislocation -> Twinning -> SF -> Amorphous -> Mag -> Rot -> Vac -> Shuffle
            # Mag is applied AFTER dislocation.
            # But Mag usually doesn't set 'perturb_annotation' unless it's a specific mag-only perturbation?
            # Let's check run.py.
            
            # run.py L749: apply_magnetic_perturbation
            # apply_magnetic_perturbation usually modifies atoms.set_initial_magnetic_moments
            # It DOES NOT usually set 'perturb_annotation' unless it's designed to.
            # But wait, run.py doesn't seem to set 'perturb_annotation' for Mag explicitly in the block
            # (unlike dislocation/GB/etc).
            # So the annotation should remain from Dislocation!
            
            self.assertIsNotNone(info)
            self.assertEqual(info['type'], 'dislocation')
            self.assertEqual(info['metadata']['dislocation_type'], 'edge')
            
            # Check Magnetic Moments
            moms = struct.get_initial_magnetic_moments()
            self.assertTrue(np.any(np.abs(moms) > 0.01))
            
            # Since axis is random, moments should change direction or magnitude across samples
            # But for random_collinear with random axis, it sets a global axis for the structure?
            # Or per atom? 'random_collinear' implies all atoms share an axis (collinear) but that axis is random.
            # Let's verify variance if possible, but 5 samples might be small.
            
            # Check Sobol usage
            # We can't easily check if it came from Sobol inside the struct, 
            # but we trust the generator logic verified in other tests.

    def test_grain_boundary_magnetic_random_axis(self):
        """
        Test combined Grain Boundary + Magnetic Random Axis perturbation using Sobol.
        """
        num_samples = 3
        
        generator = perturb(
            self.atoms,
            num=num_samples,
            sampler='sobol',
            # GB settings
            gb=True,
            gb_axis='0,0,1',
            gb_angle=30.0,
            
            # Magnetic settings
            mag_mode='random_collinear',
            mag_noise=0.1,
            mag_kwargs={'axis': 'random'},
            
            # General settings
            min_distance=0.1,
            skip_normal=True,
            yield_atoms=True,
            validate_structure=False
        )
        
        results = list(generator)
        self.assertEqual(len(results), num_samples)
        
        for i, struct in enumerate(results):
            config_type = struct.info.get('Config_type', '')
            self.assertIn('gb0,0,1', config_type)
            self.assertIn('mag_random_collinear', config_type)
            
            info = struct.info.get('perturb_annotation')
            self.assertIsNotNone(info)
            self.assertEqual(info['type'], 'grain_boundary')
            self.assertIn('axis', info['metadata'])
            self.assertIn('magnetic_mode', info['metadata'])
            
            moms = struct.get_initial_magnetic_moments()
            self.assertTrue(np.any(np.abs(moms) > 0.01))

    def test_twinning_magnetic_random_axis(self):
        """
        Test combined Twinning + Magnetic Random Axis perturbation using Sobol.
        """
        num_samples = 3
        
        generator = perturb(
            self.atoms,
            num=num_samples,
            sampler='sobol',
            # Twinning settings
            twinning=True,
            twinning_min_dist=0.0, # Zero dist to avoid variable atom deletion
            
            # Magnetic settings
            mag_mode='random_collinear',
            mag_noise=0.1,
            mag_kwargs={'axis': 'random'},
            
            # General settings
            min_distance=0.1,
            skip_normal=True,
            yield_atoms=True,
            validate_structure=False
        )
        
        results = list(generator)
        self.assertEqual(len(results), num_samples)
        
        for i, struct in enumerate(results):
            config_type = struct.info.get('Config_type', '')
            self.assertIn('twin_', config_type)
            self.assertIn('mag_random_collinear', config_type)
            
            info = struct.info.get('perturb_annotation')
            self.assertIsNotNone(info)
            self.assertEqual(info['type'], 'twinning')
            self.assertIn('normal_cart', info['metadata'])
            self.assertIn('magnetic_mode', info['metadata'])
            
            moms = struct.get_initial_magnetic_moments()
            self.assertTrue(np.any(np.abs(moms) > 0.01))

    def test_stacking_fault_magnetic_random_axis(self):
        """
        Test combined Stacking Fault + Magnetic Random Axis perturbation using Sobol.
        """
        num_samples = 3
        
        generator = perturb(
            self.atoms,
            num=num_samples,
            sampler='sobol',
            # SF settings
            stacking_fault=True,
            
            # Magnetic settings
            mag_mode='random_collinear',
            mag_noise=0.1,
            mag_kwargs={'axis': 'random'},
            
            # General settings
            min_distance=0.1,
            skip_normal=True,
            yield_atoms=True,
            validate_structure=False
        )
        
        results = list(generator)
        self.assertEqual(len(results), num_samples)
        
        for i, struct in enumerate(results):
            config_type = struct.info.get('Config_type', '')
            self.assertIn('sf_', config_type)
            self.assertIn('mag_random_collinear', config_type)
            
            info = struct.info.get('perturb_annotation')
            self.assertIsNotNone(info)
            self.assertEqual(info['type'], 'stacking_fault')
            self.assertIn('normal_cart', info['metadata'])
            self.assertIn('magnetic_mode', info['metadata'])
            
            moms = struct.get_initial_magnetic_moments()
            self.assertTrue(np.any(np.abs(moms) > 0.01))


if __name__ == '__main__':
    unittest.main()
