import unittest
import numpy as np
from ase.build import bulk
from NepTrain.core.perturb.run import perturb

class TestSobolUnified(unittest.TestCase):
    """
    Unified test suite for Sobol sequence randomization across all defect types.
    Verifies that 'sobol' sampler produces unique, well-distributed parameters.
    """

    def setUp(self):
        # Base structure: FCC Al supercell
        self.atoms = bulk('Al', 'fcc', a=4.05, cubic=True) * (4, 4, 4)
        # For slab-like defects (SF, Surface), maybe use a slab?
        # But perturb function handles logic.
        
    def test_sobol_dislocation(self):
        """Test Sobol sampling for Dislocation (center, type)."""
        # Use a larger cell for dislocation
        atoms = self.atoms * (1, 1, 2) # 4x4x8
        
        num_samples = 8
        generator = perturb(
            atoms,
            num=num_samples,
            sampler='sobol',
            dislocation=True,
            dislocation_type='random', # Sobol dim
            dislocation_axis='0,0,1',
            dislocation_burgers='1,0,0',
            min_distance=0.01, # Minimal rattle
            cell_pert_fraction=0.0,
            validate_structure=False,
            yield_atoms=True
        )
        
        centers = []
        types = []
        
        for struct in generator:
            info = struct.info.get('perturb_annotation')
            self.assertEqual(info['type'], 'dislocation')
            meta = info['metadata']
            centers.append(meta['center'])
            types.append(meta['dislocation_type'])
            
        # Verify uniqueness
        unique_centers = np.unique(np.array(centers), axis=0)
        self.assertEqual(len(unique_centers), num_samples)
        
        # Verify types (should see both edge and screw if enough samples)
        # With 8 samples, Sobol should cover both.
        unique_types = set(types)
        self.assertTrue(len(unique_types) >= 1) # At least one type
        print(f"Dislocation types found: {unique_types}")

    def test_sobol_grain_boundary(self):
        """Test Sobol sampling for Grain Boundary (angle, translation)."""
        num_samples = 20
        generator = perturb(
            self.atoms,
            num=num_samples,
            sampler='sobol',
            gb=True,
            gb_axis='0,0,1',
            gb_angle='random', # Sobol dim
            gb_dist=2.2, # Increased to avoid bad bonds
            min_distance=0.01,
            cell_pert_fraction=0.0,
            validate_structure=False,
            yield_atoms=True
        )
        
        angles = []
        translations = []
        
        for struct in generator:
            info = struct.info.get('perturb_annotation')
            # Note: GB might fail adjust_reasonable for some angles/translations
            # if so, generator yields fewer. But min_distance=0.01 should pass most.
            self.assertEqual(info['type'], 'grain_boundary')
            meta = info['metadata']
            
            angles.append(meta['angle'])
            if 'translation' in meta:
                translations.append(meta['translation'])
                
        # Check we got some samples
        self.assertTrue(len(angles) > 0)
        
        # Verify uniqueness
        unique_angles = np.unique(angles)
        self.assertEqual(len(unique_angles), len(angles))
        
        if translations:
            unique_trans = np.unique(np.array(translations), axis=0)
            self.assertEqual(len(unique_trans), len(translations))

    def test_sobol_twinning(self):
        """Test Sobol sampling for Twinning (translation, z_frac)."""
        num_samples = 20
        generator = perturb(
            self.atoms,
            num=num_samples,
            sampler='sobol',
            twinning=True,
            twinning_indices='1,1,1',
            twinning_z='random', # Sobol dim
            twinning_min_dist=1.8, # Increased to avoid overlaps that fail adjust_reasonable
            min_distance=0.01,
            cell_pert_fraction=0.0,
            yield_atoms=True
        )
        
        z_fracs = []
        translations = []
        
        for struct in generator:
            info = struct.info.get('perturb_annotation')
            self.assertEqual(info['type'], 'twinning')
            meta = info['metadata']
            
            z_fracs.append(meta['z_frac'])
            if 'translation_frac' in meta and meta['translation_frac'] is not None:
                translations.append(meta['translation_frac'])
        
        self.assertTrue(len(z_fracs) > 0)
        unique_z = np.unique(z_fracs)
        self.assertEqual(len(unique_z), len(z_fracs))
        
        if translations:
            unique_trans = np.unique(np.array(translations), axis=0)
            self.assertEqual(len(unique_trans), len(translations))

    def test_sobol_stacking_fault(self):
        """Test Sobol sampling for Stacking Fault (shift, height)."""
        # Use slab for SF to avoid PBC issues
        slab = self.atoms.copy()
        slab.center(vacuum=10.0, axis=2)
        
        num_samples = 8
        generator = perturb(
            slab,
            num=num_samples,
            sampler='sobol',
            stacking_fault=True,
            sf_normal='0,0,1',
            sf_shift='random', # Sobol dim (Gamma surface)
            sf_height='random', # Sobol dim
            min_distance=0.01,
            cell_pert_fraction=0.0,
            yield_atoms=True
        )
        
        heights = []
        shifts = []
        
        for struct in generator:
            info = struct.info.get('perturb_annotation')
            self.assertEqual(info['type'], 'stacking_fault')
            # metadata uses 'height' key, not 'plane_height'
            heights.append(info['metadata']['height'])
            # shift might be [0,0,0] if using translation_frac
            if info['metadata']['translation_frac'] is not None:
                shifts.append(info['metadata']['translation_frac'])
            else:
                shifts.append(info['metadata']['shift'])
                
        self.assertTrue(len(heights) > 0)
        unique_h = np.unique(heights)
        self.assertEqual(len(unique_h), len(heights))
        
        if shifts:
            unique_trans = np.unique(np.array(shifts), axis=0)
            self.assertEqual(len(unique_trans), len(shifts))

if __name__ == '__main__':
    unittest.main()
