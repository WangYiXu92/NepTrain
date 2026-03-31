import unittest
import numpy as np
from ase.build import bulk
from NepTrain.core.perturb.run import perturb

class TestSobolStackingFault(unittest.TestCase):
    def test_sobol_sf_randomness(self):
        # Create a supercell (FCC Al)
        # Use a SLAB to avoid PBC issues at the boundary when shifting randomly
        atoms = bulk('Al', 'fcc', a=4.05, cubic=True) * (3, 3, 6)
        atoms.center(vacuum=10.0, axis=2)
        
        num_samples = 10
        
        # Test with random shift and random height
        # Normal along Z [0,0,1]
        generator = perturb(
            atoms,
            num=num_samples,
            sampler='sobol',
            stacking_fault=True,
            sf_normal='0,0,1',
            sf_shift='random',
            sf_height='random',
            min_distance=0.01, # Small rattle to avoid destroying structure
            cell_pert_fraction=0.0,
            yield_atoms=True
        )
        
        translations = []
        heights = []
        shifts = []
        
        for i, struct in enumerate(generator):
            info = struct.info.get('perturb_annotation')
            self.assertIsNotNone(info)
            self.assertEqual(info['type'], 'stacking_fault')
            
            meta = info['metadata']
            
            # Check translation_frac exists
            trans = meta.get('translation_frac')
            self.assertIsNotNone(trans)
            self.assertEqual(len(trans), 2)
            translations.append(trans)
            
            # Check plane_height
            h = meta.get('height')
            self.assertIsNotNone(h)
            heights.append(h)
            
            # Check calculated shift vector
            # In run.py, shift is stored as 'shift' in metadata
            s = meta.get('shift')
            shifts.append(s)
            
            # Check that shift vector has z-component approx 0 (since normal is Z)
            # But wait, extra_shift logic projects cell vectors.
            # If cell is cubic aligned, u, v are in XY plane.
            # So shift.z should be 0.
            self.assertAlmostEqual(s[2], 0.0, delta=1e-5)
            
            print(f"Sample {i}: Height={h:.2f}, TransFrac={np.round(trans, 2)}, Shift={np.round(s, 2)}")
            
        # Verify uniqueness
        translations = np.array(translations)
        heights = np.array(heights)
        
        unique_trans = np.unique(translations, axis=0)
        unique_h = np.unique(heights)
        
        print(f"Unique translations: {len(unique_trans)}/{len(translations)}")
        print(f"Unique heights: {len(unique_h)}/{len(heights)}")
        
        self.assertGreater(len(translations), 0)
        self.assertEqual(len(unique_trans), len(translations))
        self.assertEqual(len(unique_h), len(heights))

if __name__ == '__main__':
    unittest.main()
