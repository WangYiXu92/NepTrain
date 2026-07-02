import unittest
import numpy as np
from ase.build import bulk
from NepTrain.core.perturb.run import perturb

class TestSobolTwinning(unittest.TestCase):
    def test_sobol_twinning_randomness(self):
        # Create a supercell (FCC Al)
        atoms = bulk('Al', 'fcc', a=4.05, cubic=True) * (3, 3, 6)
        
        num_samples = 10
        
        # Test with random z and implicit translation sampling via Sobol
        # Use min_dist=2.0 to pass adjust_reasonable check for Al (covalent radii sum ~2.4)
        generator = perturb(
            atoms,
            num=num_samples,
            sampler='sobol',
            twinning=True,
            twinning_indices='1,1,1',
            twinning_z='random',
            twinning_min_dist=2.0,
            cell_pert_fraction=0.0,
            min_distance=0.0,
            validate_structure=False,
            similarity_threshold=1.0,
            yield_atoms=True
        )
        
        translations = []
        z_fracs = []
        
        for i, struct in enumerate(generator):
            info = struct.info.get('perturb_annotation')
            self.assertIsNotNone(info)
            self.assertEqual(info['type'], 'twinning')
            
            meta = info['metadata']
            
            # Check translation_frac exists
            trans = meta.get('translation_frac')
            self.assertIsNotNone(trans)
            self.assertEqual(len(trans), 2)
            translations.append(trans)
            
            # Current twinning builds a symmetric slab; z_frac is no longer a
            # sampled API output. The wrapper records the resulting plane_d.
            z_cut = meta.get('plane_d')
            self.assertIsNotNone(z_cut)
            z_fracs.append(z_cut)
            
            print(f"Sample {i}: Z_cut={z_cut:.2f}, TransFrac={np.round(trans, 2)}")
            
        # Verify uniqueness
        translations = np.array(translations)
        z_fracs = np.array(z_fracs)
        
        unique_trans = np.unique(translations, axis=0)
        unique_z = np.unique(z_fracs)
        
        print(f"Unique translations: {len(unique_trans)}/{len(translations)}")
        print(f"Unique Z heights: {len(unique_z)}/{len(z_fracs)}")
        
        self.assertGreater(len(translations), 0, "Should generate at least some valid structures")
        self.assertEqual(len(unique_trans), len(translations), "All generated translations should be unique")
        self.assertGreater(len(unique_z), 0, "Generated twins should report plane positions")

if __name__ == '__main__':
    unittest.main()
