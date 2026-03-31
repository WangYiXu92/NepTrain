import unittest
import numpy as np
from ase.build import bulk
from NepTrain.core.perturb.run import perturb

class TestSobolGB(unittest.TestCase):
    def test_sobol_gb_randomness(self):
        # Create a supercell
        atoms = bulk('Al', 'fcc', a=4.05, cubic=True) * (3, 3, 6)
        
        num_samples = 10
        
        # Test with random angle and implicit translation sampling via Sobol
        generator = perturb(
            atoms,
            num=num_samples,
            sampler='sobol',
            gb=True,
            gb_axis='0,0,1',
            gb_angle='random',
            cell_pert_fraction=0.0,
            yield_atoms=True # perturb generator yields atoms
        )
        
        translations = []
        angles = []
        
        for i, struct in enumerate(generator):
            info = struct.info.get('perturb_annotation')
            self.assertIsNotNone(info)
            self.assertEqual(info['type'], 'grain_boundary')
            
            meta = info['metadata']
            
            # Check translation exists and is 3D vector
            trans = meta.get('translation')
            self.assertIsNotNone(trans)
            self.assertEqual(len(trans), 3)
            translations.append(trans)
            
            # Check angle
            angle = meta.get('angle')
            self.assertIsNotNone(angle)
            angles.append(angle)
            
            print(f"Sample {i}: Angle={angle:.2f}, Trans={np.round(trans, 2)}")
            
        # Verify uniqueness
        translations = np.array(translations)
        angles = np.array(angles)
        
        unique_trans = np.unique(translations, axis=0)
        unique_angles = np.unique(angles)
        
        print(f"Unique translations: {len(unique_trans)}/{num_samples}")
        print(f"Unique angles: {len(unique_angles)}/{num_samples}")
        
        self.assertEqual(len(unique_trans), num_samples, "Translations should be unique")
        self.assertEqual(len(unique_angles), num_samples, "Angles should be unique")
        
        # Verify ranges
        self.assertTrue(np.all(angles >= 15.0))
        self.assertTrue(np.all(angles <= 90.0))

if __name__ == '__main__':
    unittest.main()
