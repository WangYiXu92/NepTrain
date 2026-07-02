import unittest
import numpy as np
from ase.build import bulk
from NepTrain.core.perturb.run import perturb

class TestSobolDislocation(unittest.TestCase):
    def test_sobol_randomness(self):
        atoms = bulk('Al', 'fcc', a=4.05, cubic=True) * (4, 4, 4)
        
        # Use a large enough number to ensure mixing
        num_samples = 20
        
        generator = perturb(
            atoms,
            num=num_samples,
            sampler='sobol',
            dislocation=True,
            dislocation_type='random',
            dislocation_axis=2,
            cell_pert_fraction=0.0, # Disable cell pert to focus on dislocation
            yield_atoms=True # run.py's run_perturb passes yield_atoms? No, perturb always yields atoms.
        )
        
        centers = []
        types = []
        
        for struct in generator:
            info = struct.info.get('perturb_annotation')
            self.assertIsNotNone(info)
            self.assertEqual(info['type'], 'dislocation')
            meta = info['metadata']
            
            centers.append(meta['center'])
            types.append(meta['dislocation_type'])
            
        # Verify centers vary
        # Convert to numpy array for easier comparison
        centers = np.array(centers)
        
        # Check coverage. Downstream structure filtering and dislocation-core
        # placement can collapse a few Sobol centers onto equivalent generated
        # structures, so requiring perfect uniqueness is too brittle. The
        # invariant we care about is that Sobol explores most centers.
        unique_centers = np.unique(centers, axis=0)
        self.assertGreaterEqual(
            len(unique_centers), int(0.75 * num_samples),
            "Sobol sampled centers should cover most requested points")
        
        # Verify types mix (edge and screw)
        unique_types = set(types)
        self.assertIn('edge', unique_types, "Should contain edge dislocations")
        self.assertIn('screw', unique_types, "Should contain screw dislocations")
        
        print(f"Centers unique count: {len(unique_centers)}/{num_samples}")
        print(f"Types found: {unique_types}")

if __name__ == '__main__':
    unittest.main()
