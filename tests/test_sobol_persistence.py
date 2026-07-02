import unittest
import os
import json
import numpy as np
from ase import Atoms
from NepTrain.core.perturb.run import perturb

class TestSobolPersistence(unittest.TestCase):
    def setUp(self):
        self.atoms = Atoms('Si2', positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 5, 5], pbc=True)
        self.state_file = 'test_state.json'
        if os.path.exists(self.state_file):
            os.remove(self.state_file)

    def tearDown(self):
        if os.path.exists(self.state_file):
            os.remove(self.state_file)

    def test_save_state(self):
        # Generate 5 structures and save state
        gen = perturb(self.atoms, num=5, sampler='sobol', state_file=self.state_file, similarity_threshold=1.0)
        list(gen)
        
        self.assertTrue(os.path.exists(self.state_file))
        with open(self.state_file, 'r') as f:
            state = json.load(f)
        self.assertEqual(state['num_generated'], 5)
        self.assertIn('seed', state)
        self.assertIn('total_d', state)

    def test_resume_state(self):
        # 1. Generate first 5
        gen1 = perturb(self.atoms, num=5, sampler='sobol', state_file=self.state_file, similarity_threshold=1.0)
        list(gen1)
        
        # 2. Generate next 5 (resuming)
        # Note: In practice, resuming means "start from N and generate M more".
        # But our perturb(num=X) usually means "generate X structures total" or "X more"?
        # Standard interpretation: num is "number to generate in THIS run".
        # So if we resume, we fast-forward by saved state, then generate num.
        gen2 = perturb(self.atoms, num=5, sampler='sobol', state_file=self.state_file, resume=True, similarity_threshold=1.0)
        structs2 = list(gen2)
        self.assertEqual(len(structs2), 5)
        
        # 3. Verify total generated is 10 in state file
        with open(self.state_file, 'r') as f:
            state = json.load(f)
        self.assertEqual(state['num_generated'], 10)

    def test_resume_consistency(self):
        # Verify that splitting 10 into 5+5 yields same as 10 at once
        # This requires fixing the seed
        seed = 42
        
        # Run A: 10 at once
        gen_a = perturb(self.atoms, num=10, sampler='sobol', seed=seed, similarity_threshold=1.0)
        structs_a = list(gen_a)
        
        # Run B: 5 then 5
        state_file_b = 'test_state_b.json'
        if os.path.exists(state_file_b):
            os.remove(state_file_b)
            
        try:
            gen_b1 = perturb(self.atoms, num=5, sampler='sobol', seed=seed, state_file=state_file_b, similarity_threshold=1.0)
            structs_b1 = list(gen_b1)
            
            gen_b2 = perturb(self.atoms, num=5, sampler='sobol', seed=seed, state_file=state_file_b, resume=True, similarity_threshold=1.0)
            structs_b2 = list(gen_b2)
            
            # Check positions of 6th structure (index 0 of second batch vs index 5 of first batch)
            # Note: Floating point exact match might be tricky, use almost equal
            pos_a = structs_a[5].get_positions()
            pos_b = structs_b2[0].get_positions()
            np.testing.assert_allclose(pos_a, pos_b, atol=1e-6)
        finally:
            if os.path.exists(state_file_b):
                os.remove(state_file_b)

if __name__ == '__main__':
    unittest.main()
