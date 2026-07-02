import unittest
import os
import shutil
from ase.build import bulk
from ase.io import write
from NepTrain.core.perturb.run import perturb
import numpy as np

class TestWorkflowSampler(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'test_sampler_workflow_output'
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        self.atoms = bulk('Cu', 'fcc', a=3.6)
        self.test_file = os.path.join(self.test_dir, 'input.xyz')
        write(self.test_file, self.atoms)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_sobol_execution(self):
        """Test that SobolSampler runs without error."""
        results = perturb(
            self.atoms,
            num=2,
            sampler='sobol',
            seed=42,
            cell_pert_fraction=0.01,
            similarity_threshold=1.0,
        )
        results = list(results)
        self.assertEqual(len(results), 2)

if __name__ == '__main__':
    unittest.main()
