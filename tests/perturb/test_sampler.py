import unittest
import numpy as np
from NepTrain.core.perturb.sampler import SobolSampler, RandomSampler

class TestSampler(unittest.TestCase):
    def test_random_sampler(self):
        sampler = RandomSampler(d=2)
        samples = sampler.random(n=10)
        self.assertEqual(samples.shape, (10, 2))
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples <= 1))

    def test_sobol_sampler(self):
        sampler = SobolSampler(d=2, scramble=True)
        samples = sampler.random(n=4) # Sobol usually works best with powers of 2
        self.assertEqual(samples.shape, (4, 2))
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples <= 1))
        
    def test_scaling(self):
        # Test scaling from [0, 1] to [min, max]
        sampler = SobolSampler(d=1)
        samples = sampler.random(n=5)
        scaled = sampler.scale(samples, l_bounds=[-1], u_bounds=[1])
        self.assertTrue(np.all(scaled >= -1))
        self.assertTrue(np.all(scaled <= 1))

if __name__ == '__main__':
    unittest.main()
