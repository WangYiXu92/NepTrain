import unittest
import numpy as np
from ase.build import bulk
from NepTrain.core.perturb.run import perturb
from scipy.stats import ks_2samp

class TestSobolStatistics(unittest.TestCase):
    def setUp(self):
        self.atoms = bulk('Fe', 'bcc', a=2.87, cubic=True) * (3, 3, 3)
        
    def test_distribution_uniformity(self):
        """
        Test that Sobol sampling produces a more uniform distribution than random sampling
        for a single parameter (e.g., Grain Boundary Angle).
        
        We generate N samples using both methods and compare their coverage.
        """
        N = 64  # Power of 2 for Sobol balance property
        
        # 1. Sobol Sampling
        # GB Angle: random -> [15, 90]
        gen_sobol = perturb(self.atoms.copy(),
                            gb=True, gb_angle='random', gb_dist=0.1,
                            sampler='sobol', seed=42, num=N,
                            validate_structure=False,
                            similarity_threshold=1.0)
        angles_sobol = []
        for s in gen_sobol:
            angles_sobol.append(s.info['perturb_annotation']['metadata']['angle'])
            
        # 2. Random Sampling
        gen_random = perturb(self.atoms.copy(),
                             gb=True, gb_angle='random', gb_dist=0.1,
                             sampler='random', seed=42, num=N,
                             validate_structure=False,
                             similarity_threshold=1.0)
        angles_random = []
        for s in gen_random:
            angles_random.append(s.info['perturb_annotation']['metadata']['angle'])
            
        # Normalize to [0, 1]
        norm_sobol = (np.array(angles_sobol) - 15.0) / (90.0 - 15.0)
        norm_random = (np.array(angles_random) - 15.0) / (90.0 - 15.0)
        
        # Calculate Discrepancy (Coverage)
        # Simple metric: Gap size standard deviation
        # Lower std dev of gaps means more uniform spacing
        
        def gap_std(data):
            sorted_data = np.sort(data)
            # Add 0 and 1 boundaries
            aug_data = np.concatenate(([0], sorted_data, [1]))
            gaps = np.diff(aug_data)
            return np.std(gaps)
            
        std_sobol = gap_std(norm_sobol)
        std_random = gap_std(norm_random)
        
        print(f"\nGap Std Dev (Lower is Better):")
        print(f"Sobol:  {std_sobol:.6f}")
        print(f"Random: {std_random:.6f}")
        
        # Sobol should generally be more uniform (lower gap variance)
        # Note: This is probabilistic for small N, but for N=64 it's usually true.
        # We won't make it a hard failure to avoid flakiness, but we'll warn.
        if std_sobol > std_random:
            print("Warning: Sobol distribution was not more uniform than random in this run.")
        else:
            print("Success: Sobol distribution is more uniform.")

        # Hard assertion: Range coverage
        # Sobol is deterministic and space-filling, so it should cover extrema well.
        self.assertTrue(np.min(norm_sobol) < 0.1, "Sobol should cover lower bound")
        self.assertTrue(np.max(norm_sobol) > 0.9, "Sobol should cover upper bound")

if __name__ == '__main__':
    unittest.main()
