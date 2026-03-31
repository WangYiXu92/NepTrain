import unittest
import os
from NepTrain.core.perturb.run import run_perturb

class TestSobolFull(unittest.TestCase):
    def test_full_chain(self):
        # Run perturb with vacancy and shuffle using Sobol
        # Verify it runs without error and returns results
        
        # Create a dummy file with reasonable distances
        # Fe radius ~1.32. Min dist ~1.85. 
        # Let's put them 2.5A apart.
        with open("test_full_sobol.xyz", "w") as f:
            f.write("4\nLattice=\"10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0\" Properties=species:S:1:pos:R:3\nFe 0.0 0.0 0.0\nFe 2.5 0.0 0.0\nNi 0.0 2.5 0.0\nNi 2.5 2.5 0.0\n")
            
        gen = run_perturb("test_full_sobol.xyz", 
                      num=2, 
                      sampler='sobol',
                      vac_elements='Fe', vac_num=1,
                      shuffle_elements='Ni',
                      skip_normal=True) # Skip normal to isolate logic
                      
        results = list(gen)
        # Should succeed
        self.assertTrue(len(results) > 0)
        
        # Cleanup
        if os.path.exists("test_full_sobol.xyz"):
            os.remove("test_full_sobol.xyz")
