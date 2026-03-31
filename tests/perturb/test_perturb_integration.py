import unittest
import os
import shutil
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from ase import Atoms
from NepTrain.core.perturb.run import perturb
from NepTrain.core.perturb.vacancy import generate_vacancies
from unittest.mock import MagicMock

# We need to bypass the decorator or mock it because iter_path_to_atoms expects a path.
# However, perturb is decorated. 
# We can access the original function using `perturb.__wrapped__` if it was wrapped with functools.wraps,
# but the custom decorator might not have used it.
# Let's inspect utils.py:
#   def iter_path_to_atoms(...):
#       def decorator(func):
#           def wrapper(...): ...
#           return wrapper
#       return decorator
# It does NOT use functools.wraps. So __wrapped__ might not be available.
# But we can import `perturb` from `NepTrain.core.perturb.run` and if it is the wrapper,
# we might need to redefine it or mock the decorator.

# Actually, the perturb function defined in run.py is decorated.
# So `perturb` IS the wrapper.
# The wrapper expects `path` as first argument.
# But our test passes `atoms`.
# The wrapper tries `path.is_dir()`.
# So we must pass a file path to `perturb`.

class TestPerturbIntegration(unittest.TestCase):

    def setUp(self):
        # Create a dummy structure file
        # Use larger spacing to pass adjust_reasonable
        self.atoms = Atoms('Fe2O2', positions=[[0,0,0], [2,0,0], [4,0,0], [6,0,0]])
        self.atoms.set_cell([10, 10, 10])
        self.atoms.set_pbc(True)
        if not os.path.exists("test_perturb_gen"):
            os.makedirs("test_perturb_gen")
        self.atoms.write("test_perturb_gen/structure.xyz")
        
    def tearDown(self):
        if os.path.exists("test_perturb_gen"):
            shutil.rmtree("test_perturb_gen")

    def test_perturb_with_vacancy(self):
        """Test that perturb function handles vacancy arguments correctly."""
        # Pass atoms object
        results = perturb(
            self.atoms, 
            num=2, 
            vac_elements='Fe', 
            vac_num=1
        )
        results = list(results)
        
        # perturb returns a list/generator of Atoms objects.
        self.assertEqual(len(results), 2)
        
        for s in results:
            # Original: 4 atoms (Fe2O2)
            # Vacancy: 1 Fe replaced by X, then filtered out.
            # So should have 3 atoms.
            self.assertEqual(len(s), 3)
            
            # Symbols should not contain 'X' (filtered)
            self.assertNotIn('X', s.get_chemical_symbols())
            
            # Should have 1 Fe and 2 O
            syms = s.get_chemical_symbols()
            self.assertEqual(syms.count('Fe'), 1)
            self.assertEqual(syms.count('O'), 2)
            
    def test_perturb_without_vacancy(self):
        """Test standard perturb still works."""
        results = perturb(self.atoms, num=2, min_distance=0.0)
        results = list(results)
        self.assertEqual(len(results), 2)
        for s in results:
            self.assertEqual(len(s), 4)
            self.assertEqual(s.get_chemical_formula(), 'Fe2O2')

if __name__ == '__main__':
    unittest.main()
