import unittest
import numpy as np
from ase import Atoms
from NepTrain.core.perturb.vacancy import generate_vacancies

class TestVacancySobol(unittest.TestCase):
    def test_deterministic_vacancy(self):
        atoms = Atoms('Fe10')
        # We want to remove 3 atoms. Candidates are all 10.
        # If we provide rng_values, the selection should be deterministic based on sorting.
        # rng_values: assign low values to specific indices to force their selection.
        # e.g. indices 0, 5, 9 get low values.
        
        # We need 10 random values for 10 candidates
        rng = np.ones(10) * 0.9 # High values
        rng[0] = 0.1
        rng[5] = 0.2
        rng[9] = 0.3
        
        # We expect indices 0, 5, 9 to be selected for vacancy (marked 'X')
        # because they have the smallest random values (if we implement sort-based selection)
        new_atoms, _ = generate_vacancies(atoms, ['Fe'], num_vacancies=3, mode='random', rng_values=rng)
        
        symbols = new_atoms.get_chemical_symbols()
        self.assertEqual(symbols[0], 'X')
        self.assertEqual(symbols[5], 'X')
        self.assertEqual(symbols[9], 'X')
        self.assertEqual(symbols.count('X'), 3)
