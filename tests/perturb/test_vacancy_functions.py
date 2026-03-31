import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from ase import Atoms
from NepTrain.core.perturb.vacancy import (
    insert_vacancy,
    count_vacancies,
    remove_vacancy,
    replace_vacancy,
    _filter_vacancies_for_export,
    generate_vacancies
)

class TestVacancyFunctions(unittest.TestCase):

    def setUp(self):
        # Create a simple structure: Fe, O, Fe, O
        # Explicit symbols to ensure order
        self.atoms = Atoms(symbols=['Fe', 'O', 'Fe', 'O'], positions=[[0,0,0], [1,0,0], [2,0,0], [3,0,0]])

    def test_insert_vacancy(self):
        # Insert at beginning
        s1 = insert_vacancy(self.atoms, 0)
        self.assertEqual(len(s1), 5)
        self.assertEqual(s1[0].symbol, 'X')
        self.assertEqual(s1[1].symbol, 'Fe')
        
        # Insert at end
        s2 = insert_vacancy(self.atoms, 4)
        self.assertEqual(len(s2), 5)
        self.assertEqual(s2[4].symbol, 'X')
        
        # Insert in middle
        s3 = insert_vacancy(self.atoms, 2)
        self.assertEqual(len(s3), 5)
        self.assertEqual(s3[2].symbol, 'X')
        self.assertEqual(s3[3].symbol, 'Fe') # Originally at 2, now at 3? No.
        # Original: 0:Fe, 1:O, 2:Fe, 3:O
        # Insert at 2: 0:Fe, 1:O, 2:X, 3:Fe, 4:O
        self.assertEqual(s3[3].symbol, 'Fe') 

        # Invalid index
        with self.assertRaises(IndexError):
            insert_vacancy(self.atoms, 10)
        with self.assertRaises(ValueError):
            insert_vacancy(self.atoms, -1)

    def test_count_vacancies(self):
        self.assertEqual(count_vacancies(self.atoms), 0)
        s = insert_vacancy(self.atoms, 0)
        self.assertEqual(count_vacancies(s), 1)
        s = insert_vacancy(s, 0)
        self.assertEqual(count_vacancies(s), 2)

    def test_remove_vacancy(self):
        s = insert_vacancy(self.atoms, 1) # Fe, X, O, Fe, O
        self.assertEqual(s[1].symbol, 'X')
        
        s_removed = remove_vacancy(s, 1)
        self.assertEqual(len(s_removed), 4)
        self.assertEqual(s_removed[1].symbol, 'O')
        
        # Try removing non-vacancy
        with self.assertRaises(ValueError):
            remove_vacancy(self.atoms, 0)
            
        # Invalid index
        with self.assertRaises(IndexError):
            remove_vacancy(s, 10)

    def test_replace_vacancy(self):
        s = insert_vacancy(self.atoms, 2) # Fe, O, X, Fe, O
        s_replaced = replace_vacancy(s, 2, 'N')
        self.assertEqual(s_replaced[2].symbol, 'N')
        self.assertEqual(len(s_replaced), 5)
        
        # Try replacing non-vacancy
        with self.assertRaises(ValueError):
            replace_vacancy(self.atoms, 0, 'N')

    def test_filter_vacancies_for_export(self):
        s = insert_vacancy(self.atoms, 1)
        s = insert_vacancy(s, 3) 
        # Original: Fe, O, Fe, O
        # Insert at 1: Fe, X, O, Fe, O
        # Insert at 3: Fe, X, O, X, Fe, O
        
        self.assertEqual(count_vacancies(s), 2)
        
        filtered = _filter_vacancies_for_export(s)
        self.assertEqual(len(filtered), 4)
        self.assertEqual(count_vacancies(filtered), 0)
        self.assertEqual(filtered.get_chemical_formula(), 'Fe2O2')
        
        # Check that filtered is a copy
        s[0].position = [10,10,10]
        self.assertFalse(filtered[0].position[0] == 10)

    def test_edge_cases(self):
        # Empty structure
        empty = Atoms()
        s = insert_vacancy(empty, 0)
        self.assertEqual(len(s), 1)
        self.assertEqual(s[0].symbol, 'X')
        
        removed = remove_vacancy(s, 0)
        self.assertEqual(len(removed), 0)
        
        # Structure with only vacancies
        only_v = Atoms('X3', positions=[[0,0,0]]*3)
        filtered = _filter_vacancies_for_export(only_v)
        self.assertEqual(len(filtered), 0)

    def test_generate_vacancies(self):
        # Create Fe2O2 structure
        atoms = Atoms(symbols=['Fe', 'O', 'Fe', 'O'], positions=[[0,0,0], [1,0,0], [2,0,0], [3,0,0]])
        
        # Generate 1 vacancy in Fe
        s, metadata = generate_vacancies(atoms, elements=['Fe'], num_vacancies=1)
        
        # Check that one Fe is replaced by X
        symbols = s.get_chemical_symbols()
        self.assertEqual(symbols.count('X'), 1)
        self.assertEqual(symbols.count('Fe'), 1)
        self.assertEqual(symbols.count('O'), 2)
        
        # Check metadata
        self.assertEqual(len(metadata), 1)
        idx = list(metadata.keys())[0]
        self.assertEqual(metadata[idx]['original_symbol'], 'Fe')
        
        # Error: more vacancies than atoms
        with self.assertRaises(ValueError):
            generate_vacancies(atoms, ['Fe'], 3)
            
        # Error: element not found
        with self.assertRaises(ValueError):
            generate_vacancies(atoms, ['Si'], 1)

if __name__ == '__main__':
    unittest.main()
