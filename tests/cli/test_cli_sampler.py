import unittest
import argparse
from NepTrain.cli.cli import build_perturb

class TestCLISampler(unittest.TestCase):
    def test_sampler_args(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        build_perturb(subparsers)
        
        # Test default
        args = parser.parse_args(['perturb', 'dummy.xyz'])
        self.assertEqual(args.sampler, 'random')
        self.assertTrue(args.scramble)
        
        # Test sobol
        args = parser.parse_args(['perturb', 'dummy.xyz', '--sampler', 'sobol'])
        self.assertEqual(args.sampler, 'sobol')
        
        # Test scramble
        args = parser.parse_args(['perturb', 'dummy.xyz', '--no-scramble'])
        self.assertFalse(args.scramble)

if __name__ == '__main__':
    unittest.main()
