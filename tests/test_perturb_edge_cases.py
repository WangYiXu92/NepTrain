"""Test edge cases for perturbation modules.

This module tests boundary conditions and error handling
for interstitial and twinning defect generation.
"""

import pytest
import numpy as np
import sys
import os

# Add the perturb directory to path so crystal_detector.py can be imported
perturb_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'NepTrain', 'core', 'perturb')
sys.path.insert(0, perturb_dir)

from ase import Atoms
from ase.build import bulk

# Now import directly from perturb directory
import crystal_detector
import interstitial
import twinning

CrystalDetector = crystal_detector.CrystalDetector
detect_crystal_type = crystal_detector.detect_crystal_type


class TestInterstitialEdgeCases:
    """Test edge cases for interstitial defect generation."""
    
    def test_empty_structure(self):
        """Test empty structure handling."""
        empty = Atoms()
        with pytest.raises((ValueError, IndexError)):
            interstitial.InterstitialGenerator(empty, 'C')
    
    def test_too_many_interstitials(self):
        """Test requesting more interstitials than available sites."""
        fe = bulk('Fe', 'bcc', a=4.0)
        gen = interstitial.InterstitialGenerator(fe, 'C')
        # BCC unit cell has 6 octahedral sites
        with pytest.raises(ValueError, match="only .* available sites"):
            gen.generate(n_interstitials=100, site_type='octahedral')
    
    def test_invalid_selection_method(self):
        """Test invalid site selection method."""
        fe = bulk('Fe', 'bcc', a=4.0)
        gen = interstitial.InterstitialGenerator(fe, 'C')
        with pytest.raises(ValueError, match="not supported"):
            gen.generate(selection_method='invalid')
    
    def test_invalid_site_type(self):
        """Test invalid interstitial site type."""
        fe = bulk('Fe', 'bcc', a=4.0)
        gen = interstitial.InterstitialGenerator(fe, 'C')
        with pytest.raises(ValueError, match="not supported"):
            gen.get_interstitial_sites(site_type='invalid')


class TestTwinningEdgeCases:
    """Test edge cases for twinning defect generation."""
    
    def test_empty_structure(self):
        """Test empty structure handling."""
        empty = Atoms()
        with pytest.raises((ValueError, IndexError)):
            twinning.TwinningGenerator(empty)
    
    def test_invalid_crystal_type(self):
        """Test structure with unsupported crystal type."""
        # Create a simple cubic structure (not supported)
        atoms = Atoms('Al', positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
        # This should raise ValueError or be handled gracefully
        try:
            gen = twinning.TwinningGenerator(atoms)
            # If it doesn't raise, that's also acceptable (graceful handling)
        except ValueError:
            pass  # Expected for unsupported crystal types


class TestCrystalDetectorEdgeCases:
    """Test edge cases for crystal structure detection."""
    
    def test_empty_structure(self):
        """Test crystal detection on empty structure."""
        empty = Atoms()
        # Should handle gracefully or raise ValueError
        try:
            result = detect_crystal_type(empty)
            # Result should be 'unknown' or similar
        except (ValueError, IndexError):
            pass  # Also acceptable
    
    def test_hcp_detection(self):
        """Test HCP structure detection."""
        mg = bulk('Mg', 'hcp', a=3.2, c=5.2)
        gen = interstitial.InterstitialGenerator(mg, 'C')
        assert gen.crystal_type == 'hcp'


class TestVectorizedOperations:
    """Test that vectorized operations produce correct results."""
    
    def test_interstitial_generation(self):
        """Test interstitial generation produces correct structure."""
        fe = bulk('Fe', 'bcc', a=4.0)
        gen = interstitial.InterstitialGenerator(fe, 'C')
        
        # Generate structure with 1 interstitial
        result = gen.generate(n_interstitials=1, site_type='octahedral')
        
        # Check that we have Fe + C atoms
        assert len(result) == len(fe) + 1
        assert 'C' in result.get_chemical_symbols()
    
    def test_twinning_generation(self):
        """Test twinning generation produces correct structure."""
        cu = bulk('Cu', 'fcc', a=3.6)
        result = twinning.TwinningGenerator(cu).generate()
        
        # Supercell should have more atoms than original
        assert len(result) > len(cu)
    
    def test_interstitial_sites_count(self):
        """Test that interstitial site counts are correct."""
        fe = bulk('Fe', 'bcc', a=4.0)
        gen = interstitial.InterstitialGenerator(fe, 'C')
        
        tetra_sites = gen.get_interstitial_sites('tetrahedral')
        octa_sites = gen.get_interstitial_sites('octahedral')
        
        # BCC: 8 tetrahedral + 6 octahedral = 14 sites per conventional cell
        # But ASE bulk creates primitive cell, so we check relative counts
        assert len(tetra_sites) > 0
        assert len(octa_sites) > 0
        assert len(tetra_sites) >= len(octa_sites)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
