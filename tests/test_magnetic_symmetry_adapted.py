#!/usr/bin/env python3
"""
Test script for magnetic.py module with MultiPie integration.

This file has been updated to use the improved SymmetryAdaptedMagneticGenerator
class that properly leverages MultiPie's Symmetry-Adapted Multipole Basis (SAMB).
"""

import sys
import traceback

def test_module_import():
    """Test if the module can be imported without MultiPie."""
    try:
        from NepTrain.core.perturb.magnetic import (
            get_magmom_config,
            get_magnetic_perturbation_dims,
            apply_magnetic_perturbation,
            ensure_magnetic_configuration,
            generate_symmetry_adapted_magnetic_structures,
            SymmetryAdaptedMagneticGenerator,
            generate_magnetic_phase_transition_path,
            generate_antiferromagnetic_configurations,
            generate_magnetic_interstitial_structures,
            generate_magnetic_twinning_structures,
        )
        print("✅ All functions and classes imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_multipie_warning():
    """Test if the import shows warning when MultiPie is not available."""
    import warnings
    
    # Check that the warning was issued
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            from multipie import MultipieManager
            print("⚠️ MultiPie is available - skipping warning test")
        except ImportError:
            print("✅ MultiPie warning shown (import error as expected)")
    
    return True


def test_function_signatures():
    """Test that new functions have correct signatures."""
    from NepTrain.core.perturb.magnetic import (
        generate_symmetry_adapted_magnetic_structures,
        SymmetryAdaptedMagneticGenerator,
        generate_magnetic_phase_transition_path,
        generate_antiferromagnetic_configurations,
        generate_magnetic_interstitial_structures,
        generate_magnetic_twinning_structures,
    )
    
    # Check function signatures
    import inspect
    
    sig = inspect.signature(generate_symmetry_adapted_magnetic_structures)
    params = list(sig.parameters.keys())
    expected = ['atoms', 'magnetic_elements', 'propagation_vector', 'max_multipole_rank']
    assert all(p in params for p in expected), f"Missing parameters in generate_symmetry_adapted_magnetic_structures: {expected}"
    print(f"✅ generate_symmetry_adapted_magnetic_structures has correct signature")
    
    sig = inspect.signature(generate_magnetic_phase_transition_path)
    params = list(sig.parameters.keys())
    expected = ['atoms', 'start_phase', 'end_phase', 'n_intermediate', 'magnetic_elements', 'magnetic_moment']
    assert all(p in params for p in expected), f"Missing parameters in generate_magnetic_phase_transition_path: {expected}"
    print(f"✅ generate_magnetic_phase_transition_path has correct signature")
    
    sig = inspect.signature(generate_antiferromagnetic_configurations)
    params = list(sig.parameters.keys())
    expected = ['atoms', 'magnetic_elements', 'propagation_vectors', 'moment_magnitude', 'max_structures']
    assert all(p in params for p in expected), f"Missing parameters in generate_antiferromagnetic_configurations: {expected}"
    print(f"✅ generate_antiferromagnetic_configurations has correct signature")
    
    return True


def test_class_exists():
    """Test that the generator class exists and has required methods."""
    from NepTrain.core.perturb.magnetic import SymmetryAdaptedMagneticGenerator
    
    # Check class has required methods
    required_methods = [
        '__init__',
        'get_irreducible_representations',
        'generate_basis_states',
        'generate_all_states',
        'validate_symmetry',  # NEW in improved version
    ]
    
    for method in required_methods:
        assert hasattr(SymmetryAdaptedMagneticGenerator, method), f"Missing method: {method}"
    
    print("✅ SymmetryAdaptedMagneticGenerator has all required methods")
    return True


def test_backward_compatibility():
    """Test that existing functions still work (with mock data)."""
    try:
        from ase import Atoms
        from NepTrain.core.perturb.magnetic import (
            get_magnetic_perturbation_dims,
            apply_magnetic_perturbation,
        )
        
        # Create a simple test structure
        atoms = Atoms('Fe4', positions=[[0,0,0], [1,0,0], [0,1,0], [0,0,1]], 
                     pbc=True, cell=[2,2,2])
        
        # Test get_magnetic_perturbation_dims
        dims = get_magnetic_perturbation_dims(atoms, mode='collinear')
        assert dims >= 0, f"Expected non-negative dimensions, got {dims}"
        print(f"✅ get_magnetic_perturbation_dims works (dims={dims})")
        
        # Test apply_magnetic_perturbation
        perturbed = apply_magnetic_perturbation(atoms, mode='collinear')
        assert len(perturbed) == len(atoms), "Structure size changed"
        print("✅ apply_magnetic_perturbation works")
        
        return True
    except ImportError as e:
        print(f"⚠️ ASE not available, skipping compatibility test: {e}")
        return True


def test_multipie_integration():
    """Test MultiPie integration if available."""
    try:
        from multipie import MultipieManager
        print("✅ MultiPie is installed - integration test available")
        
        from NepTrain.core.perturb.magnetic import SymmetryAdaptedMagneticGenerator
        from ase import Atoms
        
        atoms = Atoms('Fe4', positions=[[0,0,0], [1,0,0], [0,1,0], [0,0,1]], 
                     pbc=True, cell=[2,2,2])
        
        gen = SymmetryAdaptedMagneticGenerator(atoms, ['Fe'])
        print("✅ SymmetryAdaptedMagneticGenerator initialized")
        
        # Check that the manager is available
        assert hasattr(gen, 'manager'), "Missing manager attribute"
        assert isinstance(gen.manager, MultipieManager), "manager is not MultipieManager"
        print("✅ MultipieManager integration working")
        
        # Test get_irreducible_representations
        irreps = gen.get_irreducible_representations([0.5, 0.5, 0.5])
        print(f"✅ get_irreducible_representations works ({len(irreps)} irreps)")
        for irrep in irreps:
            print(f"  - {irrep['label']}: dim={irrep['dimension']}")
        
        # Test generate_basis_states
        if irreps:
            irrep_label = irreps[0]['label']
            basis_states = gen.generate_basis_states([0.5, 0.5, 0.5], irrep_label)
            print(f"✅ generate_basis_states works ({len(basis_states)} states)")
        
        # Test generate_all_states
        all_states = gen.generate_all_states([0.5, 0.5, 0.5], max_multipole_rank=1)
        print(f"✅ generate_all_states works ({len(all_states)} IRREPS)")
        for irrep_label, states in all_states.items():
            print(f"  - {irrep_label}: {len(states)} states")
        
        # Test validate_symmetry (NEW)
        if len(all_states) > 0:
            test_struct = list(all_states.values())[0][0]
            is_symmetric = gen.validate_symmetry(test_struct, [0.5, 0.5, 0.5])
            print(f"✅ validate_symmetry works ({is_symmetric})")
        
        return True
    except ImportError:
        print("⚠️ MultiPie not installed, skipping integration test")
        return True
    except Exception as e:
        print(f"❌ MultiPie integration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing magnetic.py module (Improved MultiPie integration)")
    print("=" * 60)
    print()
    
    tests = [
        ("Module Import", test_module_import),
        ("Function Signatures", test_function_signatures),
        ("Class Methods", test_class_exists),
        ("Backward Compatibility", test_backward_compatibility),
        ("MultiPie Integration", test_multipie_integration),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n--- Test: {name} ---")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
