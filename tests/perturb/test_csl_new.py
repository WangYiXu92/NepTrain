
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from NepTrain.core.perturb import csl_core

def test_csl_general():
    print("Testing General CSL Generation...")
    
    # Test Sigma 5 [001]
    print("\n--- Sigma 5 [001] ---")
    data = csl_core.get_csl_data([0,0,1], max_sigma=5)
    for d in data:
        print(d)
    
    angles = [d['angle'] for d in data if d['sigma'] == 5]
    assert any(abs(a - 36.87) < 0.1 for a in angles), "Missing 36.87"
    assert any(abs(a - 53.13) < 0.1 for a in angles), "Missing 53.13"
    
    # Test Sigma 3 [111] -> 60 deg
    print("\n--- Sigma 3 [111] ---")
    data = csl_core.get_csl_data([1,1,1], max_sigma=3)
    for d in data:
        print(d)
    angles = [d['angle'] for d in data if d['sigma'] == 3]
    assert any(abs(a - 60.0) < 0.1 for a in angles), "Missing 60.0"

    # Test Random Axis
    print("\n--- Random Axis Sample ---")
    axis = csl_core.sample_random_axis()
    print(f"Axis: {axis}")
    data = csl_core.get_csl_data(axis, max_sigma=20)
    print(f"Found {len(data)} CSLs")
    if len(data) > 0:
        print(data[0])

if __name__ == "__main__":
    test_csl_general()
