
import numpy as np

def snap_angle_cubic_001(target_angle, tol=0.1):
    """
    Finds exact angle 2*arctan(n/m) close to target_angle.
    """
    best_angle = None
    best_diff = 1e9
    
    # Search n, m just for checking
    for m in range(1, 20):
        for n in range(1, m): # n < m for < 90 deg usually?
            # angle = 2 * arctan(n/m)
            rad = 2 * np.arctan(n/m)
            deg = np.degrees(rad)
            
            diff = abs(deg - target_angle)
            if diff < best_diff:
                best_diff = diff
                best_angle = deg
                
    if best_diff < tol:
        return best_angle
    return None

target = 36.87
snapped = snap_angle_cubic_001(target)
print(f"Target: {target}, Snapped: {snapped}, Diff: {abs(target - snapped) if snapped else 'N/A'}")

# Sigma 5 exact
exact = 2 * np.degrees(np.arctan(1/3))
print(f"Exact Sigma 5: {exact}")

# Test Sigma 13 (22.62 deg, 2*atan(1/5))
target13 = 22.62
snapped13 = snap_angle_cubic_001(target13)
print(f"Target 13: {target13}, Snapped: {snapped13}")
