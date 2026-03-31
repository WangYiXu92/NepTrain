import numpy as np

def get_rotation_matrix(axis, theta_deg):
    """
    Returns the 3x3 rotation matrix for a rotation of theta degrees about the given axis.
    """
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    
    theta = np.radians(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1 - c
    
    x, y, z = axis
    
    R = np.array([
        [t*x*x + c,   t*x*y - z*s, t*x*z + y*s],
        [t*x*y + z*s, t*y*y + c,   t*y*z - x*s],
        [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
    ])
    
    return R

def find_csl_basis(sigma, axis, angle_deg, limit=5, tolerance=1e-3):
    """
    Finds the CSL basis vectors for a given rotation.
    
    Args:
        sigma (int): The Sigma value (inverse of coincidence density).
        axis (list): Rotation axis [u, v, w].
        angle_deg (float): Rotation angle in degrees.
        limit (int): Search range for integer vectors [-limit, limit].
        tolerance (float): Tolerance for checking integer coordinates.
        
    Returns:
        M (3x3 numpy array): Columns are the CSL basis vectors u1, u2, u3 in the reference frame.
                             These vectors define the supercell for Grain A.
                             M is an integer matrix.
        M_prime (3x3 numpy array): Columns are the CSL basis vectors in the rotated frame (before rotation).
                                   M_prime = R.T @ M.
                                   M_prime is also an integer matrix (within tolerance).
    """
    R = get_rotation_matrix(axis, angle_deg)
    R_T = R.T
    
    valid_pairs = []
    
    # Grid search for integer vectors u such that R.T @ u is also integer
    rng = range(-limit, limit + 1)
    
    for x in rng:
        for y in rng:
            for z in rng:
                if x == 0 and y == 0 and z == 0:
                    continue
                
                u = np.array([x, y, z])
                v = R_T @ u
                
                # Check if v is close to integer
                v_rounded = np.round(v)
                if np.allclose(v, v_rounded, atol=tolerance):
                    v_int = v_rounded.astype(int)
                    # Check length preservation (exact CSL requirement)
                    if np.abs(np.linalg.norm(u) - np.linalg.norm(v_int)) < tolerance:
                        valid_pairs.append((u, v_int))
    
    if len(valid_pairs) < 3:
        # Try increasing limit if not enough vectors found
        if limit < 15:
            return find_csl_basis(sigma, axis, angle_deg, limit=limit+5, tolerance=tolerance)
        else:
            raise ValueError(f"Could not find enough CSL vectors for Sigma {sigma} with limit {limit}")

    # Sort pairs by length of u
    valid_pairs.sort(key=lambda p: np.linalg.norm(p[0]))
    
    import itertools
    
    # Try to find a triplet that forms a basis with determinant ~ k * Sigma
    # We want the smallest volume >= Sigma (usually exactly Sigma * integer)
    # Actually, for FCC, the volume of the CSL cell in terms of the cubic lattice parameter 'a'
    # is V_CSL = det(M) * a^3.
    # The primitive FCC cell volume is a^3 / 4.
    # So V_CSL = Sigma * V_prim = Sigma * a^3 / 4.
    # Thus det(M) should be Sigma / 4 * k (where k is integer).
    # Wait, det(M) is an integer. So Sigma must be a multiple of 4? No.
    # If Sigma=5, V_CSL = 5/4 a^3. det(M) = 1.25? No, M must be integer.
    # This implies that for Sigma 5, the CSL unit cell in the Cubic basis might be larger.
    # Standard CSL theory: The CSL is defined on the lattice.
    # If we work with the conventional cubic cell (basis I), M defines a supercell of the cubic cell.
    # The volume of this supercell is det(M) * a^3.
    # We want this to be a CSL for the FCC lattice.
    # The FCC lattice points are a subset of the Cubic lattice.
    # If M defines a CSL for Cubic, it is a candidate for FCC.
    # We need to ensure the density of FCC points in this cell matches.
    # For Sigma 5 (001), the area is 5 * area_primitive_2D.
    # In cubic 2D lattice, area is 1. Primitive FCC (001) area is 0.5.
    # So Sigma 5 area should be 2.5? Or 5 * 0.5 = 2.5.
    # If we find integer vectors, the area will be integer.
    # Example: [2, 1, 0] and [-1, 2, 0]. Cross product magnitude is 5.
    # This matches 5 * 1 (cubic area).
    # Is this Sigma 5 for FCC?
    # Sigma is the ratio of the volume of the coincidence site lattice cell to the volume of the crystal primitive cell?
    # No, Sigma is the inverse density of coincidence sites.
    # Volume(CSL) = Sigma * Volume(Primitive).
    # If we use integer vectors in Cubic frame, Volume = det(M).
    # Volume(Primitive FCC) = 0.25.
    # So det(M) = Sigma * 0.25 * k.
    # For Sigma=5, det(M) = 1.25 * k. Smallest integer is 5 (k=4).
    # So we expect det(M) = 5 for Sigma 5.
    
    # We prioritize vectors perpendicular to the rotation axis for the interface.
    axis_vec = np.array(axis, dtype=float)
    axis_vec /= np.linalg.norm(axis_vec)
    
    candidates = []
    
    # Take top 30 vectors to form combinations
    top_pairs = valid_pairs[:30]
    
    for triplet in itertools.combinations(top_pairs, 3):
        u_vecs = [p[0] for p in triplet]
        v_vecs = [p[1] for p in triplet]
        
        M = np.column_stack(u_vecs)
        det = np.linalg.det(M)
        
        if abs(det) < 1e-3:
            continue
            
        # Ensure right-handed system (positive determinant)
        if det < 0:
            # Swap first two columns
            u_vecs[0], u_vecs[1] = u_vecs[1], u_vecs[0]
            v_vecs[0], v_vecs[1] = v_vecs[1], v_vecs[0]
            M = np.column_stack(u_vecs)
            det = -det
            
        # Check volume scaling
        # We look for det(M) being a multiple of Sigma (or related)
        # For Sigma 5, we expect 5.
        if abs(det) < sigma - 0.1: # Too small
            continue
            
        # Orthogonality check with axis
        # Count how many vectors are perpendicular to axis
        perp_count = 0
        parallel_count = 0
        
        for u in u_vecs:
            u_norm = u / np.linalg.norm(u)
            proj = abs(np.dot(u_norm, axis_vec))
            if proj < 0.1: # Perpendicular
                perp_count += 1
            elif proj > 0.9: # Parallel
                parallel_count += 1
        
        # We prefer 2 perpendicular and 1 parallel
        score = perp_count * 10 + parallel_count * 5 - abs(abs(det) - sigma) # Penalize large volume
        
        candidates.append({
            'M': M,
            'M_prime': np.column_stack(v_vecs),
            'det': abs(det),
            'score': score,
            'perp': perp_count
        })
    
    if not candidates:
        if limit < 15:
            return find_csl_basis(sigma, axis, angle_deg, limit=limit+5, tolerance=tolerance)
        return None, None
        
    # Sort candidates by score
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    best = candidates[0]
    return best['M'], best['M_prime']
