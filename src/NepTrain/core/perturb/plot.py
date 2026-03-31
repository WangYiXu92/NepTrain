
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from ase.data.colors import jmol_colors
from ase.data import atomic_numbers

def _get_dislocation_marker_path(angle_deg=0):
    """
    Create a matplotlib Path for the edge dislocation symbol (upside-down T).
    Stem points in the direction of 'angle_deg' (0 = pointed UP/+Y).
    """
    from matplotlib.path import Path
    # Base T shape (upside down perp symbol): 
    # horizontal bar at bottom, vertical stem pointing UP.
    verts = [
        (-0.5, 0.0), (0.5, 0.0), # horizontal bar
        (0.0, 0.0), (0.0, 0.8)   # vertical stem
    ]
    codes = [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO]
    
    # Rotate vertices around (0,0)
    rad = np.radians(angle_deg)
    rot_mat = np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad),  np.cos(rad)]
    ])
    rotated_verts = [np.dot(rot_mat, v) for v in verts]
    return Path(rotated_verts, codes)

def _get_depth_style(z_val, z_min, z_max, min_alpha=0.15, base_color='gray'):
    """
    Calculate color and alpha based on depth (Z).
    Shallower (High Z) -> Opaque
    Deeper (Low Z) -> Transparent (fades to min_alpha)
    Returns: (final_rgba_color, alpha_value)
    """
    z_span = z_max - z_min
    if z_span < 1e-3: z_span = 1.0
    
    depth_norm = (z_val - z_min) / z_span
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    
    # Linear interpolation: Low Z (0) -> min_alpha, High Z (1) -> 1.0
    alpha = min_alpha + depth_norm * (1.0 - min_alpha)
    
    # Mix with white background
    bg_color = np.array([1.0, 1.0, 1.0])
    
    # Ensure base_color is RGB array
    if isinstance(base_color, str):
        from matplotlib.colors import to_rgb
        c_rgb = np.array(to_rgb(base_color))
    else:
        c_rgb = np.array(base_color)[:3] # Ignore alpha if present
        
    final_color = c_rgb * alpha + bg_color * (1.0 - alpha)
    
    return final_color, 1.0

def _get_axes_indices(projection):
    """
    Returns (idx_x, idx_y, idx_depth) for the given projection string.
    projection: 'xy', 'xz', 'yz' (case insensitive)
    """
    proj = projection.lower()
    if proj == 'xy':
        return 0, 1, 2
    elif proj == 'xz':
        return 0, 2, 1
    elif proj == 'yz':
        return 1, 2, 0
    else:
        raise ValueError(f"Unknown projection: {projection}")

def _get_axis_labels(projection):
    """Returns (xlabel, ylabel) based on projection."""
    proj = projection.lower()
    if proj == 'xy': return 'x (Å)', 'y (Å)'
    if proj == 'xz': return 'x (Å)', 'z (Å)'
    if proj == 'yz': return 'y (Å)', 'z (Å)'
    return 'x', 'y'

def _get_grain_z_cut(atoms):
    """Extract the Z-cut height from perturb_annotation. Returns None if not applicable."""
    if 'perturb_annotation' not in atoms.info:
        return None
    info = atoms.info['perturb_annotation']
    ptype = info.get('type')
    meta = info.get('metadata', {})
    if ptype not in ['grain_boundary', 'twinning']:
        return None
    z_cut = meta.get('plane_height')
    if z_cut is None:
        z_frac = meta.get('z_frac')
        if z_frac is not None:
            z_cut = abs(atoms.cell[2, 2]) * z_frac
    return z_cut


def _get_atom_colors_for_grains(atoms, base_colors):
    """
    Tint atoms by grain: lower grain -> blue tint, upper grain -> red/orange tint.
    Preserves element identity via blending.
    """
    z_cut = _get_grain_z_cut(atoms)
    if z_cut is None:
        return base_colors

    from matplotlib.colors import to_rgb
    LOWER_TINT = np.array([0.3, 0.5, 0.85])  # Blue
    UPPER_TINT = np.array([0.85, 0.35, 0.25])  # Red/orange
    BLEND = 0.4  # 40% tint, 60% original

    pos = atoms.get_positions()
    z_vals = pos[:, 2]

    new_colors = []
    for i, color in enumerate(base_colors):
        c = np.array(to_rgb(color)) if isinstance(color, str) else np.array(color)[:3]
        tint = UPPER_TINT if z_vals[i] > z_cut else LOWER_TINT
        blended = c * (1 - BLEND) + tint * BLEND
        new_colors.append(np.clip(blended, 0, 1))

    return new_colors


def _highlight_interface_atoms(ax, atoms, projection='xy', tolerance=1.5, ring_radius=0.55):
    """
    Draw gold rings around atoms within ±tolerance of the GB interface plane.
    Returns a legend handle or None.
    """
    z_cut = _get_grain_z_cut(atoms)
    if z_cut is None:
        return None

    idx_x, idx_y, _ = _get_axes_indices(projection)
    pos = atoms.get_positions()
    z_vals = pos[:, 2]

    drawn = False
    for i in range(len(atoms)):
        if abs(z_vals[i] - z_cut) <= tolerance:
            ring = mpatches.Circle(
                (pos[i, idx_x], pos[i, idx_y]),
                radius=ring_radius,
                facecolor='none', edgecolor='gold',
                linewidth=1.5, linestyle='-', alpha=0.8, zorder=15
            )
            ax.add_patch(ring)
            drawn = True

    if drawn:
        return mpatches.Patch(facecolor='none', edgecolor='gold', linewidth=1.5, label='Interface atoms')
    return None

def plot_projection(ax, atoms, projection='xy', radii=0.4, depth_of_field=True, dof_min_alpha=0.15, **kwargs):
    """
    Refactored atom plotter supporting multiple projections.
    
    Args:
        projection (str): 'xy', 'xz', or 'yz'.
    """
    pos = atoms.get_positions()
    numbers = atoms.numbers
    cell = atoms.get_cell()
    
    idx_x, idx_y, idx_z = _get_axes_indices(projection)
    
    # Get coordinates for this projection
    x_vals = pos[:, idx_x]
    y_vals = pos[:, idx_y]
    z_vals = pos[:, idx_z] # Depth coordinate
    
    # Sort by depth for correct occlusion
    # We want shallow (high Z) on top, so draw deep (low Z) first
    # argsort sorts low to high, so iterating this order draws low Z first
    indices = np.argsort(z_vals)
    
    # Calculate depth range
    min_z, max_z = np.min(z_vals), np.max(z_vals)
    z_span = max_z - min_z
    if z_span < 1e-3: z_span = 1.0
    
    # Draw cell boundary (Projected)
    corners = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                corners.append(i*cell[0] + j*cell[1] + k*cell[2])
    corners = np.array(corners)
    
    # Project corners
    proj_corners_x = corners[:, idx_x]
    proj_corners_y = corners[:, idx_y]
    
    # Compute Hull of corners to draw boundary
    if len(corners) > 0:
        try:
            hull = ConvexHull(corners[:, [idx_x, idx_y]])
            hull_pts = corners[hull.vertices][:, [idx_x, idx_y]]
            # Close loop
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            ax.plot(hull_pts[:, 0], hull_pts[:, 1], '--k', alpha=0.3, zorder=1)
        except:
            pass
            
    # Prepare base colors (Jmol)
    base_colors = [jmol_colors[n] for n in numbers]
    
    # Apply Grain Tinting
    final_atom_colors = _get_atom_colors_for_grains(atoms, base_colors)
    
    # Draw atoms
    for i in indices:
        z_num = numbers[i]
        
        # Skip 'X' vacancies
        if z_num == 0: continue
        
        # Use tint-adjusted color
        color = final_atom_colors[i]
        
        # Depth Style
        alpha = 1.0
        final_color = color
        current_radius = radii
        
        z_curr = z_vals[i]
        
        if depth_of_field:
            final_color, alpha = _get_depth_style(z_curr, min_z, max_z, dof_min_alpha, color)
            
            # Size Scaling: Scale from 0.8 to 1.2 based on depth
            # depth_norm = 0 (deep) -> 0.8, 1 (shallow) -> 1.2
            depth_norm = (z_curr - min_z) / z_span
            depth_norm = np.clip(depth_norm, 0.0, 1.0)
            scale = 0.8 + 0.4 * depth_norm
            current_radius = radii * scale
        
        # Draw atom
        circle = mpatches.Circle((x_vals[i], y_vals[i]), radius=current_radius, 
                               facecolor=final_color, edgecolor='black', linewidth=0.5, zorder=2+depth_norm if depth_of_field else 2)
        ax.add_patch(circle)
        
    # Set limits
    margin = 1.0
    all_x = np.concatenate([x_vals, proj_corners_x])
    all_y = np.concatenate([y_vals, proj_corners_y])
    
    min_x, max_x = np.min(all_x), np.max(all_x)
    min_y, max_y = np.min(all_y), np.max(all_y)
    
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    ax.set_aspect('equal')
    
    return ax

def get_atom_handles(atoms):
    symbols = sorted(list(set(atoms.get_chemical_symbols())))
    handles = []
    for s in symbols:
        if s == 'X': continue 
        z = atomic_numbers[s]
        color = jmol_colors[z]
        handles.append(mpatches.Patch(color=color, label=s))
    
    # Check for grains to add "Upper Grain" legend?
    # Maybe too cluttered. The visual distinction is usually enough.
    return handles

def plot_mags(atoms, axis, projection='xy', depth_of_field=True, dof_min_alpha=0.15, **kwargs):
    idx_x, idx_y, idx_z = _get_axes_indices(projection)
    
    info = atoms.info.get('perturb_annotation', {})
    ptype = info.get('type', '')
    meta = info.get('metadata', {})
    is_magnetic = ptype.startswith('magnetic') or 'magnetic_mode' in meta
    
    if not is_magnetic: return None

    try:
        moms = atoms.get_initial_magnetic_moments()
    except Exception:
        return None
        
    if moms is None or np.all(np.abs(moms) < 1e-3): return None
        
    pos = atoms.get_positions()
    x_vals = pos[:, idx_x]
    y_vals = pos[:, idx_y]
    z_vals = pos[:, idx_z]
    min_z, max_z = np.min(z_vals), np.max(z_vals)
    
    def get_colors(indices, base_color):
        if not depth_of_field: return base_color
        colors = []
        for i in indices:
            c, _ = _get_depth_style(z_vals[i], min_z, max_z, dof_min_alpha, base_color)
            colors.append(c)
        return colors

    is_vector = (len(moms.shape) == 2 and moms.shape[1] == 3)
    mag_handle = None
    
    if not is_vector:
        # Scalar moments
        mag_axis = info.get('axis', [0, 0, 1])
        moms_vec = np.outer(moms, mag_axis)
        
        # Check if projected axis has length
        v_proj = np.array([mag_axis[idx_x], mag_axis[idx_y]])
        len_proj = np.linalg.norm(v_proj)
        
        if len_proj < 1e-3:
            # Perpendicular to view (Point)
            up_indices = np.where(moms > 1e-3)[0]
            if len(up_indices) > 0:
                c_up = get_colors(up_indices, 'red')
                axis.scatter(x_vals[up_indices], y_vals[up_indices], s=80, marker='^', c=c_up, zorder=10)
            
            down_indices = np.where(moms < -1e-3)[0]
            if len(down_indices) > 0:
                c_down = get_colors(down_indices, 'blue')
                axis.scatter(x_vals[down_indices], y_vals[down_indices], s=80, marker='v', c=c_down, zorder=10)
            
            mag_handle = mlines.Line2D([], [], color='red', marker='^', linestyle='None', markersize=10, label=f'Mag ({projection.upper()} \u22A5)')
        else:
            # Arrow in plane
            # Project moms_vec
            u = moms_vec[:, idx_x]
            v = moms_vec[:, idx_y]
            c_all = get_colors(range(len(pos)), 'red')
            
            # Avoid warnings for zero vectors in quiver
            m_norms = np.sqrt(u**2 + v**2)
            mask = m_norms > 1e-6
            if np.any(mask):
                axis.quiver(x_vals[mask], y_vals[mask], u[mask], v[mask], 
                            color=np.array(c_all)[mask], width=0.005, pivot='mid', zorder=10)
            mag_handle = mlines.Line2D([], [], color='red', marker=r'$\uparrow$', linestyle='None', markersize=10, label=f'Mag ({projection.upper()})')
            
    else:
        # Vector moments
        u = moms[:, idx_x]
        v = moms[:, idx_y]
        c_all = get_colors(range(len(pos)), 'red')
        
        m_norms = np.sqrt(u**2 + v**2)
        mask = m_norms > 1e-6
        if np.any(mask):
            axis.quiver(x_vals[mask], y_vals[mask], u[mask], v[mask], 
                        color=np.array(c_all)[mask], width=0.005, pivot='mid', zorder=10)
        mag_handle = mlines.Line2D([], [], color='red', marker=r'$\rightarrow$', linestyle='None', markersize=10, label='Mag Moment')
        
    return mag_handle

def plot_rigid_bodies(atoms, axis, projection='xy', depth_of_field=True, dof_min_alpha=0.15, **kwargs):
    if 'rigid_bodies' not in atoms.info: return None
    
    idx_x, idx_y, idx_z = _get_axes_indices(projection)
    rigid_list = atoms.info['rigid_bodies']
    positions = atoms.get_positions()
    
    z_all = positions[:, idx_z]
    min_z, max_z = np.min(z_all), np.max(z_all)
    drawn = False
    
    for indices in rigid_list:
        points_3d = positions[indices]
        points_proj = points_3d[:, [idx_x, idx_y]]
        avg_z = np.mean(points_3d[:, idx_z])
        
        base_cyan = np.array([0.0, 1.0, 1.0])
        white = np.array([1.0, 1.0, 1.0])
        eff_cyan = base_cyan * 0.2 + white * 0.8
        
        face_c = eff_cyan
        edge_c = 'blue'
        face_a = 1.0
        edge_a = 1.0
        
        if depth_of_field:
            face_c, _ = _get_depth_style(avg_z, min_z, max_z, dof_min_alpha, eff_cyan)
            edge_c, _ = _get_depth_style(avg_z, min_z, max_z, dof_min_alpha, 'blue')

        # Draw 2D Hull
        if len(points_proj) >= 3:
            try:
                hull = ConvexHull(points_proj)
                poly_pts = points_proj[hull.vertices]
                poly = Polygon(poly_pts, facecolor=face_c, alpha=face_a, edgecolor=edge_c, linestyle='--', zorder=5)
                axis.add_patch(poly)
                drawn = True
            except: pass
            
    if drawn:
        return mpatches.Patch(facecolor='cyan', alpha=0.2, edgecolor='blue', linestyle='--', label='Rigid Body')
    return None

def _wrap_point(point, atoms):
    if atoms.pbc.any():
        try:
            scaled = atoms.cell.scaled_positions(np.array([point]))
            scaled %= 1.0
            return atoms.cell.cartesian_positions(scaled)[0]
        except: return point
    return point

def plot_annotations(atoms, axis, projection='xy', depth_of_field=True, dof_min_alpha=0.15, **kwargs):
    if 'perturb_annotation' not in atoms.info: return []
    
    idx_x, idx_y, idx_z = _get_axes_indices(projection)
    handles = []
    
    info = atoms.info['perturb_annotation']
    ptype = info.get('type')
    meta = info.get('metadata', {})
    
    pos = atoms.get_positions()
    z_all = pos[:, idx_z]
    min_z, max_z = np.min(z_all), np.max(z_all)
    
    def get_val(key, default=None):
        return meta.get(key, info.get(key, default))
        
    def parse_vector(val, default=None):
        if val is None: return default
        if isinstance(val, (list, tuple, np.ndarray)): return np.array(val, dtype=float)
        if isinstance(val, str):
            try: return np.array([float(x) for x in val.split(',')], dtype=float)
            except: pass
        return default

    # VACANCY
    if ptype == 'vacancy':
        first = True
        for idx, data in meta.items():
            if not isinstance(data, dict): continue
            vac_pos = _wrap_point(data.get('position', [0,0,0]), atoms)
            px, py, pz = vac_pos[idx_x], vac_pos[idx_y], vac_pos[idx_z]
            
            edge_c = 'red'
            if depth_of_field:
                edge_c, _ = _get_depth_style(pz, min_z, max_z, dof_min_alpha, 'red')
                
            scale = 1.0
            if depth_of_field:
                 z_span = max_z - min_z if (max_z - min_z) > 1e-3 else 1.0
                 d_norm = np.clip((pz - min_z)/z_span, 0, 1)
                 scale = 0.8 + 0.4 * d_norm
            
            circle = mpatches.Circle((px, py), radius=0.4*scale, facecolor='none', edgecolor=edge_c, linestyle='--', linewidth=2, zorder=10)
            axis.add_patch(circle)
            
            if first:
                handles.append(mpatches.Patch(facecolor='none', edgecolor='red', linestyle='--', linewidth=2, label='Vacancy'))
                first = False

    # DISLOCATION & PLANAR
    elif ptype in ['dislocation', 'grain_boundary', 'twinning', 'stacking_fault']:
        if ptype == 'dislocation':
            sub_type = get_val('sub_type', 'edge')
            center = parse_vector(get_val('center'))
            line_axis = get_val('axis', 2)
            
            if center is not None:
                c_wrapped = _wrap_point(center, atoms)
                cx, cy = c_wrapped[idx_x], c_wrapped[idx_y]
                
                # Case 1: Core view (looking down the line)
                if idx_z == line_axis:
                    if sub_type == 'edge':
                        # Option B: Point towards extra half-plane.
                        # Standard is +Y in local coords for +b.
                        burgers = get_val('burgers', 1.0)
                        
                        # Handle vector burgers (e.g. [0.5, 0.5, 0])
                        if isinstance(burgers, (list, tuple, np.ndarray)):
                             # Use the projection of burgers vector onto the viewing plane
                             # or just the first non-zero component to determine "direction"
                             b_val = burgers[0] if abs(burgers[0]) > 1e-6 else burgers[1]
                        else:
                             b_val = burgers
                        
                        try:
                            b_val = float(b_val)
                        except (ValueError, TypeError):
                            b_val = 1.0
                             
                        angle_marker = 0 if b_val >= 0 else 180
                        marker = _get_dislocation_marker_path(angle_marker)
                        axis.plot(cx, cy, marker=marker, markersize=14, color='darkred', 
                                  markeredgewidth=2, zorder=25)
                        handles.append(mlines.Line2D([], [], color='darkred', marker=marker, 
                                                   linestyle='None', markersize=10, label='Edge Core'))
                    else:
                        # Screw core symbol: circle with dot
                        axis.plot(cx, cy, marker='o', markersize=10, color='darkred', mfc='white', mew=1.5, zorder=25)
                        axis.plot(cx, cy, marker='.', markersize=4, color='darkred', zorder=26)
                        handles.append(mlines.Line2D([], [], color='darkred', marker='o', mfc='white', label='Screw Core'))
                
                # Case 2: Side view (line visible)
                elif idx_x == line_axis:
                    axis.axhline(cy, color='purple', linestyle='--', linewidth=2, alpha=0.6, zorder=19)
                    handles.append(mlines.Line2D([], [], color='purple', linestyle='--', label='Dislocation Line'))
                elif idx_y == line_axis:
                    axis.axvline(cx, color='purple', linestyle='--', linewidth=2, alpha=0.6, zorder=19)
                    handles.append(mlines.Line2D([], [], color='purple', linestyle='--', label='Dislocation Line'))

        # Add Interface Line for planar defects
        if ptype in ['grain_boundary', 'twinning', 'stacking_fault']:
            plane_height = get_val('plane_height')
            if plane_height is None and 'z_frac' in meta:
                plane_height = abs(atoms.cell[2, 2]) * meta['z_frac']
                
            if plane_height is not None:
                if idx_y == 2:
                    xlim = axis.get_xlim()
                    axis.hlines(plane_height, xlim[0], xlim[1], colors='k', linestyles='--', label='Interface')
                    handles.append(mlines.Line2D([], [], color='k', linestyle='--', label='Interface'))
        
        # Enriched text box with metadata
        lines = [f"{ptype.replace('_', ' ').title()}"]
        
        # Sigma value
        sigma = get_val('sigma')
        if sigma and sigma != 'unknown':
            lines[0] += f" (Σ{sigma})"
        
        # Rotation axis
        gb_axis = get_val('axis')
        if gb_axis is not None:
            if isinstance(gb_axis, (list, tuple, np.ndarray)):
                ax_str = ','.join(str(int(x)) if float(x) == int(float(x)) else f'{x:.1f}' for x in gb_axis)
            else:
                ax_str = str(gb_axis)
            lines.append(f"Axis: [{ax_str}]")
        
        # Rotation angle
        angle = get_val('angle')
        if angle is not None:
            lines.append(f"Angle: {float(angle):.2f}°")
        
        # Translation
        trans = get_val('translation')
        if trans is not None:
            if isinstance(trans, (list, tuple, np.ndarray)):
                t_str = ', '.join(f'{float(t):.2f}' for t in trans)
                lines.append(f"Trans: ({t_str})")
        
        # Atom count and volume
        lines.append(f"N={len(atoms)}, V={atoms.get_volume():.0f} ų")
        
        axis.text(0.02, 0.98, "\n".join(lines), transform=axis.transAxes, 
                  fontsize=7, verticalalignment='top', family='monospace',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7), zorder=20)

    return handles


def plot_z_density(ax, atoms, n_bins=40):
    """Plot a horizontal histogram of atom Z-coordinates with interface line."""
    pos = atoms.get_positions()
    z_vals = pos[:, 2]
    
    z_min_cell = 0.0
    z_max_cell = abs(atoms.cell[2, 2])
    
    bins = np.linspace(z_min_cell, z_max_cell, n_bins + 1)
    counts, edges = np.histogram(z_vals, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    
    ax.barh(centers, counts, height=(edges[1] - edges[0]) * 0.9,
            color='steelblue', alpha=0.7, edgecolor='navy', linewidth=0.5)
    
    # Draw interface line
    z_cut = _get_grain_z_cut(atoms)
    
    # Check for other planes (Stacking Fault, Surface) if GB z_cut is not found
    if z_cut is None and 'perturb_annotation' in atoms.info:
        ann = atoms.info['perturb_annotation']
        meta = ann.get('metadata', {})
        ptype = ann.get('type')
        if ptype == 'stacking_fault':
            z_cut = meta.get('plane_height')
        elif ptype == 'surface':
            # Surface usually has a vacuum. Interface is at the slab edge.
            # But let's just show the density for now.
            pass
        elif ptype == 'dislocation':
            center = meta.get('center')
            if center is not None: z_cut = center[2] # Z-coordinate of core
            
    if z_cut is not None:
        ax.axhline(z_cut, color='red', linestyle='--', linewidth=1.5, label='Defect Plane')
    
    ax.set_ylabel('Z (Å)')
    ax.set_xlabel('Count')
    ax.set_title('Z-Density', fontsize=9)
    ax.set_ylim(z_min_cell, z_max_cell)
    if z_cut is not None:
        ax.legend(fontsize=7, loc='upper right')

def smart_wrap(atoms):
    original_pbc = atoms.pbc
    atoms.pbc = [True, True, True]
    cell = atoms.get_cell()
    inv_cell = np.linalg.inv(cell)
    positions = atoms.get_positions()
    
    handled_indices = set()
    if 'rigid_bodies' in atoms.info:
        for indices in atoms.info['rigid_bodies']:
            body_pos = positions[indices]
            com = np.mean(body_pos, axis=0)
            frac_com = np.dot(com, inv_cell)
            shift_int = np.floor(frac_com)
            shift_vec = np.dot(shift_int, cell)
            positions[indices] -= shift_vec
            handled_indices.update(indices)
            
    all_indices = set(range(len(atoms)))
    remaining = list(all_indices - handled_indices)
    if remaining:
        rem_pos = positions[remaining]
        frac_rem = np.dot(rem_pos, inv_cell)
        shift_rem = np.floor(frac_rem)
        positions[remaining] -= np.dot(shift_rem, cell)
        
    atoms.set_positions(positions)
    atoms.pbc = original_pbc
    return atoms

def _has_planar_defect(atoms):
    """Check if atoms have a defect that benefits from Z-density visualization."""
    if 'perturb_annotation' not in atoms.info:
        return False
    ptype = atoms.info['perturb_annotation'].get('type')
    return ptype in ['grain_boundary', 'twinning', 'stacking_fault', 'surface', 'dislocation']


def plot_comparison(original, perturbed, filename, intermediate=None, depth_of_field=True, dof_min_alpha=0.15, theme='jmol'):
    """
    Generate a comparison plot:
    - 2x3 (Original, Perturbed) if intermediate is None
    - 3x3 (Original, Defect Only, Perturbed) if intermediate is provided
    
    Cols: XY, XZ, YZ
    Optional 4th Col: Z-Density (if planar defect present)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Detect if we need density column
    # Check both perturbed and intermediate for planar defects
    show_density = _has_planar_defect(perturbed) or (intermediate and _has_planar_defect(intermediate))
    
    n_cols = 4 if show_density else 3
    n_rows = 3 if intermediate else 2

    logger.debug(f"plot_comparison for {filename}")
    logger.debug(f"intermediate present? {bool(intermediate)}")
    if intermediate:
        logger.debug(f"intermediate type: {type(intermediate)}, len={len(intermediate)}")
    logger.debug(f"n_rows={n_rows}, n_cols={n_cols}")
    
    width_ratios = [1, 1, 1, 0.35] if show_density else [1, 1, 1]
    
    # Adjust figure height based on rows
    fig_height = 5 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18 + (3 if show_density else 0), fig_height),
                             gridspec_kw={'width_ratios': width_ratios})
    
    # Wrap structures
    orig_w = original.copy()
    smart_wrap(orig_w)
    
    pert_w = perturbed.copy()
    smart_wrap(pert_w)
    
    inter_w = None
    if intermediate:
        inter_w = intermediate.copy()
        smart_wrap(inter_w)
    
    projections = ['xy', 'xz', 'yz']
    
    # Helper to plot a single row
    def plot_row(ax_row, atoms, title_prefix, is_final=False):
        row_handles = []
        for col, proj in enumerate(projections):
            ax = ax_row[col]
            plot_projection(ax, atoms, projection=proj, depth_of_field=depth_of_field, dof_min_alpha=dof_min_alpha)
            h_mag = plot_mags(atoms, ax, projection=proj, depth_of_field=depth_of_field)
            h_rb = plot_rigid_bodies(atoms, ax, projection=proj, depth_of_field=depth_of_field)
            
            # Annotations only on intermediate (Defect) or Final
            h_ann = []
            h_iface = None
            if title_prefix != "Original":
                h_ann = plot_annotations(atoms, ax, projection=proj, depth_of_field=depth_of_field)
                h_iface = _highlight_interface_atoms(ax, atoms, projection=proj)
            
            if col == 0:
                row_handles.extend(get_atom_handles(atoms))
                if h_mag: row_handles.append(h_mag)
                if h_rb: row_handles.append(h_rb)
                row_handles.extend(h_ann)
                if h_iface: row_handles.append(h_iface)

            xl, yl = _get_axis_labels(proj)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(f"{title_prefix} ({proj.upper()})")
            
        # Z-Density
        if show_density:
            plot_z_density(ax_row[3], atoms)
            ax_row[3].set_title(f'Z-Density ({title_prefix})', fontsize=9)
            
        return row_handles

    # Plot Rows
    handles = []
    
    # Row 0: Original
    h0 = plot_row(axes[0], orig_w, "Original")
    handles.extend(h0)
    
    # Row 1: Intermediate (if exists) OR Final (if 2 rows)
    if intermediate:
        h1 = plot_row(axes[1], inter_w, "Defect Only")
        handles.extend(h1)
        h2 = plot_row(axes[2], pert_w, "Perturbed (Final)", is_final=True)
        handles.extend(h2)
    else:
        h1 = plot_row(axes[1], pert_w, "Perturbed")
        handles.extend(h1)
    
    # Add grain legend entries if density shown
    if show_density:
        handles.append(mpatches.Patch(facecolor=np.array([0.3, 0.5, 0.85]), label='Lower grain'))
        handles.append(mpatches.Patch(facecolor=np.array([0.85, 0.35, 0.25]), label='Upper grain'))

    # Legend (Unified) — filter duplicates
    by_label = {}
    for h in handles:
        l = h.get_label()
        if l and l not in by_label:
            by_label[l] = h
            
    fig.legend(handles=list(by_label.values()), loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
