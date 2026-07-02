"""
Generalized Grain Boundary Generator
支持所有晶体结构 (7大晶系): Triclinic, Monoclinic, Orthorhombic, Tetragonal, Trigonal, Hexagonal, Cubic
"""
import numpy as np
from ase.build import bulk, make_supercell
from ase import Atoms
from typing import Optional, Tuple, Dict, List
from scipy.spatial.transform import Rotation as R


# ============================================
# 晶体系统检测与对称性处理
# ============================================

def detect_crystal_system(cell: np.ndarray, tol: float = 1e-4) -> str:
    """
    根据晶胞参数检测晶体系统
    
    Args:
        cell: 3x3 晶胞矩阵 (列向量为基矢)
        tol: 公差
        
    Returns:
        晶体系统名称: 'triclinic', 'monoclinic', 'orthorhombic', 
                     'tetragonal', 'trigonal', 'hexagonal', 'cubic'
    """
    # 获取晶胞参数
    lengths = np.array([np.linalg.norm(cell[:, i]) for i in range(3)])
    angles = np.array([
        np.degrees(np.arccos(np.dot(cell[:, 1], cell[:, 2]) / (lengths[1] * lengths[2]))),
        np.degrees(np.arccos(np.dot(cell[:, 0], cell[:, 2]) / (lengths[0] * lengths[2]))),
        np.degrees(np.arccos(np.dot(cell[:, 0], cell[:, 1]) / (lengths[0] * lengths[1])))
    ])
    
    a, b, c = lengths
    alpha, beta, gamma = angles
    
    # 检查立方晶系 (a=b=c, α=β=γ=90°)
    if (abs(a - b) < tol * a and abs(b - c) < tol * b and 
        abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 90) < tol):
        return 'cubic'
    
    # 检查六方/三方晶系 (a=b≠c, α=β=90°, γ=120°)
    if (abs(a - b) < tol * a and abs(c - a) > tol * a and
        abs(alpha - 90) < tol and abs(beta - 90) < tol and 
        abs(gamma - 120) < tol):
        return 'hexagonal'
    
    # 三方晶系 (菱方坐标系) - 特殊处理
    # 当使用菱方坐标系时: a=b=c, α=β=γ≠90°
    if (abs(a - b) < tol * a and abs(b - c) < tol * b and
        abs(alpha - beta) < tol and abs(beta - gamma) < tol and
        abs(alpha - 90) > tol):
        return 'trigonal'
    
    # 检查四方晶系 (a=b≠c, α=β=γ=90°)
    if (abs(a - b) < tol * a and abs(c - a) > tol * a and
        abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 90) < tol):
        return 'tetragonal'
    
    # 检查正交晶系 (a≠b≠c, α=β=γ=90°)
    if (abs(a - b) > tol * a and abs(b - c) > tol * b and
        abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 90) < tol):
        return 'orthorhombic'
    
    # 检查单斜晶系 (a≠b≠c, α=γ=90°, β≠90°)
    if (abs(alpha - 90) < tol and abs(gamma - 90) < tol and 
        abs(beta - 90) > tol):
        return 'monoclinic'
    
    # 三斜晶系 (a≠b≠c, α≠β≠γ≠90°)
    return 'triclinic'


def miller_to_cartesian(miller: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """
    Miller指数转换为笛卡尔坐标 (针对一般晶系)
    
    Args:
        miller: Miller指数 [h, k, l]
        cell: 3x3 晶胞矩阵
        
    Returns:
        笛卡尔坐标向量
    """
    # Miller指数对应的是倒空间中的点
    # 转换为笛卡尔坐标需要乘以倒晶胞矩阵
    recips = np.linalg.inv(cell).T  # 倒晶胞矩阵 (列向量为倒晶格基矢)
    return recips @ miller


def cartesian_to_miller(cart: np.ndarray, cell: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    """
    笛卡尔坐标转换为Miller指数
    
    Args:
        cart: 笛卡尔坐标向量
        cell: 3x3 晶胞矩阵
        tol: 公差
        
    Returns:
        Miller指数 [h, k, l] (四舍五入到最近整数)
    """
    recips = np.linalg.inv(cell).T
    miller = recips.T @ cart  # 转换到Miller指数空间
    return np.round(miller).astype(int)


# ============================================
# 旋转矩阵生成 (支持任意晶系)
# ============================================

def rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    生成旋转矩阵 (Rodrigues公式)
    
    Args:
        axis: 旋转轴 (单位向量或Miller指数)
        angle_deg: 旋转角度 (度)
        
    Returns:
        3x3 旋转矩阵
    """
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    
    theta = np.radians(angle_deg)
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


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    从轴和角度生成旋转矩阵 (更高效的实现)
    
    Args:
        axis: 旋转轴 (3D向量)
        angle_deg: 旋转角度 (度)
        
    Returns:
        3x3 旋转矩阵
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    
    theta = np.radians(angle_deg)
    
    # 使用scipy的旋转
    rot = R.from_rotvec(axis * theta)
    return rot.as_matrix()


# ============================================
# 通用晶界生成核心算法
# ============================================

def create_grain_boundary(
    atoms: Atoms,
    rotation_matrix: np.ndarray,
    boundary_plane: np.ndarray,
    size: Tuple[int, int, int],
    shift: float = 0.0
) -> Atoms:
    """
    创建晶界结构 (通用算法，适用于所有晶系)
    
    算法：
    1. 创建超胞
    2. 复制原子
    3. 对一半原子应用旋转
    4. 拼接两个晶粒
    5. 删除重叠原子
    6. 弛豫晶界
    
    Args:
        atoms: 原始原子结构
        rotation_matrix: 旋转矩阵 (用于 Grain B)
        boundary_plane: 晶界面法向 (Miller指数)
        size: 超胞大小 (nx, ny, nz)
        shift: 沿法向的平移分数坐标
        
    Returns:
        晶界结构 (Atoms对象)
    """
    # 1. 创建超胞
    prim = atoms.copy()
    sc_matrix = np.diag(size)
    supercell = make_supercell(prim, sc_matrix)
    
    # 2. 确定晶界面法向的笛卡尔坐标
    boundary_normal = miller_to_cartesian(boundary_plane, supercell.cell)
    boundary_normal /= np.linalg.norm(boundary_normal)
    
    # 3. 将晶界平移到超胞中心
    center = np.mean(supercell.positions, axis=0)
    supercell.translate(-center + boundary_normal * (size[2] * np.linalg.norm(supercell.cell[2]) / 2))
    
    # 4. 分割晶粒
    # 沿晶界面法向分割
    positions = supercell.positions.copy()
    projections = positions @ boundary_normal
    
    # 找到中点
    mid_projection = np.mean(projections)
    
    # Grain A: 前半部分 (保留原位)
    indices_a = np.where(projections < mid_projection + shift)[0]
    
    # Grain B: 后半部分 (旋转)
    indices_b = np.where(projections >= mid_projection + shift)[0]
    
    # 5. 创建两个晶粒
    grain_a = supercell.copy()
    grain_b = supercell.copy()
    
    # 移除不需要的原子
    delete_a = [i for i in range(len(grain_a)) if i not in indices_a]
    delete_b = [i for i in range(len(grain_b)) if i not in indices_b]
    
    if delete_a:
        del grain_a[delete_a]
    if delete_b:
        del grain_b[delete_b]
    
    # 6. 对 Grain B 应用旋转
    # 绕晶界面法向旋转
    center_b = np.mean(grain_b.positions, axis=0)
    grain_b.positions -= center_b
    grain_b.positions = grain_b.positions @ rotation_matrix.T
    grain_b.positions += center_b
    
    # 7. 拼接晶粒
    # 计算平移量 (使晶界对齐)
    c_vector = boundary_normal * np.linalg.norm(supercell.cell[2]) * size[2] / 2
    grain_b.translate(c_vector)
    
    # 合并
    bicrystal = grain_a + grain_b
    
    # 8. 更新晶胞
    final_cell = supercell.cell.copy()
    final_cell[2] *= 2  # 叠加两个晶粒
    bicrystal.set_cell(final_cell)
    bicrystal.pbc = [True, True, True]
    
    # 9. 包装原子
    bicrystal.wrap()
    
    return bicrystal


# ============================================
# 基于旋转的晶界生成器 (支持所有晶系)
# ============================================

class GrainBoundaryGenerator:
    """
    通用晶界生成器，支持所有晶体结构
    
    晶界类型:
    - Tilt boundary: 倾转晶界 (旋转轴平行于晶界面)
    - Twist boundary: 扭转晶界 (旋转轴垂直于晶界面)
    - Mixed boundary: 混合晶界 (旋转轴与晶界面成一定角度)
    """
    
    def __init__(self, atoms: Atoms, crystal_system: str = None):
        """
        初始化晶界生成器
        
        Args:
            atoms: ASE Atoms对象
            crystal_system: 晶体系统名称 (自动检测如果未指定)
        """
        self.atoms = atoms.copy()
        self.cell = atoms.cell.copy()
        
        # 检测晶体系统
        self.crystal_system = crystal_system or detect_crystal_system(self.cell)
        
        # 存储晶面信息
        self.planes = {}
        self.vectors = {}
        
    def _detect_crystal_system(self) -> str:
        """自动检测晶体系统"""
        return detect_crystal_system(self.cell)
    
    def get_crystal_system_info(self) -> Dict:
        """
        获取晶体系统的对称性信息
        
        Returns:
            包含晶系参数的字典
        """
        lengths = np.array([np.linalg.norm(self.cell[:, i]) for i in range(3)])
        angles = np.array([
            np.degrees(np.arccos(np.dot(self.cell[:, 1], self.cell[:, 2]) / 
                                  (lengths[1] * lengths[2]))),
            np.degrees(np.arccos(np.dot(self.cell[:, 0], self.cell[:, 2]) / 
                                  (lengths[0] * lengths[2]))),
            np.degrees(np.arccos(np.dot(self.cell[:, 0], self.cell[:, 1]) / 
                                  (lengths[0] * lengths[1])))
        ])
        
        return {
            'crystal_system': self.crystal_system,
            'lattice_parameters': {
                'a': lengths[0], 'b': lengths[1], 'c': lengths[2],
                'alpha': angles[0], 'beta': angles[1], 'gamma': angles[2]
            },
            'symmetry_features': self._get_symmetry_features()
        }
    
    def _get_symmetry_features(self) -> Dict:
        """获取晶体的对称性特征"""
        features = {}
        
        if self.crystal_system == 'cubic':
            features['rotational_symmetry'] = '4-fold about <100>, 3-fold about <111>'
            features['mirror_planes'] = '(100), (110), (111)'
        elif self.crystal_system == 'hexagonal':
            features['rotational_symmetry'] = '6-fold about [0001]'
            features['mirror_planes'] = '(0001), {10-10}'
        elif self.crystal_system == 'tetragonal':
            features['rotational_symmetry'] = '4-fold about [001]'
            features['mirror_planes'] = '(001), (100), (010)'
        elif self.crystal_system == 'orthorhombic':
            features['rotational_symmetry'] = '2-fold about [100], [010], [001]'
            features['mirror_planes'] = '(001), (100), (010)'
        elif self.crystal_system == 'monoclinic':
            features['rotational_symmetry'] = '2-fold about one axis'
            features['mirror_planes'] = 'one mirror plane'
        elif self.crystal_system == 'trigonal':
            features['rotational_symmetry'] = '3-fold about [111]'
            features['mirror_planes'] = '(0001), {10-10}'
        else:  # triclinic
            features['rotational_symmetry'] = 'None'
            features['mirror_planes'] = 'None'
            
        return features
    
    def generate_tilt_boundary(
        self,
        rotation_axis: np.ndarray,
        tilt_angle: float,
        boundary_plane: np.ndarray,
        size: Tuple[int, int, int] = (1, 1, 1),
        delete_overlap: bool = True,
        tol: float = 1.5
    ) -> Atoms:
        """
        生成倾转晶界 (Tilt Boundary)
        
        在倾转晶界中，两个晶粒围绕平行于晶界面的轴旋转。
        旋转轴 ∈ 晶界面, boundary_normal ⊥ rotation_axis
        
        Args:
            rotation_axis: 旋转轴 (Miller指数或笛卡尔坐标)
            tilt_angle: 倾转角度 (度)
            boundary_plane: 晶界面法向 (Miller指数)
            size: 超胞大小 (nx, ny, nz)
            delete_overlap: 是否删除重叠原子
            tol: 重叠删除公差 (Å)
            
        Returns:
            倾转晶界结构 (Atoms对象)
        """
        # 转换为笛卡尔坐标
        rotation_axis_cart = self._to_cartesian(rotation_axis)
        boundary_normal = self._to_cartesian(boundary_plane)
        
        # 归一化
        rotation_axis_unit = rotation_axis_cart / np.linalg.norm(rotation_axis_cart)
        boundary_normal_unit = boundary_normal / np.linalg.norm(boundary_normal)
        
        # 验证: 旋转轴应在晶界面内
        if abs(np.dot(rotation_axis_unit, boundary_normal_unit)) > 0.1:
            raise ValueError(
                f"Rotation axis must be parallel to boundary plane. "
                f"Dot product: {np.dot(rotation_axis_unit, boundary_normal_unit)}"
            )
        
        # 生成旋转矩阵 (绕旋转轴)
        R = rotation_matrix(rotation_axis_unit, tilt_angle)
        
        # 创建晶界
        bicrystal = create_grain_boundary(
            self.atoms, R, boundary_plane, size, shift=0.0
        )
        
        # 删除重叠原子
        if delete_overlap:
            bicrystal = self._delete_overlapping_atoms(bicrystal, tol)
            
        return bicrystal
    
    def generate_twist_boundary(
        self,
        rotation_axis: np.ndarray,
        twist_angle: float,
        boundary_plane: Optional[np.ndarray] = None,
        size: Tuple[int, int, int] = (1, 1, 1),
        delete_overlap: bool = True,
        tol: float = 1.5
    ) -> Atoms:
        """
        生成扭转晶界 (Twist Boundary)
        
        在扭转晶界中，两个晶粒围绕垂直于晶界面的轴旋转。
        旋转轴 ∥ boundary_normal
        
        Args:
            rotation_axis: 旋转轴 (垂直于晶界面, Miller指数或笛卡尔坐标)
            twist_angle: 扭转角度 (度)
            boundary_plane: 晶界面法向 (Miller指数，默认[0,0,1])
            size: 超胞大小 (nx, ny, nz)
            delete_overlap: 是否删除重叠原子
            tol: 重叠删除公差 (Å)
            
        Returns:
            扭转晶界结构 (Atoms对象)
        """
        # 默认晶界面为(001)
        boundary_plane = boundary_plane or [0, 0, 1]
        
        # 转换为笛卡尔坐标
        rotation_axis_cart = self._to_cartesian(rotation_axis)
        boundary_normal = self._to_cartesian(boundary_plane)
        
        # 归一化
        rotation_axis_unit = rotation_axis_cart / np.linalg.norm(rotation_axis_cart)
        boundary_normal_unit = boundary_normal / np.linalg.norm(boundary_normal)
        
        # 验证: 旋转轴应接近垂直于晶界面 (允许一定的公差)
        dot_product = abs(np.dot(rotation_axis_unit, boundary_normal_unit))
        if dot_product < 0.8:  # 允许25度的偏差
            raise ValueError(
                f"Rotation axis must be nearly perpendicular to boundary plane. "
                f"Dot product: {dot_product:.3f} (require >= 0.8)"
            )
        
        # 生成旋转矩阵 (绕法向轴)
        R = rotation_matrix(boundary_normal_unit, twist_angle)
        
        # 创建晶界
        bicrystal = create_grain_boundary(
            self.atoms, R, boundary_plane, size, shift=0.0
        )
        
        # 删除重叠原子
        if delete_overlap:
            bicrystal = self._delete_overlapping_atoms(bicrystal, tol)
            
        return bicrystal
    
    def generate_mixed_boundary(
        self,
        rotation_axis: np.ndarray,
        rotation_angle: float,
        boundary_plane: np.ndarray,
        tilt_fraction: float = 0.5,
        size: Tuple[int, int, int] = (1, 1, 1),
        delete_overlap: bool = True,
        tol: float = 1.5
    ) -> Atoms:
        """
        生成混合晶界 (Mixed Boundary) - 倾转 + 扭转
        
        通过组合倾转和扭转分量生成混合晶界。
        总旋转 = Tilt_rotation @ Twist_rotation
        
        Args:
            rotation_axis: 旋转轴 (Miller指数或笛卡尔坐标)
            rotation_angle: 旋转角度 (度)
            boundary_plane: 晶界面法向 (Miller指数)
            tilt_fraction: 倾转分量比例 (0-1, 0=纯扭转, 1=纯倾转)
            size: 超胞大小 (nx, ny, nz)
            delete_overlap: 是否删除重叠原子
            tol: 重叠删除公差 (Å)
            
        Returns:
            混合晶界结构 (Atoms对象)
        """
        # 转换为笛卡尔坐标
        rotation_axis_cart = self._to_cartesian(rotation_axis)
        boundary_normal = self._to_cartesian(boundary_plane)
        
        # 归一化
        rotation_axis_unit = rotation_axis_cart / np.linalg.norm(rotation_axis_cart)
        boundary_normal_unit = boundary_normal / np.linalg.norm(boundary_normal)
        
        # 分解旋转轴为平行和垂直于界面的分量
        parallel_component = np.dot(rotation_axis_unit, boundary_normal_unit) * boundary_normal_unit
        perpendicular_component = rotation_axis_unit - parallel_component
        
        # 归一化分量
        if np.linalg.norm(perpendicular_component) > 0.1:
            tilt_axis = perpendicular_component / np.linalg.norm(perpendicular_component)
        else:
            tilt_axis = np.array([1, 0, 0])  # 默认分量
            
        if np.linalg.norm(parallel_component) > 0.1:
            twist_axis = parallel_component / np.linalg.norm(parallel_component)
        else:
            twist_axis = np.cross(boundary_normal_unit, tilt_axis)
            if np.linalg.norm(twist_axis) < 0.1:
                twist_axis = np.array([0, 1, 0])
        
        # 计算倾转和扭转角度
        tilt_angle = rotation_angle * tilt_fraction
        twist_angle = rotation_angle * (1 - tilt_fraction)
        
        # 生成旋转矩阵
        R_tilt = rotation_matrix(tilt_axis, tilt_angle)
        R_twist = rotation_matrix(boundary_normal_unit, twist_angle)
        
        # 总旋转: 先扭转后倾转
        R_total = R_tilt @ R_twist
        
        # 创建晶界
        bicrystal = create_grain_boundary(
            self.atoms, R_total, boundary_plane, size, shift=0.0
        )
        
        # 删除重叠原子
        if delete_overlap:
            bicrystal = self._delete_overlapping_atoms(bicrystal, tol)
            
        return bicrystal
    
    def generate_grain_boundary_universal(
        self,
        boundary_type: str,
        rotation_axis: np.ndarray,
        rotation_angle: float,
        boundary_plane: Optional[np.ndarray] = None,
        size: Tuple[int, int, int] = (1, 1, 1),
        delete_overlap: bool = True,
        tol: float = 1.5,
        **kwargs
    ) -> Atoms:
        """
        通用晶界生成函数
        
        Args:
            boundary_type: 'tilt', 'twist', 'mixed'
            rotation_axis: 旋转轴 (Miller指数或笛卡尔坐标)
            rotation_angle: 旋转角度 (度)
            boundary_plane: 晶界面法向 (对于tilt和mixed必需)
            size: 超胞大小 (nx, ny, nz)
            delete_overlap: 是否删除重叠原子
            tol: 重叠删除公差 (Å)
            **kwargs: 其他参数
            
        Returns:
            晶界结构 (Atoms对象)
        """
        boundary_type = boundary_type.lower()
        
        boundary_params = {
            'size': size,
            'delete_overlap': delete_overlap,
            'tol': tol
        }
        boundary_params.update(kwargs)
        
        if boundary_type == 'tilt':
            if boundary_plane is None:
                raise ValueError("boundary_plane is required for tilt boundary")
            return self.generate_tilt_boundary(
                rotation_axis, rotation_angle, boundary_plane, **boundary_params
            )
        elif boundary_type == 'twist':
            return self.generate_twist_boundary(
                rotation_axis, rotation_angle, **boundary_params
            )
        elif boundary_type == 'mixed':
            if boundary_plane is None:
                raise ValueError("boundary_plane is required for mixed boundary")
            return self.generate_mixed_boundary(
                rotation_axis, rotation_angle, boundary_plane, **kwargs
            )
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}. Use 'tilt', 'twist', or 'mixed'")
    
    def _to_cartesian(self, vector: np.ndarray) -> np.ndarray:
        """
        将 Miller 指数或笛卡尔坐标转换为笛卡尔坐标
        
        Args:
            vector: Miller指数 [h,k,l] 或笛卡尔坐标 [x,y,z]
            
        Returns:
            笛卡尔坐标
        """
        vector = np.asarray(vector, dtype=float)
        
        if vector.shape != (3,):
            raise ValueError("Vector must be 3D")
            
        # 如果向量接近整数，认为是Miller指数
        if np.allclose(vector, np.round(vector), atol=0.01):
            return miller_to_cartesian(vector.astype(int), self.cell)
        else:
            return vector
    
    def _delete_overlapping_atoms(self, atoms: Atoms, tol: float = 1.5) -> Atoms:
        """
        删除距离过近的原子
        
        Args:
            atoms: 原子结构
            tol: 最小距离公差 (Å)
            
        Returns:
            删除重叠后的原子结构
        """
        from ase.neighborlist import NeighborList
        
        nl = NeighborList([tol/2]*len(atoms), skin=0.0, 
                         self_interaction=False, bothways=True)
        nl.update(atoms)
        
        to_delete = set()
        for i in range(len(atoms)):
            if i in to_delete:
                continue
            indices, offsets = nl.get_neighbors(i)
            for j in indices:
                if j > i and j not in to_delete:
                    to_delete.add(j)
        
        if to_delete:
            print(f"Deleting {len(to_delete)} overlapping atoms with tol={tol} Å")
            del atoms[list(to_delete)]
            
        return atoms
    
    def get_boundary_plane_vectors(self, plane: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取晶界平面内的两个正交向量
        
        Args:
            plane: 晶界面法向 (Miller指数)
            
        Returns:
            (v1, v2): 平面内的两个正交向量 (笛卡尔坐标)
        """
        normal = self._to_cartesian(plane)
        normal /= np.linalg.norm(normal)
        
        # 找到平面内的向量
        # 找到与normal不平行的参考向量
        if abs(normal[0]) < 0.9:
            ref = np.array([1, 0, 0])
        else:
            ref = np.array([0, 1, 0])
            
        # v1 = ref - (ref·normal) * normal
        v1 = ref - np.dot(ref, normal) * normal
        v1 /= np.linalg.norm(v1)
        
        # v2 = normal × v1
        v2 = np.cross(normal, v1)
        v2 /= np.linalg.norm(v2)
        
        return v1, v2


# ============================================
# 辅助函数
# ============================================

def generate_grain_boundary_csl(element, sigma, axis, angle, plane=[0, 0, 1], size=(1, 1, 1), vacuum=0.0, delete_overlap=True, tol=1.5):
    """
    CSL晶界生成器 (向后兼容，支持立方晶系)
    
    Args:
        element: 元素符号 (如 'Al', 'Fe')
        sigma: Sigma值
        axis: 旋转轴 [u, v, w]
        angle: 旋转角度 (度)
        plane: 晶界面 (可选)
        size: 超胞大小
        vacuum: 真空层 (未实现)
        delete_overlap: 删除重叠原子
        tol: 重叠公差
        
    Returns:
        CSL晶界结构
    """
    # 导入CSL核心模块
    from NepTrain.core.perturb.csl_core import find_csl_basis, get_rotation_matrix
    
    # 获取CSL基矢
    M, M_prime = find_csl_basis(sigma, axis, angle)
    
    if M is None:
        raise ValueError(f"Could not find CSL vectors for Sigma {sigma}")
    
    # 创建primitive单元
    prim = bulk(element, cubic=True)
    
    # 创建 Grain A
    grain_a = make_supercell(prim, M.T)
    
    # 创建 Grain B (旋转)
    grain_b_unrotated = make_supercell(prim, M_prime.T)
    
    # 计算旋转矩阵
    try:
        M_inv = np.linalg.inv(M_prime)
        R_exact = M @ M_inv
        if np.allclose(R_exact @ R_exact.T, np.eye(3), atol=1e-3):
            R = R_exact
        else:
            R = get_rotation_matrix(axis, angle)
    except np.linalg.LinAlgError:
        R = get_rotation_matrix(axis, angle)
    
    # 应用旋转
    grain_b = grain_b_unrotated.copy()
    grain_b.set_positions(grain_b.positions @ R.T)
    grain_b.set_cell(grain_b.cell @ R.T)
    grain_b.set_cell(grain_a.cell)
    
    # 标准化方向
    cell = grain_a.get_cell()
    v1, v2, v3 = cell[0], cell[1], cell[2]
    
    x_new = v1 / np.linalg.norm(v1)
    normal = np.cross(v1, v2)
    z_new = normal / np.linalg.norm(normal)
    y_new = np.cross(z_new, x_new)
    
    Q = np.array([x_new, y_new, z_new])
    
    grain_a.set_positions(grain_a.positions @ Q.T)
    grain_a.set_cell(grain_a.cell @ Q.T)
    
    grain_b.set_positions(grain_b.positions @ Q.T)
    grain_b.set_cell(grain_b.cell @ Q.T)
    
    # 超胞
    nx, ny, nz = size
    grain_a = make_supercell(grain_a, np.diag([nx, ny, nz]))
    grain_b = make_supercell(grain_b, np.diag([nx, ny, nz]))
    
    # 叠加
    c_vec = grain_a.cell[2]
    grain_b.translate(c_vec)
    
    bicrystal = grain_a + grain_b
    
    final_cell = grain_a.cell.copy()
    final_cell[2] *= 2
    bicrystal.set_cell(final_cell)
    bicrystal.pbc = [True, True, True]
    bicrystal.wrap()
    
    if delete_overlap:
        from ase.neighborlist import NeighborList
        nl = NeighborList([tol/2]*len(bicrystal), skin=0.0, 
                         self_interaction=False, bothways=True)
        nl.update(bicrystal)
        
        to_delete = set()
        for i in range(len(bicrystal)):
            if i in to_delete:
                continue
            indices, offsets = nl.get_neighbors(i)
            for j in indices:
                if j > i and j not in to_delete:
                    to_delete.add(j)
        
        if to_delete:
            print(f"Deleting {len(to_delete)} overlapping atoms")
            del bicrystal[list(to_delete)]

    bicrystal.info['perturb_annotation'] = {
        'type': 'grain_boundary',
        'metadata': {'sigma': sigma, 'axis': list(axis), 'angle': angle}
    }

    return bicrystal


def generate_grain_boundary(atoms_or_element, sigma=None, axis=None, angle=None,
                            angle_deg=None, min_dist=None, vacuum=0.0,
                            delete_overlap=True, translation=None,
                            translation_frac=None, tol=1.5, **kwargs):
    """Backward-compatible grain boundary generator.

    Accepts both the old API (Atoms first arg + angle_deg/min_dist kwargs)
    and the new API (element string + sigma/axis/angle).

    When ``sigma`` is explicitly provided → use CSL generation.
    When ``sigma`` is None → use simple rotation (no CSL matching needed).

    Maps ``angle_deg`` → ``angle``, ``min_dist`` → ``tol``.
    """
    # Normalize angle
    if angle is None and angle_deg is not None:
        angle = angle_deg
    if axis is None:
        axis = [0, 0, 1]

    # Map min_dist → tol
    if min_dist is not None:
        tol = min_dist

    # ----------------------------------------------------------------
    # Dual-mode dispatch: CSL vs simple rotation
    # ----------------------------------------------------------------
    use_csl = sigma is not None

    if isinstance(atoms_or_element, Atoms):
        symbols = set(atoms_or_element.get_chemical_symbols())
        element = symbols.pop() if len(symbols) == 1 else atoms_or_element.get_chemical_symbols()[0]
        source_atoms = atoms_or_element
    else:
        element = atoms_or_element
        source_atoms = None

    if use_csl:
        # CSL mode — requires exact CSL matching
        result = generate_grain_boundary_csl(
            element, sigma=sigma, axis=axis, angle=angle or 36.87,
            vacuum=vacuum, delete_overlap=delete_overlap, tol=tol)
    else:
        # Simple rotation mode — no CSL, just rotate and stack
        result = _generate_simple_rotation_gb(
            element, axis=axis, angle_deg=angle or 30.0,
            vacuum=vacuum, delete_overlap=delete_overlap, tol=tol,
            source_atoms=source_atoms)

    # Apply translation if specified
    if translation_frac is not None and result is not None:
        tx, ty = translation_frac[0], translation_frac[1]
        pos = result.get_positions()
        cell = result.get_cell()
        mid_z = (pos[:, 2].max() + pos[:, 2].min()) / 2
        upper = pos[:, 2] > mid_z
        pos[upper, 0] += tx * cell[0, 0]
        pos[upper, 1] += ty * cell[1, 1]
        result.set_positions(pos)

    return result


def _generate_simple_rotation_gb(element, axis=[0,0,1], angle_deg=30.0,
                                  vacuum=0.0, delete_overlap=True, tol=1.5,
                                  source_atoms=None):
    """Generate a grain boundary by simple rotation without CSL matching.

    Algorithm:
    1. Build a conventional cubic cell
    2. Create two supercells stacked along z
    3. Rotate the upper grain about the given axis by angle_deg
    4. Optionally remove overlapping atoms
    """
    from ase.build import bulk, make_supercell as _make_sc

    if source_atoms is not None:
        prim = source_atoms.copy()
    else:
        prim = bulk(element, cubic=True)

    # Build supercell (2x2x2 for adequate grain size)
    sc = _make_sc(prim, np.diag([2, 2, 2]))

    # Stack two copies along z
    grain_a = sc.copy()
    grain_b = sc.copy()

    # Translate grain_b above grain_a
    c_vec = grain_a.cell[2]
    grain_b.translate(c_vec)
    if vacuum > 0:
        grain_b.translate(np.array([0, 0, vacuum]))

    bicrystal = grain_a + grain_b

    # Update cell
    final_cell = grain_a.cell.copy()
    final_cell[2] = final_cell[2] * 2
    if vacuum > 0:
        final_cell[2] = final_cell[2] + np.array([0, 0, vacuum])
    bicrystal.set_cell(final_cell)
    bicrystal.pbc = [True, True, True]

    # Rotate upper half of atoms
    R = get_rotation_matrix_local(axis, angle_deg)
    pos = bicrystal.get_positions()
    mid_z = np.mean(pos[:, 2])
    upper_mask = pos[:, 2] > mid_z

    # Rotate about the midplane
    pos[upper_mask] -= np.array([0, 0, mid_z])
    pos[upper_mask] = pos[upper_mask] @ R.T
    pos[upper_mask] += np.array([0, 0, mid_z])
    bicrystal.set_positions(pos)

    bicrystal.wrap()

    if delete_overlap:
        from ase.neighborlist import NeighborList
        nl = NeighborList([tol / 2] * len(bicrystal), skin=0.0,
                          self_interaction=False, bothways=True)
        nl.update(bicrystal)

        to_delete = set()
        for i in range(len(bicrystal)):
            if i in to_delete:
                continue
            indices, _ = nl.get_neighbors(i)
            for j in indices:
                if j > i and j not in to_delete:
                    to_delete.add(j)

        if to_delete:
            del bicrystal[list(to_delete)]

    # Attach annotation metadata
    bicrystal.info['perturb_annotation'] = {
        'type': 'grain_boundary',
        'metadata': {
            'axis': list(axis),
            'angle': angle_deg,
            'mode': 'simple_rotation'
        }
    }

    return bicrystal


def get_rotation_matrix_local(axis, theta_deg):
    """Local copy of rotation matrix to avoid circular imports."""
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)

    theta = np.radians(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1 - c

    x, y, z = axis

    return np.array([
        [t*x*x + c,   t*x*y - z*s, t*x*z + y*s],
        [t*x*y + z*s, t*y*y + c,   t*y*z - x*s],
        [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
    ])

