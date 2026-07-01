import numpy as np
from ase import Atoms

from NepTrain.core.select.select import (
    augment_descriptors_with_magnetism,
    farthest_point_sampling,
    magnetic_structure_features,
)


def _fe_with_spin(spin):
    atoms = Atoms("Fe2", positions=[[0, 0, 0], [1, 1, 1]], cell=[3, 3, 3], pbc=True)
    atoms.arrays["spin"] = np.asarray(spin, dtype=float)
    atoms.arrays["moment"] = np.asarray(spin, dtype=float)
    return atoms


def test_magnetic_features_distinguish_same_structure_different_spin_texture():
    fm = _fe_with_spin([[0, 0, 2], [0, 0, 2]])
    afm = _fe_with_spin([[0, 0, 2], [0, 0, -2]])

    feats = magnetic_structure_features([fm, afm])

    assert feats.shape == (2, 11)
    assert not np.allclose(feats[0], feats[1])
    # Pair alignment mean: FM=+1, AFM=-1 for two atoms.
    assert feats[0, 8] > 0.9
    assert feats[1, 8] < -0.9


def test_magnetic_descriptor_augmentation_makes_fps_select_spin_diversity():
    base = [_fe_with_spin([[0, 0, 2], [0, 0, 2]])]
    candidates = [
        _fe_with_spin([[0, 0, 2], [0, 0, 2]]),
        _fe_with_spin([[0, 0, 2], [0, 0, -2]]),
    ]
    # Structural descriptors are identical; only spin features should separate candidates.
    train_des = np.zeros((1, 2))
    new_des = np.zeros((2, 2))

    train_aug, new_aug = augment_descriptors_with_magnetism(train_des, new_des, base, candidates, weight=1.0)
    selected = farthest_point_sampling(new_aug, 1, min_dist=0.0, selected_data=train_aug)

    assert selected == [1]
