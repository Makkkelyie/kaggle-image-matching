import numpy as np
import pytest

from imc.geometry import (
    apply_similarity, camera_center, compose_poses, format_matrix, invert_pose, is_rotation,
    parse_matrix, relative_pose, rotation_error_deg, rotation_from_axis_angle,
    translation_angle_error_deg, umeyama_similarity,
)


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_rotation_error_known_angle():
    R = rotation_from_axis_angle([0, 0, 1], np.radians(30))
    assert rotation_error_deg(R, np.eye(3)) == pytest.approx(30, abs=1e-6)
    assert rotation_error_deg(R, R) == pytest.approx(0, abs=1e-6)


def test_rotation_error_symmetric_and_clipped(rng):
    A, B = random_rotation(rng), random_rotation(rng)
    assert rotation_error_deg(A, B) == pytest.approx(rotation_error_deg(B, A))
    R180 = rotation_from_axis_angle([1, 0, 0], np.pi)
    assert rotation_error_deg(R180, np.eye(3)) == pytest.approx(180, abs=1e-4)


def test_translation_angle_error_scale_invariant():
    assert translation_angle_error_deg([1, 0, 0], [5, 0, 0]) == pytest.approx(0)
    assert translation_angle_error_deg([1, 0, 0], [0, 2, 0]) == pytest.approx(90)
    assert translation_angle_error_deg([1, 0, 0], [-1, 0, 0]) == pytest.approx(180)
    assert translation_angle_error_deg([0, 0, 0], [0, 0, 0]) == 0.0


def test_camera_center_projects_to_origin(rng):
    R, t = random_rotation(rng), rng.normal(size=3)
    C = camera_center(R, t)
    np.testing.assert_allclose(R @ C + t, 0, atol=1e-12)


def test_invert_and_compose_are_consistent(rng):
    R, t = random_rotation(rng), rng.normal(size=3)
    Ri, ti = invert_pose(R, t)
    Rc, tc = compose_poses(R, t, Ri, ti)
    np.testing.assert_allclose(Rc, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(tc, 0, atol=1e-12)


def test_compose_matches_sequential_application(rng):
    R1, t1, R2, t2 = random_rotation(rng), rng.normal(size=3), random_rotation(rng), rng.normal(size=3)
    X = rng.normal(size=3)
    R, t = compose_poses(R2, t2, R1, t1)  # apply 1 then 2
    np.testing.assert_allclose(R @ X + t, R2 @ (R1 @ X + t1) + t2, atol=1e-12)
    assert is_rotation(R)


def test_relative_pose_maps_cam_i_to_cam_j(rng):
    Ri, ti, Rj, tj = random_rotation(rng), rng.normal(size=3), random_rotation(rng), rng.normal(size=3)
    Rij, tij = relative_pose(Ri, ti, Rj, tj)
    X = rng.normal(size=3)
    np.testing.assert_allclose(Rij @ (Ri @ X + ti) + tij, Rj @ X + tj, atol=1e-12)


def test_relative_pose_invariant_to_world_similarity(rng):
    """Relative rotation does not depend on the world frame (gauge freedom of SfM)."""
    Ri, ti, Rj, tj = random_rotation(rng), rng.normal(size=3), random_rotation(rng), rng.normal(size=3)
    G, g = random_rotation(rng), rng.normal(size=3)  # new world frame: X = G X' + g
    Gi, gi = compose_poses(Ri, ti, G, g)
    Gj, gj = compose_poses(Rj, tj, G, g)
    assert rotation_error_deg(relative_pose(Ri, ti, Rj, tj)[0], relative_pose(Gi, gi, Gj, gj)[0]) < 1e-6


def test_umeyama_recovers_similarity(rng):
    src = rng.normal(size=(20, 3))
    s, R, t = 2.5, random_rotation(rng), rng.normal(size=3)
    dst = apply_similarity(s, R, t, src)
    s2, R2, t2 = umeyama_similarity(src, dst)
    assert s2 == pytest.approx(s)
    np.testing.assert_allclose(R2, R, atol=1e-9)
    np.testing.assert_allclose(t2, t, atol=1e-9)


def test_umeyama_never_returns_reflection(rng):
    src = rng.normal(size=(10, 3))
    dst = src * np.array([1, 1, -1])  # mirror image
    _, R, _ = umeyama_similarity(src, dst)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_matrix_roundtrip(rng):
    R = random_rotation(rng)
    np.testing.assert_allclose(parse_matrix(format_matrix(R), (3, 3)), R, atol=1e-8)
    assert np.isnan(parse_matrix("nan;nan;nan", (3,))).all()
