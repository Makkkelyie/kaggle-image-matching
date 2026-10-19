"""Small, dependency-free (NumPy only) camera-geometry helpers.

Convention (same as COLMAP and the competition CSV): a pose (R, t) maps a
world point X into camera coordinates, x_cam = R @ X + t ("cam_from_world").
The camera centre in world coordinates is therefore C = -R^T t.
"""
from __future__ import annotations

import numpy as np


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World-space position of the camera for a cam_from_world pose."""
    return -np.asarray(R).T @ np.asarray(t)


def invert_pose(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) = A_from_B  ->  B_from_A."""
    R = np.asarray(R)
    return R.T, -R.T @ np.asarray(t)


def compose_poses(R_ab, t_ab, R_bc, t_bc) -> tuple[np.ndarray, np.ndarray]:
    """A_from_B o B_from_C = A_from_C, i.e. first apply (R_bc, t_bc), then (R_ab, t_ab)."""
    R_ab, R_bc = np.asarray(R_ab), np.asarray(R_bc)
    return R_ab @ R_bc, R_ab @ np.asarray(t_bc) + np.asarray(t_ab)


def relative_pose(R_i, t_i, R_j, t_j) -> tuple[np.ndarray, np.ndarray]:
    """Pose of camera j relative to camera i: cam_j_from_cam_i."""
    R_wi, t_wi = invert_pose(R_i, t_i)  # world_from_cam_i
    return compose_poses(R_j, t_j, R_wi, t_wi)


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Geodesic angle between two rotations, in degrees."""
    cos = (np.trace(np.asarray(R_est).T @ np.asarray(R_gt)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def translation_angle_error_deg(t_est: np.ndarray, t_gt: np.ndarray, eps: float = 1e-12) -> float:
    """Angle between two translation directions, in degrees (scale-invariant)."""
    a, b = np.asarray(t_est, float), np.asarray(t_gt, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0 if (na < eps and nb < eps) else 180.0
    cos = np.dot(a, b) / (na * nb)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def rotation_from_axis_angle(axis, angle_rad: float) -> np.ndarray:
    """Rodrigues' formula."""
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * K @ K


def is_rotation(R: np.ndarray, atol: float = 1e-6) -> bool:
    R = np.asarray(R)
    return R.shape == (3, 3) and np.allclose(R.T @ R, np.eye(3), atol=atol) and np.isclose(np.linalg.det(R), 1.0, atol=atol)


def umeyama_similarity(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity transform dst ~ s * R @ src + t (Umeyama 1991).

    src, dst: (N, 3) arrays with N >= 3 non-collinear points.
    Returns (s, R, t).
    """
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s) if (with_scale and var_s > 0) else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply_similarity(s: float, R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return s * np.asarray(pts) @ np.asarray(R).T + np.asarray(t)


def parse_matrix(text: str, shape) -> np.ndarray:
    """Parse the competition's ';'-separated, row-major encoding."""
    return np.array([float(v) for v in str(text).split(";")], dtype=float).reshape(shape)


def format_matrix(arr) -> str:
    return ";".join(f"{v:.9g}" for v in np.asarray(arr, float).ravel())
