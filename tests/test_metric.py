import numpy as np
import pandas as pd
import pytest

from imc.geometry import apply_similarity, compose_poses, format_matrix, invert_pose, rotation_from_axis_angle
from imc.metric import count_accurate_cameras, score

NAN_R, NAN_T = ";".join(["nan"] * 9), ";".join(["nan"] * 3)


def make_scene(rng, dataset, scene, n, prefix):
    rows = []
    for i in range(n):
        R = rotation_from_axis_angle(rng.normal(size=3), rng.uniform(0, np.pi))
        C = rng.uniform(-5, 5, size=3)
        rows.append(dict(dataset=dataset, scene=scene, image=f"{prefix}{i}.png",
                         R=R, t=-R @ C))
    return rows


def to_df(rows):
    return pd.DataFrame([dict(dataset=r["dataset"], scene=r["scene"], image=r["image"],
                              rotation_matrix=format_matrix(r["R"]) if r["R"] is not None else NAN_R,
                              translation_vector=format_matrix(r["t"]) if r["t"] is not None else NAN_T)
                         for r in rows])


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    rows = make_scene(rng, "d1", "a", 12, "a") + make_scene(rng, "d1", "b", 10, "b")
    rows += [dict(dataset="d1", scene="outliers", image=f"o{i}.png", R=None, t=None) for i in range(3)]
    gt = to_df(rows)
    ths = pd.DataFrame([dict(dataset="d1", scene=s, thresholds="0.05;0.1;0.25;0.5;1.0") for s in "ab"])
    return rng, rows, gt, ths


def test_count_accurate_is_similarity_invariant():
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(15, 3))
    R = rotation_from_axis_angle([1, 2, 3], 0.7)
    pred = apply_similarity(0.3, R, rng.normal(size=3), gt)
    assert list(count_accurate_cameras(pred, gt, [0.01, 0.1])) == [15, 15]


def test_count_accurate_rejects_outlier_cameras():
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(15, 3))
    pred = gt.copy()
    pred[:4] += 10.0
    assert list(count_accurate_cameras(pred, gt, [0.1])) == [11]


def test_perfect_submission_scores_one(data):
    _, _, gt, ths = data
    final, table = score(gt, gt, ths)
    assert final == pytest.approx(1.0)
    assert table.loc[0, "clustering"] == pytest.approx(1.0)


def test_scores_are_invariant_to_world_frame(data):
    _, rows, gt, ths = data
    # rotate + translate the world of scene "a" only (each scene is reconstructed independently)
    G, g = rotation_from_axis_angle([0, 1, 0], 1.0), np.array([3.0, -1.0, 2.0])
    moved = [({**r, **dict(zip(("R", "t"), compose_poses(r["R"], r["t"], *invert_pose(G, g))))}
              if r["scene"] == "a" else r) for r in rows]
    final, _ = score(gt, to_df(moved), ths)
    assert final == pytest.approx(1.0)


def test_merging_scenes_hurts_clustering_not_maa(data):
    _, rows, gt, ths = data
    merged = [{**r, "scene": "cluster1"} if r["scene"] != "outliers" else r for r in rows]
    final, table = score(gt, to_df(merged), ths)
    # both scenes pick the merged cluster: clustering = (12+10)/(22+22)
    assert table.loc[0, "clustering"] == pytest.approx(22 / 44)
    # poses inside the cluster are still right for each scene, so mAA is untouched
    assert table.loc[0, "mAA"] == pytest.approx(1.0)
    assert final < 1.0


def test_all_outliers_scores_zero(data):
    _, rows, gt, ths = data
    pred = to_df([{**r, "scene": "outliers", "R": None, "t": None} for r in rows])
    final, _ = score(gt, pred, ths)
    assert final == 0.0


def test_unregistered_images_reduce_maa(data):
    _, rows, gt, ths = data
    pred_rows = [({**r, "R": None, "t": None} if r["image"] in {"a0.png", "a1.png", "a2.png"} else r) for r in rows]
    _, table = score(gt, to_df(pred_rows), ths)
    # scene a: 9 of 12 registered -> (9-3)/(12-3); scene b perfect -> 7/7
    assert table.loc[0, "mAA"] == pytest.approx((6 + 7) / (9 + 7))
    assert table.loc[0, "clustering"] == pytest.approx(1.0)
