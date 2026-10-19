import numpy as np
import pandas as pd
import pytest

from imc.data import NAN_R, clusters_to_rows, subsample_dataset, write_submission


def test_clusters_to_rows_assigns_outliers_and_dedups():
    I = (np.eye(3), np.zeros(3))
    rows = clusters_to_rows("d", ["a", "b", "c", "d"], [{"a": I, "b": I}, {"b": I, "c": I}])
    by_img = {r["image"]: r for r in rows}
    assert by_img["a"]["scene"] == by_img["b"]["scene"] == "cluster0"  # larger cluster wins b
    assert by_img["c"]["scene"] == "cluster1"
    assert by_img["d"]["scene"] == "outliers" and by_img["d"]["rotation_matrix"] == NAN_R
    assert len(rows) == 4


def test_write_submission_follows_sample_order(tmp_path):
    ss = pd.DataFrame(dict(image_id=["d_b", "d_a"], dataset="d", scene="x", image=["b", "a"],
                           rotation_matrix="", translation_vector=""))
    ss.to_csv(tmp_path / "ss.csv", index=False)
    rows = clusters_to_rows("d", ["a"], [{"a": (np.eye(3), np.zeros(3))}])  # "b" missing entirely
    out = write_submission(rows, tmp_path / "sub.csv", tmp_path / "ss.csv")
    assert list(out["image"]) == ["b", "a"]
    assert list(out["scene"]) == ["outliers", "cluster0"]


def test_subsample_keeps_every_scene():
    labels = pd.DataFrame(dict(dataset="d", scene=["s1"] * 10 + ["s2"] * 3 + ["outliers"] * 5,
                               image=[f"{i}.png" for i in range(18)]))
    sub = subsample_dataset(labels, "d", max_per_scene=4)
    assert sub["scene"].value_counts().to_dict() == {"s1": 4, "s2": 3, "outliers": 4}


def test_shortlist_pairs_topk_and_exhaustive():
    torch = pytest.importorskip("torch")
    from imc.features import FeatureConfig, shortlist_pairs

    small = torch.nn.functional.normalize(torch.randn(5, 8), dim=1)
    assert len(shortlist_pairs(small, FeatureConfig(exhaustive_below=10))) == 10
    # two well-separated groups: top-1 neighbours never cross groups
    a = torch.tensor([1.0, 0, 0]) + 0.01 * torch.randn(6, 3)
    b = torch.tensor([0, 1.0, 0]) + 0.01 * torch.randn(6, 3)
    desc = torch.nn.functional.normalize(torch.cat([a, b]), dim=1)
    pairs = shortlist_pairs(desc, FeatureConfig(exhaustive_below=2, topk=1, sim_threshold=0.99))
    assert pairs and all((i < 6) == (j < 6) for i, j in pairs)
