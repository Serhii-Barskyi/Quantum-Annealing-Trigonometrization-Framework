from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import numpy as np

@dataclass(slots=True)
class Case300Dataset:
    case: str
    n: int
    M_complex: int
    rank: int
    N_train: int
    N_test: int
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    theta_train: np.ndarray
    theta_test: np.ndarray
    meta_train: list[str]
    meta_test: list[str]
    edges: list[tuple[int, int, float]]
    slack_buses: list[int]
    non_slack: list[int]
    params: dict[str, Any]

    @property
    def n_train(self) -> int:
        return self.N_train

    @property
    def n_test(self) -> int:
        return self.N_test


def load_dataset(path: str | Path) -> Case300Dataset:
    """Load the frozen case300 JSON into the historical notebook contract."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    n_train = int(raw["n_train"])
    n_test = int(raw["n_test"])
    n = int(raw["n"])
    re = np.asarray(raw["X_modeA_re"], dtype=np.float64)
    im = np.asarray(raw["X_modeA_im"], dtype=np.float64)
    X = re + 1j * im
    y = np.asarray(raw["y_lsf"], dtype=np.float64)
    theta = np.asarray(raw["theta_rad"], dtype=np.float64)
    meta = [str(x) for x in raw["meta"]]
    params = dict(raw.get("params", {}))
    slack = [int(x) for x in params.get("slack_buses", [0])]
    slack_set = set(slack)
    edges = [(int(i), int(j), float(b)) for i, j, b in raw["edges"]]
    if X.shape != (n_train + n_test, int(raw["M_complex"])):
        raise ValueError(f"Unexpected case300 feature shape: {X.shape}")
    if y.shape != (n_train + n_test, n):
        raise ValueError(f"Unexpected case300 target shape: {y.shape}")
    return Case300Dataset(
        case=str(raw.get("case", "case300")), n=n,
        M_complex=int(raw["M_complex"]), rank=int(raw.get("rank_modeA", np.linalg.matrix_rank(X[:n_train]))),
        N_train=n_train, N_test=n_test,
        X_train=X[:n_train], X_test=X[n_train:n_train+n_test],
        y_train=y[:n_train], y_test=y[n_train:n_train+n_test],
        theta_train=theta[:n_train], theta_test=theta[n_train:n_train+n_test],
        meta_train=meta[:n_train], meta_test=meta[n_train:n_train+n_test],
        edges=edges, slack_buses=slack,
        non_slack=[i for i in range(n) if i not in slack_set], params=params,
    )
