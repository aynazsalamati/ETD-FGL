from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import gudhi
import joblib
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from baseline_suite.common import StateDict


FEATURE_NAMES = [
    "update_l1",
    "update_l2",
    "update_linf",
    "update_mean",
    "update_std",
    "h0_count",
    "h0_total_persistence",
    "h0_max_lifetime",
    "h0_entropy",
    "h1_count",
    "h1_total_persistence",
    "h1_max_lifetime",
    "h1_entropy",
]


def flatten_update(
    local_state: StateDict,
    global_state: StateDict,
) -> np.ndarray:
    parts: list[np.ndarray] = []

    for key, local_value in local_state.items():
        if not torch.is_floating_point(local_value):
            continue
        delta = (
            local_value.detach().float().cpu()
            - global_state[key].detach().float().cpu()
        )
        parts.append(delta.reshape(-1).numpy())

    if not parts:
        raise ValueError("No floating-point parameters were found.")

    return np.concatenate(parts).astype(np.float64, copy=False)


def _resample_to_square(
    vector: np.ndarray,
    side: int = 24,
) -> np.ndarray:
    target_size = side * side

    if vector.size == 0:
        raise ValueError("Update vector is empty.")

    vector = vector.astype(np.float64, copy=False)
    median = float(np.median(vector))
    mad = float(np.median(np.abs(vector - median)))
    scale = max(1.4826 * mad, 1e-12)
    normalized = np.clip((vector - median) / scale, -8.0, 8.0)

    source_positions = np.linspace(0.0, 1.0, normalized.size)
    target_positions = np.linspace(0.0, 1.0, target_size)
    resampled = np.interp(
        target_positions,
        source_positions,
        normalized,
    )
    return resampled.reshape(side, side)


def _lifetime_statistics(
    persistence: list[tuple[int, tuple[float, float]]],
    dimension: int,
) -> tuple[float, float, float, float]:
    lifetimes: list[float] = []

    for dim, (birth, death) in persistence:
        if dim != dimension or not math.isfinite(death):
            continue
        lifetime = max(0.0, float(death - birth))
        if lifetime > 0.0:
            lifetimes.append(lifetime)

    if not lifetimes:
        return 0.0, 0.0, 0.0, 0.0

    array = np.asarray(lifetimes, dtype=np.float64)
    total = float(array.sum())
    probabilities = array / max(total, 1e-12)
    entropy = float(
        -(probabilities * np.log(probabilities + 1e-12)).sum()
    )

    return (
        float(array.size),
        total,
        float(array.max()),
        entropy,
    )


def persistence_features(
    local_state: StateDict,
    global_state: StateDict,
) -> np.ndarray:
    """Build a compact persistence-diagram signature of one model update."""

    vector = flatten_update(local_state, global_state)
    image = _resample_to_square(vector)

    complex_ = gudhi.CubicalComplex(
        top_dimensional_cells=image
    )
    persistence = complex_.persistence(
        homology_coeff_field=2,
        min_persistence=0.0,
    )

    h0 = _lifetime_statistics(persistence, 0)
    h1 = _lifetime_statistics(persistence, 1)

    features = np.asarray(
        [
            float(np.abs(vector).sum()),
            float(np.linalg.norm(vector)),
            float(np.abs(vector).max(initial=0.0)),
            float(vector.mean()),
            float(vector.std()),
            *h0,
            *h1,
        ],
        dtype=np.float64,
    )

    return features


def fit_detector(
    feature_matrix: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    if feature_matrix.ndim != 2:
        raise ValueError("feature_matrix must be two-dimensional.")
    if feature_matrix.shape[0] < 10:
        raise ValueError("At least 10 clean signatures are required.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    detector = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    detector.fit(scaled)

    clean_scores = detector.decision_function(scaled)
    score_median = float(np.median(clean_scores))
    score_mad = float(
        np.median(np.abs(clean_scores - score_median))
    )

    return {
        "feature_names": FEATURE_NAMES,
        "scaler": scaler,
        "detector": detector,
        "clean_score_median": score_median,
        "clean_score_mad": max(score_mad, 1e-6),
    }


def save_detector(package: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, path)


def load_detector(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            "PD-FL detector was not found. Run calibrate_pdfl_cora.py first:\n"
            f"{path}"
        )
    return joblib.load(path)


class PDFLAggregator:
    """Persistence-signature anomaly filtering adapted to GCN updates."""

    def __init__(
        self,
        detector_path: Path,
        malicious_floor: float = 0.03,
    ) -> None:
        package = load_detector(detector_path)
        self.scaler = package["scaler"]
        self.detector = package["detector"]
        self.clean_score_median = float(
            package["clean_score_median"]
        )
        self.clean_score_mad = float(package["clean_score_mad"])
        self.malicious_floor = malicious_floor

    def __call__(
        self,
        local_states: list[StateDict],
        client_counts: list[int],
        global_state: StateDict,
        round_number: int,
    ) -> tuple[StateDict, dict[str, float]]:
        del round_number

        features = np.stack(
            [
                persistence_features(state, global_state)
                for state in local_states
            ]
        )
        scaled = self.scaler.transform(features)
        decision_scores = self.detector.decision_function(scaled)

        robust_z = (
            decision_scores - self.clean_score_median
        ) / (1.4826 * self.clean_score_mad + 1e-12)

        # Positive z means clean-like. Negative z means anomalous.
        trust = 1.0 / (1.0 + np.exp(-robust_z))
        trust = np.clip(trust, self.malicious_floor, 1.0)

        raw_weights = trust * np.asarray(
            client_counts,
            dtype=np.float64,
        )
        normalized_weights = raw_weights / raw_weights.sum()

        averaged: StateDict = OrderedDict()
        for key in local_states[0].keys():
            reference = local_states[0][key]
            if not torch.is_floating_point(reference):
                averaged[key] = reference.clone()
                continue

            output = torch.zeros_like(reference)
            for state, weight in zip(
                local_states,
                normalized_weights,
                strict=True,
            ):
                output.add_(state[key], alpha=float(weight))
            averaged[key] = output

        diagnostics: dict[str, float] = {}
        for client_id, value in enumerate(decision_scores):
            diagnostics[f"pdfl_score_client_{client_id}"] = float(value)
            diagnostics[f"pdfl_trust_client_{client_id}"] = float(
                trust[client_id]
            )
            diagnostics[f"pdfl_weight_client_{client_id}"] = float(
                normalized_weights[client_id]
            )

        diagnostics["pdfl_min_trust"] = float(trust.min())
        diagnostics["pdfl_suspected_client"] = float(
            int(np.argmin(trust))
        )

        return averaged, diagnostics
