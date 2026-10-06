from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from baseline_suite.common import (
    clone_state,
    load_cora_context,
    sample_size_fedavg,
)
from baseline_suite.config import DEFAULT_CONFIG
from baseline_suite.pdfl import (
    fit_detector,
    persistence_features,
    save_detector,
)
from train_fedavg_cora import GCN
from train_fedavg_backdoor_cora import train_local_client


CALIBRATION_ROUNDS = 8


def main() -> None:
    config = DEFAULT_CONFIG
    project_dir = Path(__file__).resolve().parent
    context = load_cora_context(project_dir, config)

    clean_graphs = [
        data.to(context.device)
        for data in context.clean_client_graphs_cpu
    ]

    model = GCN(
        input_channels=context.dataset.num_features,
        hidden_channels=config.hidden_channels,
        output_channels=context.dataset.num_classes,
        dropout=config.dropout,
    ).to(context.device)

    collected: list[np.ndarray] = []

    print("=" * 88)
    print("Calibrating PD-FL persistence detector on clean client updates")
    print("=" * 88)

    for round_number in range(1, CALIBRATION_ROUNDS + 1):
        global_state = clone_state(model)
        local_states: list[OrderedDict[str, torch.Tensor]] = []
        counts: list[int] = []

        for graph in clean_graphs:
            local_state, _, train_count = train_local_client(
                global_state=global_state,
                client_data=graph,
                input_channels=context.dataset.num_features,
                output_channels=context.dataset.num_classes,
                device=context.device,
            )
            local_states.append(local_state)
            counts.append(train_count)
            collected.append(
                persistence_features(local_state, global_state)
            )

        averaged, _ = sample_size_fedavg(
            local_states,
            counts,
            global_state,
            round_number,
        )
        model.load_state_dict(averaged)
        print(
            f"Calibration round {round_number:02d} | "
            f"signatures collected: {len(collected)}"
        )

    feature_matrix = np.stack(collected)
    package = fit_detector(feature_matrix, seed=config.seed)

    output_path = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "pdfl_calibration"
        / "pdfl_detector.joblib"
    )
    save_detector(package, output_path)

    print("-" * 88)
    print(f"Detector saved to: {output_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
