from __future__ import annotations

from pathlib import Path

from baseline_suite.common import (
    load_cora_context,
    run_federated_baseline,
)
from baseline_suite.config import DEFAULT_CONFIG
from baseline_suite.pdfl import PDFLAggregator


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    config = DEFAULT_CONFIG
    context = load_cora_context(project_dir, config)

    detector_path = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "pdfl_calibration"
        / "pdfl_detector.joblib"
    )

    aggregator = PDFLAggregator(detector_path)

    run_federated_baseline(
        method_name="PD-FL (GCN adaptation)",
        output_slug="baseline_pdfl",
        config=config,
        client_graphs_cpu=context.attacked_client_graphs_cpu,
        context=context,
        aggregate_fn=aggregator,
        method_metadata={
            "paper_doi": "10.1016/j.cose.2023.103557",
            "adaptation": (
                "Persistent-homology signatures are extracted from "
                "GCN local model updates and filtered with a clean-calibrated "
                "Isolation Forest before weighted aggregation."
            ),
        },
    )


if __name__ == "__main__":
    main()
