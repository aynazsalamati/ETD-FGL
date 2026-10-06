from __future__ import annotations

from pathlib import Path

from baseline_suite.common import (
    load_cora_context,
    run_federated_baseline,
)
from baseline_suite.config import DEFAULT_CONFIG
from baseline_suite.flpurifier import (
    FLPurifierAggregator,
    make_flpurifier_local_trainer,
)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    config = DEFAULT_CONFIG
    context = load_cora_context(project_dir, config)

    local_trainer = make_flpurifier_local_trainer(
        config,
        contrastive_epochs=2,
        classifier_epochs=3,
        edge_drop_rate=0.20,
        feature_drop_rate=0.20,
        temperature=0.5,
    )
    aggregator = FLPurifierAggregator(minimum_similarity=0.05)

    run_federated_baseline(
        method_name="FLPurifier-GNN (GCN adaptation)",
        output_slug="baseline_flpurifier",
        config=config,
        client_graphs_cpu=context.attacked_client_graphs_cpu,
        context=context,
        local_train_fn=local_trainer,
        aggregate_fn=aggregator,
        method_metadata={
            "paper_doi": "10.1109/TIFS.2024.3384846",
            "adaptation": (
                "The two-layer GCN is split into a conv1 feature extractor "
                "and conv2 classifier. The extractor uses decoupled graph "
                "contrastive training; the classifier uses adaptive "
                "agreement-weighted aggregation."
            ),
            "contrastive_epochs": 2,
            "classifier_epochs": 3,
        },
    )


if __name__ == "__main__":
    main()
