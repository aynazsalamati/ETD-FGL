from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    display_name: str
    num_clients: int = 5
    rounds: int = 50
    local_epochs: int = 3
    hidden_channels: int = 32
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    dropout: float = 0.5
    seed: int = 42
    malicious_client_id: int = 0
    target_class: int = 0
    poison_rate: float = 0.20
    trigger_size: int = 3
    reddit_sample_size: int = 8000
    reddit_sample_seed: int = 42
    explainer_epochs: int = 15
    explanation_probes: int = 2


DATASET_CONFIGS = {
    "cora": DatasetConfig(
        key="cora",
        display_name="Cora",
        rounds=50,
        local_epochs=3,
        hidden_channels=32,
    ),
    "pubmed": DatasetConfig(
        key="pubmed",
        display_name="PubMed",
        rounds=50,
        local_epochs=3,
        hidden_channels=32,
    ),
    "reddit": DatasetConfig(
        key="reddit",
        display_name="Reddit",
        rounds=50,
        local_epochs=3,
        hidden_channels=32,
        learning_rate=0.005,
        reddit_sample_size=8000,
        explainer_epochs=10,
        explanation_probes=2,
    ),
}


def get_config(dataset: str) -> DatasetConfig:
    key = dataset.strip().lower()
    if key not in DATASET_CONFIGS:
        raise ValueError(
            f"Unsupported dataset {dataset!r}. "
            f"Choose one of: {', '.join(DATASET_CONFIGS)}"
        )
    return DATASET_CONFIGS[key]
