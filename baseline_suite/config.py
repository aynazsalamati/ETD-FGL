from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    num_clients: int = 5
    max_rounds: int = 100
    local_epochs: int = 5
    patience: int = 20
    hidden_channels: int = 16
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    dropout: float = 0.5

    malicious_client_id: int = 0
    target_class: int = 0
    poison_rate: float = 0.20
    trigger_size: int = 3


DEFAULT_CONFIG = ExperimentConfig()
