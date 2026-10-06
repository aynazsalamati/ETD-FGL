from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    directory: str
    color: str
    linestyle: str
    marker: str


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        key="clean",
        label="Clean FedAvg",
        directory="fedavg_clean",
        color="#1f77b4",
        linestyle="-",
        marker="o",
    ),
    MethodSpec(
        key="attacked",
        label="Attacked FedAvg",
        directory="fedavg_structural_backdoor",
        color="#ff7f0e",
        linestyle="--",
        marker="^",
    ),
    MethodSpec(
        key="defended",
        label="ETD-FGL",
        directory="defended_fedavg_structural_backdoor",
        color="#2ca02c",
        linestyle="-.",
        marker="D",
    ),
)


OUTPUT_ROOT_NAME = "journal_figures"


def discover_dataset_roots(project_dir: Path) -> list[Path]:
    """Find outputs/federated_* folders containing at least one history."""

    outputs_dir = project_dir / "outputs"

    if not outputs_dir.exists():
        return []

    candidates: list[Path] = []

    for path in sorted(outputs_dir.glob("federated_*")):
        if not path.is_dir():
            continue

        if any(
            (path / method.directory / "training_history.csv").exists()
            for method in METHODS
        ):
            candidates.append(path)

    return candidates


def dataset_display_name(dataset_root: Path) -> str:
    raw = dataset_root.name.removeprefix("federated_")
    return raw.replace("_", " ").title()
