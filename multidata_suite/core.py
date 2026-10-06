from __future__ import annotations

import csv
import json
import time
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from multidata_suite.attack import (
    attach_evaluation_triggers,
    inject_clique_trigger,
    make_sparse_trigger_prototype,
)
from multidata_suite.config import DatasetConfig
from multidata_suite.data import build_client_graphs, load_dataset, load_or_create_partition
from multidata_suite.model import GCN, StateDict, clone_state, set_seed

LocalTrainFn = Callable[[StateDict, Data, int, int, torch.device, int, DatasetConfig], tuple[StateDict, float, int, dict[str, float]]]
AggregateFn = Callable[[list[StateDict], list[int], StateDict, int], tuple[StateDict, dict[str, float]]]


class ExperimentContext:
    def __init__(
        self,
        project_dir: Path,
        config: DatasetConfig,
        data_cpu: Data,
        num_features: int,
        num_classes: int,
        metadata: dict[str, Any],
        clean_graphs_cpu: list[Data],
        attacked_graphs_cpu: list[Data],
        attack_metadata: dict[str, Any],
        device: torch.device,
        triggered_data: Data,
        victim_nodes: Tensor,
        base_output_override: Path | None = None,
        partition_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.project_dir = project_dir
        self.config = config
        self.data_cpu = data_cpu
        self.num_features = num_features
        self.num_classes = num_classes
        self.metadata = metadata
        self.clean_graphs_cpu = clean_graphs_cpu
        self.attacked_graphs_cpu = attacked_graphs_cpu
        self.attack_metadata = attack_metadata
        self.device = device
        self.clean_data = data_cpu.to(device)
        self.triggered_data = triggered_data.to(device)
        self.victim_nodes = victim_nodes.to(device)
        self._base_output_override = base_output_override
        self.partition_metadata = partition_metadata or {}

    @property
    def base_output(self) -> Path:
        if self._base_output_override is not None:
            return self._base_output_override
        return (
            self.project_dir
            / "outputs"
            / "final_multidataset"
            / f"federated_{self.config.key}"
        )


def build_context(
    project_dir: Path,
    config: DatasetConfig,
    *,
    partition_strategy: str = "iid",
    dirichlet_alpha: float | None = None,
    base_output: Path | None = None,
) -> ExperimentContext:
    set_seed(config.seed)
    data, num_features, num_classes, metadata = load_dataset(project_dir, config)
    data.global_node_id = torch.arange(data.num_nodes)
    partition_base = base_output or (
        project_dir / "outputs" / "final_multidataset" / f"federated_{config.key}"
    )
    partition = load_or_create_partition(
        project_dir,
        config,
        data,
        num_classes,
        base_output=partition_base,
        strategy=partition_strategy,
        dirichlet_alpha=dirichlet_alpha,
        target_class=config.target_class,
        malicious_client_id=config.malicious_client_id,
    )
    clean_graphs = build_client_graphs(data, partition, config.num_clients)
    attacked_graph, attack_metadata = inject_clique_trigger(
        clean_graphs[config.malicious_client_id],
        target_class=config.target_class,
        poison_rate=config.poison_rate,
        trigger_size=config.trigger_size,
        seed=config.seed,
    )
    attacked_graphs = list(clean_graphs)
    attacked_graphs[config.malicious_client_id] = attacked_graph

    victim_mask = data.test_mask & data.y.ne(config.target_class)
    victim_nodes = victim_mask.nonzero(as_tuple=False).view(-1)
    prototype = make_sparse_trigger_prototype(
        clean_graphs[config.malicious_client_id].x,
        seed=config.seed,
        active_features=16,
    )
    triggered_data = attach_evaluation_triggers(
        data,
        victim_nodes=victim_nodes,
        trigger_size=config.trigger_size,
        feature_prototype=prototype,
        target_class=config.target_class,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return ExperimentContext(
        project_dir,
        config,
        data,
        num_features,
        num_classes,
        metadata,
        clean_graphs,
        attacked_graphs,
        attack_metadata,
        device,
        triggered_data,
        victim_nodes,
        base_output_override=base_output,
        partition_metadata={
            "strategy": partition_strategy,
            "dirichlet_alpha": dirichlet_alpha,
        },
    )


def accuracy(logits: Tensor, labels: Tensor, mask: Tensor) -> float:
    total = int(mask.sum())
    if total == 0:
        return 0.0
    return float(logits.argmax(dim=1)[mask].eq(labels[mask]).float().mean().item())


@torch.no_grad()
def evaluate(
    model: GCN,
    clean_data: Data,
    triggered_data: Data,
    victim_nodes: Tensor,
    target_class: int,
) -> dict[str, float]:
    model.eval()
    clean_logits = model(clean_data.x, clean_data.edge_index)
    triggered_logits = model(triggered_data.x, triggered_data.edge_index)
    clean_pred = clean_logits.argmax(dim=1)
    trigger_pred = triggered_logits.argmax(dim=1)
    clean_target = float(clean_pred[victim_nodes].eq(target_class).float().mean().item())
    asr = float(trigger_pred[victim_nodes].eq(target_class).float().mean().item())
    return {
        "train_accuracy": accuracy(clean_logits, clean_data.y, clean_data.train_mask),
        "validation_accuracy": accuracy(clean_logits, clean_data.y, clean_data.val_mask),
        "test_accuracy": accuracy(clean_logits, clean_data.y, clean_data.test_mask),
        "clean_target_rate": clean_target,
        "attack_success_rate": asr,
        "attack_effect": asr - clean_target,
    }


def standard_local_train(
    global_state: StateDict,
    client_data: Data,
    input_channels: int,
    output_channels: int,
    device: torch.device,
    round_number: int,
    config: DatasetConfig,
) -> tuple[StateDict, float, int, dict[str, float]]:
    del round_number
    model = GCN(input_channels, config.hidden_channels, output_channels, config.dropout).to(device)
    model.load_state_dict(global_state)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_count = int(client_data.train_mask.sum())
    last_loss = 0.0
    for _ in range(config.local_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(client_data.x, client_data.edge_index)
        loss = F.cross_entropy(logits[client_data.train_mask], client_data.y[client_data.train_mask])
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
    return clone_state(model), last_loss, train_count, {}


def fedavg(
    local_states: list[StateDict],
    counts: list[int],
    global_state: StateDict,
    round_number: int,
) -> tuple[StateDict, dict[str, float]]:
    del global_state, round_number
    weights = torch.tensor(counts, dtype=torch.float64)
    weights /= weights.sum()
    averaged: StateDict = OrderedDict()
    for key, reference in local_states[0].items():
        if not torch.is_floating_point(reference):
            averaged[key] = reference.clone()
            continue
        output = torch.zeros_like(reference)
        for state, weight in zip(local_states, weights.tolist(), strict=True):
            output.add_(state[key], alpha=float(weight))
        averaged[key] = output
    return averaged, {}


def _save_history(rows: list[dict[str, Any]], path: Path) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_method(
    *,
    context: ExperimentContext,
    method_name: str,
    output_slug: str,
    client_graphs_cpu: list[Data],
    local_train_fn: LocalTrainFn = standard_local_train,
    aggregate_fn: AggregateFn = fedavg,
    method_metadata: dict[str, Any] | None = None,
    model_cls: type[nn.Module] = GCN,
) -> dict[str, Any]:
    # Reset the RNG at the beginning of every compared method so all methods
    # within the same experimental condition start from a reproducible and
    # comparable stochastic state.
    set_seed(context.config.seed)

    output_dir = context.base_output / output_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "final_summary.json"
    history_path = output_dir / "training_history.csv"
    checkpoint_path = output_dir / "best_model.pt"

    client_graphs = [graph.to(context.device) for graph in client_graphs_cpu]
    model = model_cls(
        context.num_features,
        context.config.hidden_channels,
        context.num_classes,
        context.config.dropout,
    ).to(context.device)

    best_validation = -1.0
    best_round = 0
    history: list[dict[str, Any]] = []
    start = time.perf_counter()

    print("=" * 100)
    print(f"{method_name} | {context.config.display_name}")
    print("=" * 100)
    for round_number in range(1, context.config.rounds + 1):
        global_state = clone_state(model)
        local_states: list[StateDict] = []
        counts: list[int] = []
        losses: list[float] = []
        local_diags: list[dict[str, float]] = []
        for client_data in client_graphs:
            state, loss, count, diag = local_train_fn(
                global_state,
                client_data,
                context.num_features,
                context.num_classes,
                context.device,
                round_number,
                context.config,
            )
            local_states.append(state)
            losses.append(loss)
            counts.append(count)
            local_diags.append(diag)

        try:
            averaged, aggregation_diag = aggregate_fn(
                local_states, counts, global_state, round_number, local_diags=local_diags
            )
        except TypeError:
            averaged, aggregation_diag = aggregate_fn(local_states, counts, global_state, round_number)
        model.load_state_dict(averaged)
        metrics = evaluate(
            model,
            context.clean_data,
            context.triggered_data,
            context.victim_nodes,
            context.config.target_class,
        )
        weighted_loss = sum(loss * count for loss, count in zip(losses, counts, strict=True)) / sum(counts)
        row: dict[str, Any] = {
            "round": round_number,
            "mean_client_loss": weighted_loss,
            **metrics,
            **aggregation_diag,
        }
        for client_id, diag in enumerate(local_diags):
            for key, value in diag.items():
                row[f"client_{client_id}_{key}"] = value
        history.append(row)

        if metrics["validation_accuracy"] > best_validation:
            best_validation = metrics["validation_accuracy"]
            best_round = round_number
            torch.save(
                {
                    "round": round_number,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "config": asdict(context.config),
                    "method": method_name,
                },
                checkpoint_path,
            )
        if round_number == 1 or round_number % 5 == 0:
            print(
                f"Round {round_number:03d} | Loss {weighted_loss:.4f} | "
                f"Test {metrics['test_accuracy']:.4f} | "
                f"ASR {metrics['attack_success_rate']:.4f}"
            )

    elapsed = time.perf_counter() - start
    _save_history(history, history_path)
    checkpoint = torch.load(checkpoint_path, map_location=context.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = evaluate(
        model,
        context.clean_data,
        context.triggered_data,
        context.victim_nodes,
        context.config.target_class,
    )
    summary = {
        "dataset": context.config.display_name,
        "method": method_name,
        "output_slug": output_slug,
        "best_round": best_round,
        "final_metrics": final_metrics,
        "elapsed_seconds": elapsed,
        "config": asdict(context.config),
        "dataset_metadata": context.metadata,
        "partition_metadata": context.partition_metadata,
        "attack_metadata": context.attack_metadata,
        "method_metadata": method_metadata or {},
        "history_path": str(history_path),
        "checkpoint_path": str(checkpoint_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")
    return summary
