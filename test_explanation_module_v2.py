from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.metric import unfaithfulness
from torch_geometric.utils import degree

from test_explanation_module import (
    calculate_edge_concentration,
    calculate_feature_concentration,
    calculate_trigger_edge_ratio,
)
from train_fedavg_cora import GCN


SEED = 42
MALICIOUS_CLIENT_ID = 0
TARGET_CLASS = 0

HIDDEN_CHANNELS = 16
DROPOUT = 0.5

EXPLAINER_EPOCHS = 100
EXPLAINER_LEARNING_RATE = 0.01
TOP_K_EDGES = 10


def build_explainer(model: GCN) -> Explainer:
    """Create a fresh GNNExplainer instance."""

    return Explainer(
        model=model,
        algorithm=GNNExplainer(
            epochs=EXPLAINER_EPOCHS,
            lr=EXPLAINER_LEARNING_RATE,
        ),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config={
            "mode": "multiclass_classification",
            "task_level": "node",
            "return_type": "raw",
        },
    )


@torch.no_grad()
def get_predictions(
    model: GCN,
    data,
) -> tuple[Tensor, Tensor]:
    """Return predicted classes and probabilities."""

    model.eval()

    logits = model(
        data.x,
        data.edge_index,
    )

    probabilities = F.softmax(
        logits,
        dim=1,
    )

    predictions = probabilities.argmax(
        dim=1
    )

    return predictions, probabilities


@torch.no_grad()
def select_clean_probe_node(
    model: GCN,
    data,
    poisoned_node_index: int,
) -> int:
    """
    Select a useful clean comparison node.

    The selected node must:
    1. Be an original clean node
    2. Not be poisoned
    3. Have at least one local edge
    4. Be classified correctly by the current model

    A correctly classified node from the same true class as the
    poisoned node is preferred.
    """

    predictions, _ = get_predictions(
        model=model,
        data=data,
    )

    node_degrees = degree(
        data.edge_index[0],
        num_nodes=data.num_nodes,
        dtype=torch.float,
    )

    original_clean_mask = (
        data.clean_y.ge(0)
        & ~data.poisoned_node_mask
        & ~data.trigger_node_mask
    )

    correctly_classified_mask = (
        predictions.eq(data.clean_y)
    )

    connected_mask = node_degrees.gt(0)

    valid_mask = (
        original_clean_mask
        & correctly_classified_mask
        & connected_mask
    )

    poisoned_true_label = int(
        data.clean_y[
            poisoned_node_index
        ].item()
    )

    # First preference: correctly classified node
    # from the same class as the poisoned node.
    preferred_mask = (
        valid_mask
        & data.clean_y.eq(
            poisoned_true_label
        )
    )

    preferred_nodes = preferred_mask.nonzero(
        as_tuple=False
    ).view(-1)

    if preferred_nodes.numel() > 0:
        preferred_degrees = node_degrees[
            preferred_nodes
        ]

        selected_position = int(
            preferred_degrees.argmax().item()
        )

        return int(
            preferred_nodes[
                selected_position
            ].item()
        )

    # Second preference: any correctly classified
    # non-target clean node.
    fallback_mask = (
        valid_mask
        & data.clean_y.ne(TARGET_CLASS)
    )

    fallback_nodes = fallback_mask.nonzero(
        as_tuple=False
    ).view(-1)

    if fallback_nodes.numel() > 0:
        fallback_degrees = node_degrees[
            fallback_nodes
        ]

        selected_position = int(
            fallback_degrees.argmax().item()
        )

        return int(
            fallback_nodes[
                selected_position
            ].item()
        )

    # Final fallback: any correctly classified connected node.
    valid_nodes = valid_mask.nonzero(
        as_tuple=False
    ).view(-1)

    if valid_nodes.numel() == 0:
        raise RuntimeError(
            "No correctly classified connected clean node "
            "was found for comparison."
        )

    valid_degrees = node_degrees[
        valid_nodes
    ]

    selected_position = int(
        valid_degrees.argmax().item()
    )

    return int(
        valid_nodes[
            selected_position
        ].item()
    )


def explain_node(
    model: GCN,
    data,
    node_index: int,
    node_type: str,
    seed: int,
) -> dict[str, Any]:
    """Generate and summarize one node explanation."""

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    predictions, probabilities = get_predictions(
        model=model,
        data=data,
    )

    predicted_class = int(
        predictions[node_index].item()
    )

    prediction_confidence = float(
        probabilities[
            node_index,
            predicted_class,
        ].item()
    )

    node_degrees = degree(
        data.edge_index[0],
        num_nodes=data.num_nodes,
        dtype=torch.float,
    )

    explainer = build_explainer(model)

    explanation = explainer(
        data.x,
        data.edge_index,
        index=node_index,
    )

    if explanation.edge_mask is None:
        raise RuntimeError(
            "GNNExplainer did not produce an edge mask."
        )

    edge_mask = explanation.edge_mask

    edge_concentration = (
        calculate_edge_concentration(
            edge_mask=edge_mask,
            top_k=TOP_K_EDGES,
        )
    )

    feature_concentration = (
        calculate_feature_concentration(
            explanation.node_mask
        )
    )

    trigger_edge_ratio = (
        calculate_trigger_edge_ratio(
            edge_index=data.edge_index,
            edge_mask=edge_mask,
            trigger_node_mask=(
                data.trigger_node_mask
            ),
        )
    )

    official_unfaithfulness = float(
        unfaithfulness(
            explainer,
            explanation,
        )
    )

    nonzero_edge_count = int(
        edge_mask.detach()
        .gt(1e-6)
        .sum()
        .item()
    )

    total_edge_importance = float(
        edge_mask.detach()
        .abs()
        .sum()
        .item()
    )

    return {
        "node_type": node_type,
        "node_index": node_index,
        "node_degree": float(
            node_degrees[node_index].item()
        ),
        "true_label": int(
            data.clean_y[node_index].item()
        ),
        "training_label": int(
            data.y[node_index].item()
        ),
        "predicted_class": predicted_class,
        "prediction_correct": bool(
            predicted_class
            == int(
                data.clean_y[node_index].item()
            )
        ),
        "prediction_confidence": (
            prediction_confidence
        ),
        "edge_mask_size": int(
            edge_mask.numel()
        ),
        "nonzero_edge_count": (
            nonzero_edge_count
        ),
        "total_edge_importance": (
            total_edge_importance
        ),
        "top_k_edge_concentration": (
            edge_concentration
        ),
        "feature_concentration": (
            feature_concentration
        ),
        "trigger_edge_attribution_ratio": (
            trigger_edge_ratio
        ),
        "unfaithfulness": (
            official_unfaithfulness
        ),
    }


def print_report(
    report: dict[str, Any],
) -> None:
    """Print one explanation report."""

    correctness = (
        "CORRECT"
        if report["prediction_correct"]
        else "WRONG"
    )

    print(
        f"{report['node_type'].upper():8s} NODE "
        f"{report['node_index']:3d} | "
        f"Degree: {report['node_degree']:.0f} | "
        f"True: {report['true_label']} | "
        f"Train label: {report['training_label']} | "
        f"Pred: {report['predicted_class']} | "
        f"{correctness}"
    )

    print(
        f"    Confidence:               "
        f"{report['prediction_confidence']:.4f}"
    )

    print(
        f"    Nonzero explained edges:  "
        f"{report['nonzero_edge_count']}"
    )

    print(
        f"    Total edge importance:    "
        f"{report['total_edge_importance']:.4f}"
    )

    print(
        f"    Edge concentration:       "
        f"{report['top_k_edge_concentration']:.4f}"
    )

    print(
        f"    Feature concentration:    "
        f"{report['feature_concentration']:.4f}"
    )

    print(
        f"    Trigger-edge attribution: "
        f"{report['trigger_edge_attribution_ratio']:.4f}"
    )

    print(
        f"    Unfaithfulness:            "
        f"{report['unfaithfulness']:.4f}"
    )


def main() -> None:
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    project_dir = Path(__file__).resolve().parent

    attacked_graph_path = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "attacks"
        / "structural_clique"
        / f"malicious_client_{MALICIOUS_CLIENT_ID}.pt"
    )

    model_path = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "fedavg_structural_backdoor"
        / "best_backdoored_fedavg_cora.pt"
    )

    output_dir = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "explanation_module"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        output_dir
        / "single_probe_explanation_report_v2.json"
    )

    if not attacked_graph_path.exists():
        raise FileNotFoundError(
            f"Attacked graph not found:\n"
            f"{attacked_graph_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Backdoored model not found:\n"
            f"{model_path}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    attacked_package = torch.load(
        attacked_graph_path,
        map_location="cpu",
        weights_only=False,
    )

    data = attacked_package[
        "data"
    ].to(device)

    metadata = attacked_package[
        "metadata"
    ]

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    input_channels = int(
        data.x.size(1)
    )

    output_channels = int(
        checkpoint[
            "model_state_dict"
        ][
            "conv2.bias"
        ].numel()
    )

    model = GCN(
        input_channels=input_channels,
        hidden_channels=HIDDEN_CHANNELS,
        output_channels=output_channels,
        dropout=DROPOUT,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    poisoned_probe_node = int(
        metadata[
            "poisoned_victim_nodes"
        ][0]
    )

    clean_probe_node = (
        select_clean_probe_node(
            model=model,
            data=data,
            poisoned_node_index=(
                poisoned_probe_node
            ),
        )
    )

    print("=" * 76)
    print("Testing corrected GNN explanation module")
    print("=" * 76)
    print(f"Device:                 {device}")
    print(
        f"Poisoned probe node:    "
        f"{poisoned_probe_node}"
    )
    print(
        f"Clean probe node:       "
        f"{clean_probe_node}"
    )
    print(
        f"Explainer epochs:       "
        f"{EXPLAINER_EPOCHS}"
    )
    print(
        "Generating poisoned-node explanation..."
    )

    poisoned_report = explain_node(
        model=model,
        data=data,
        node_index=poisoned_probe_node,
        node_type="poisoned",
        seed=SEED,
    )

    print(
        "Generating clean-node explanation..."
    )

    clean_report = explain_node(
        model=model,
        data=data,
        node_index=clean_probe_node,
        node_type="clean",
        seed=SEED + 1,
    )

    complete_report = {
        "configuration": {
            "seed": SEED,
            "explainer": "GNNExplainer",
            "explainer_epochs": (
                EXPLAINER_EPOCHS
            ),
            "top_k_edges": TOP_K_EDGES,
            "metric": (
                "PyG official unfaithfulness"
            ),
        },
        "poisoned_probe": (
            poisoned_report
        ),
        "clean_probe": clean_report,
    }

    with report_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            complete_report,
            json_file,
            indent=2,
        )

    print("-" * 76)

    print_report(poisoned_report)

    print("-" * 76)

    print_report(clean_report)

    print("-" * 76)

    trigger_gap = (
        poisoned_report[
            "trigger_edge_attribution_ratio"
        ]
        - clean_report[
            "trigger_edge_attribution_ratio"
        ]
    )

    concentration_gap = (
        poisoned_report[
            "top_k_edge_concentration"
        ]
        - clean_report[
            "top_k_edge_concentration"
        ]
    )

    print(
        f"Trigger-attribution gap: "
        f"{trigger_gap:+.4f}"
    )

    print(
        f"Edge-concentration gap:  "
        f"{concentration_gap:+.4f}"
    )

    print(f"Report saved to: {report_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()