from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.explain import Explainer, GNNExplainer

from train_fedavg_cora import GCN


SEED = 42
MALICIOUS_CLIENT_ID = 0

HIDDEN_CHANNELS = 16
DROPOUT = 0.5

EXPLAINER_EPOCHS = 100
EXPLAINER_LEARNING_RATE = 0.01
TOP_K_EDGES = 10
EPSILON = 1e-12


def calculate_edge_concentration(
    edge_mask: Tensor,
    top_k: int,
) -> float:
    """
    Calculate the fraction of total edge importance concentrated
    in the top-k edges.
    """

    scores = edge_mask.detach().abs().flatten()

    if scores.numel() == 0:
        return 0.0

    effective_top_k = min(top_k, scores.numel())

    top_values = torch.topk(
        scores,
        k=effective_top_k,
    ).values

    return float(
        top_values.sum().item()
        / (scores.sum().item() + EPSILON)
    )


def calculate_feature_concentration(
    node_mask: Tensor | None,
) -> float:
    """
    Calculate the maximum feature attribution relative to the
    total feature attribution.
    """

    if node_mask is None:
        return 0.0

    scores = node_mask.detach().abs()

    if scores.dim() == 2:
        feature_scores = scores.mean(dim=0)
    else:
        feature_scores = scores.flatten()

    total_importance = float(
        feature_scores.sum().item()
    )

    if total_importance <= EPSILON:
        return 0.0

    return float(
        feature_scores.max().item()
        / total_importance
    )


def calculate_trigger_edge_ratio(
    edge_index: Tensor,
    edge_mask: Tensor,
    trigger_node_mask: Tensor,
) -> float:
    """
    Calculate how much attribution is assigned to edges connected
    to injected trigger nodes.

    This metric is used only to verify the pilot implementation.
    It is not available to the defense during a real unknown attack.
    """

    source = edge_index[0]
    target = edge_index[1]

    trigger_edges = (
        trigger_node_mask[source]
        | trigger_node_mask[target]
    )

    scores = edge_mask.detach().abs().flatten()

    total_importance = float(
        scores.sum().item()
    )

    if total_importance <= EPSILON:
        return 0.0

    trigger_importance = float(
        scores[trigger_edges].sum().item()
    )

    return trigger_importance / total_importance


@torch.no_grad()
def calculate_original_prediction(
    model: GCN,
    x: Tensor,
    edge_index: Tensor,
    node_index: int,
) -> tuple[int, float]:
    """Return predicted class and confidence for one node."""

    model.eval()

    logits = model(x, edge_index)
    probabilities = F.softmax(logits, dim=1)

    predicted_class = int(
        probabilities[node_index].argmax().item()
    )

    confidence = float(
        probabilities[
            node_index,
            predicted_class,
        ].item()
    )

    return predicted_class, confidence


@torch.no_grad()
def calculate_fidelity_drop(
    model: GCN,
    x: Tensor,
    edge_index: Tensor,
    edge_mask: Tensor,
    node_index: int,
    predicted_class: int,
    original_confidence: float,
    top_k: int,
) -> float:
    """
    Remove the top-k explanatory edges and measure the decrease
    in confidence for the original predicted class.
    """

    scores = edge_mask.detach().abs().flatten()

    effective_top_k = min(
        top_k,
        scores.numel(),
    )

    important_edge_indices = torch.topk(
        scores,
        k=effective_top_k,
    ).indices

    keep_mask = torch.ones(
        edge_index.size(1),
        dtype=torch.bool,
        device=edge_index.device,
    )

    keep_mask[important_edge_indices] = False

    reduced_edge_index = edge_index[:, keep_mask]

    model.eval()

    reduced_logits = model(
        x,
        reduced_edge_index,
    )

    reduced_probabilities = F.softmax(
        reduced_logits,
        dim=1,
    )

    reduced_confidence = float(
        reduced_probabilities[
            node_index,
            predicted_class,
        ].item()
    )

    return original_confidence - reduced_confidence


def explain_node(
    explainer: Explainer,
    model: GCN,
    data,
    node_index: int,
    node_type: str,
) -> dict[str, Any]:
    """Generate and summarize one node explanation."""

    predicted_class, original_confidence = (
        calculate_original_prediction(
            model=model,
            x=data.x,
            edge_index=data.edge_index,
            node_index=node_index,
        )
    )

    explanation = explainer(
        data.x,
        data.edge_index,
        index=node_index,
    )

    if explanation.edge_mask is None:
        raise RuntimeError(
            "GNNExplainer did not return an edge mask."
        )

    edge_mask = explanation.edge_mask

    edge_concentration = calculate_edge_concentration(
        edge_mask=edge_mask,
        top_k=TOP_K_EDGES,
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

    fidelity_drop = calculate_fidelity_drop(
        model=model,
        x=data.x,
        edge_index=data.edge_index,
        edge_mask=edge_mask,
        node_index=node_index,
        predicted_class=predicted_class,
        original_confidence=original_confidence,
        top_k=TOP_K_EDGES,
    )

    nonzero_edge_count = int(
        edge_mask.detach()
        .gt(1e-6)
        .sum()
        .item()
    )

    return {
        "node_type": node_type,
        "node_index": node_index,
        "true_label": int(
            data.clean_y[node_index].item()
        ),
        "training_label": int(
            data.y[node_index].item()
        ),
        "predicted_class": predicted_class,
        "prediction_confidence": (
            original_confidence
        ),
        "edge_mask_size": int(
            edge_mask.numel()
        ),
        "nonzero_edge_count": (
            nonzero_edge_count
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
        "fidelity_drop": fidelity_drop,
    }


def select_clean_probe_node(
    data,
    poisoned_node_index: int,
) -> int:
    """
    Select a clean non-target training node for comparison.
    """

    candidate_mask = (
        data.train_mask
        & ~data.poisoned_node_mask
        & data.clean_y.ge(0)
    )

    candidate_nodes = candidate_mask.nonzero(
        as_tuple=False
    ).view(-1)

    if candidate_nodes.numel() == 0:
        raise RuntimeError(
            "No clean probe node was found."
        )

    poisoned_true_label = int(
        data.clean_y[
            poisoned_node_index
        ].item()
    )

    same_class_candidates = candidate_nodes[
        data.clean_y[candidate_nodes].eq(
            poisoned_true_label
        )
    ]

    if same_class_candidates.numel() > 0:
        return int(
            same_class_candidates[0].item()
        )

    return int(candidate_nodes[0].item())


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
        / "single_probe_explanation_report.json"
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

    data = attacked_package["data"].to(device)
    metadata = attacked_package["metadata"]

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    input_channels = int(data.x.size(1))
    output_channels = int(
        checkpoint["model_state_dict"][
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
        checkpoint["model_state_dict"]
    )

    model.eval()

    poisoned_probe_node = int(
        metadata["poisoned_victim_nodes"][0]
    )

    clean_probe_node = select_clean_probe_node(
        data=data,
        poisoned_node_index=poisoned_probe_node,
    )

    explainer = Explainer(
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

    print("=" * 76)
    print("Testing GNN explanation module")
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
        explainer=explainer,
        model=model,
        data=data,
        node_index=poisoned_probe_node,
        node_type="poisoned",
    )

    print(
        "Generating clean-node explanation..."
    )

    clean_report = explain_node(
        explainer=explainer,
        model=model,
        data=data,
        node_index=clean_probe_node,
        node_type="clean",
    )

    complete_report = {
        "configuration": {
            "seed": SEED,
            "explainer": "GNNExplainer",
            "explainer_epochs": (
                EXPLAINER_EPOCHS
            ),
            "top_k_edges": TOP_K_EDGES,
        },
        "poisoned_probe": poisoned_report,
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

    for report in [
        poisoned_report,
        clean_report,
    ]:
        print(
            f"{report['node_type'].upper():8s} NODE "
            f"{report['node_index']:3d} | "
            f"True: {report['true_label']} | "
            f"Train label: "
            f"{report['training_label']} | "
            f"Pred: {report['predicted_class']} | "
            f"Confidence: "
            f"{report['prediction_confidence']:.4f}"
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
            f"    Fidelity drop:            "
            f"{report['fidelity_drop']:+.4f}"
        )

    print("-" * 76)
    print(f"Report saved to: {report_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()