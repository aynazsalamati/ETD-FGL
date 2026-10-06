from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge

from baseline_suite.common import StateDict
from baseline_suite.config import ExperimentConfig
from train_fedavg_cora import GCN


def _encode(
    model: GCN,
    x: Tensor,
    edge_index: Tensor,
) -> Tensor:
    hidden = model.conv1(x, edge_index)
    return F.relu(hidden)


def _nt_xent(
    first: Tensor,
    second: Tensor,
    temperature: float = 0.5,
) -> Tensor:
    if first.size(0) != second.size(0):
        raise ValueError("Contrastive views must have equal batch size.")
    if first.size(0) < 2:
        return first.new_tensor(0.0)

    first = F.normalize(first, dim=1)
    second = F.normalize(second, dim=1)

    logits = first @ second.t() / temperature
    labels = torch.arange(
        first.size(0),
        device=first.device,
    )

    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.t(), labels)
    )


def make_flpurifier_local_trainer(
    config: ExperimentConfig,
    *,
    contrastive_epochs: int = 2,
    classifier_epochs: int = 3,
    edge_drop_rate: float = 0.20,
    feature_drop_rate: float = 0.20,
    temperature: float = 0.5,
):
    """Create a decoupled contrastive local trainer for a two-layer GCN."""

    if contrastive_epochs + classifier_epochs != config.local_epochs:
        raise ValueError(
            "The contrastive and classifier epochs must sum to local_epochs."
        )

    def train_local(
        global_state: StateDict,
        client_data: Data,
        input_channels: int,
        output_channels: int,
        device: torch.device,
        round_number: int,
    ) -> tuple[StateDict, float, int, dict[str, float]]:
        del round_number

        model = GCN(
            input_channels=input_channels,
            hidden_channels=config.hidden_channels,
            output_channels=output_channels,
            dropout=config.dropout,
        ).to(device)
        model.load_state_dict(global_state)

        train_mask = client_data.train_mask
        train_count = int(train_mask.sum().item())
        if train_count == 0:
            raise ValueError("Client has no training nodes.")

        # Stage 1: train only the feature extractor with two stochastic
        # graph views, breaking a brittle trigger-to-target correlation.
        for parameter in model.conv2.parameters():
            parameter.requires_grad_(False)
        for parameter in model.conv1.parameters():
            parameter.requires_grad_(True)

        extractor_optimizer = torch.optim.Adam(
            model.conv1.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        last_contrastive = 0.0
        for _ in range(contrastive_epochs):
            model.train()
            extractor_optimizer.zero_grad()

            edge_first, _ = dropout_edge(
                client_data.edge_index,
                p=edge_drop_rate,
                force_undirected=False,
                training=True,
            )
            edge_second, _ = dropout_edge(
                client_data.edge_index,
                p=edge_drop_rate,
                force_undirected=False,
                training=True,
            )

            x_first = F.dropout(
                client_data.x,
                p=feature_drop_rate,
                training=True,
            )
            x_second = F.dropout(
                client_data.x,
                p=feature_drop_rate,
                training=True,
            )

            embedding_first = _encode(model, x_first, edge_first)[train_mask]
            embedding_second = _encode(model, x_second, edge_second)[train_mask]

            contrastive_loss = _nt_xent(
                embedding_first,
                embedding_second,
                temperature=temperature,
            )
            contrastive_loss.backward()
            extractor_optimizer.step()
            last_contrastive = float(contrastive_loss.item())

        # Stage 2: freeze the extractor and train only the classifier.
        for parameter in model.conv1.parameters():
            parameter.requires_grad_(False)
        for parameter in model.conv2.parameters():
            parameter.requires_grad_(True)

        classifier_optimizer = torch.optim.Adam(
            model.conv2.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        last_classifier = 0.0
        for _ in range(classifier_epochs):
            model.train()
            classifier_optimizer.zero_grad()

            with torch.no_grad():
                hidden = _encode(
                    model,
                    client_data.x,
                    client_data.edge_index,
                )

            logits = model.conv2(
                hidden,
                client_data.edge_index,
            )
            classifier_loss = F.cross_entropy(
                logits[train_mask],
                client_data.y[train_mask],
            )
            classifier_loss.backward()
            classifier_optimizer.step()
            last_classifier = float(classifier_loss.item())

        local_state: StateDict = OrderedDict(
            (
                key,
                value.detach().clone(),
            )
            for key, value in model.state_dict().items()
        )

        diagnostics = {
            "contrastive_loss": last_contrastive,
            "classifier_loss": last_classifier,
        }

        return (
            local_state,
            last_classifier,
            train_count,
            diagnostics,
        )

    return train_local


def _flatten_delta(
    state: StateDict,
    global_state: StateDict,
    keys: list[str],
) -> Tensor:
    chunks = [
        (state[key] - global_state[key]).detach().float().reshape(-1)
        for key in keys
        if torch.is_floating_point(state[key])
    ]
    if not chunks:
        raise ValueError("No floating classifier parameters found.")
    return torch.cat(chunks)


class FLPurifierAggregator:
    """
    Decoupled aggregation for the GCN adaptation of FLPurifier.

    Extractor parameters use sample-size FedAvg. Classifier updates are
    weighted by their cosine agreement with the coordinate-wise median
    classifier update, reducing the influence of a backdoor classifier.
    """

    def __init__(self, minimum_similarity: float = 0.05) -> None:
        self.minimum_similarity = minimum_similarity

    def __call__(
        self,
        local_states: list[StateDict],
        client_counts: list[int],
        global_state: StateDict,
        round_number: int,
    ) -> tuple[StateDict, dict[str, float]]:
        del round_number

        extractor_keys = [
            key for key in local_states[0]
            if key.startswith("conv1")
        ]
        classifier_keys = [
            key for key in local_states[0]
            if key.startswith("conv2")
        ]
        remaining_keys = [
            key for key in local_states[0]
            if key not in extractor_keys and key not in classifier_keys
        ]

        sample_weights = np.asarray(client_counts, dtype=np.float64)
        sample_weights /= sample_weights.sum()

        deltas = torch.stack(
            [
                _flatten_delta(state, global_state, classifier_keys)
                for state in local_states
            ]
        )
        median_delta = deltas.median(dim=0).values
        median_norm = median_delta.norm().clamp_min(1e-12)

        similarities: list[float] = []
        for delta in deltas:
            similarity = float(
                torch.dot(delta, median_delta)
                / (delta.norm().clamp_min(1e-12) * median_norm)
            )
            similarities.append(similarity)

        agreement = np.maximum(
            np.asarray(similarities, dtype=np.float64),
            self.minimum_similarity,
        )
        classifier_weights = sample_weights * agreement
        classifier_weights /= classifier_weights.sum()

        averaged: StateDict = OrderedDict()

        for key in extractor_keys:
            reference = local_states[0][key]
            if not torch.is_floating_point(reference):
                averaged[key] = reference.clone()
                continue
            output = torch.zeros_like(reference)
            for state, weight in zip(
                local_states,
                sample_weights,
                strict=True,
            ):
                output.add_(state[key], alpha=float(weight))
            averaged[key] = output

        for key in classifier_keys:
            reference = local_states[0][key]
            if not torch.is_floating_point(reference):
                averaged[key] = reference.clone()
                continue
            output = torch.zeros_like(reference)
            for state, weight in zip(
                local_states,
                classifier_weights,
                strict=True,
            ):
                output.add_(state[key], alpha=float(weight))
            averaged[key] = output

        for key in remaining_keys:
            averaged[key] = local_states[0][key].clone()

        diagnostics: dict[str, float] = {}
        for client_id, similarity in enumerate(similarities):
            diagnostics[
                f"flpurifier_classifier_similarity_client_{client_id}"
            ] = float(similarity)
            diagnostics[
                f"flpurifier_classifier_weight_client_{client_id}"
            ] = float(classifier_weights[client_id])

        diagnostics["flpurifier_min_classifier_similarity"] = float(
            np.min(similarities)
        )
        diagnostics["flpurifier_suspected_client"] = float(
            int(np.argmin(similarities))
        )

        return averaged, diagnostics
