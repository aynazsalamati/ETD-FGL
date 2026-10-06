from __future__ import annotations

import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv

StateDict = OrderedDict[str, Tensor]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GCN(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels, cached=False)
        self.conv2 = GCNConv(hidden_channels, output_channels, cached=False)
        self.dropout = float(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


def clone_state(model: torch.nn.Module) -> StateDict:
    return OrderedDict(
        (key, value.detach().clone())
        for key, value in model.state_dict().items()
    )
