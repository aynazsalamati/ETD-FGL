from pathlib import Path

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def main() -> None:
    """Download Cora and print its main properties."""

    project_dir = Path(__file__).resolve().parent
    dataset_dir = project_dir / "data" / "Planetoid"

    print("Loading Cora dataset...")

    dataset = Planetoid(
        root=str(dataset_dir),
        name="Cora",
        transform=NormalizeFeatures(),
    )

    data = dataset[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    print("\nDataset loaded successfully.")
    print("-" * 45)
    print(f"Dataset name:        {dataset.name}")
    print(f"Number of graphs:    {len(dataset)}")
    print(f"Number of nodes:     {data.num_nodes}")
    print(f"Number of edges:     {data.num_edges}")
    print(f"Number of features:  {dataset.num_features}")
    print(f"Number of classes:   {dataset.num_classes}")
    print(f"Training nodes:      {int(data.train_mask.sum())}")
    print(f"Validation nodes:    {int(data.val_mask.sum())}")
    print(f"Test nodes:          {int(data.test_mask.sum())}")
    print(f"Device:              {data.x.device}")
    print("-" * 45)

    # Basic consistency checks
    assert data.x.shape[0] == data.num_nodes
    assert data.y.shape[0] == data.num_nodes
    assert data.edge_index.shape[0] == 2
    assert data.train_mask.dtype == torch.bool

    print("All consistency checks passed.")


if __name__ == "__main__":
    main()