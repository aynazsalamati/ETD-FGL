"""Lightweight environment check for ETD-FGL.

Run this after installing PyTorch and requirements.txt. It does not download
any datasets or launch training.
"""
from __future__ import annotations

import importlib
import platform
import sys

REQUIRED_MODULES = [
    ("torch", "PyTorch"),
    ("torch_geometric", "PyTorch Geometric"),
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("sklearn", "scikit-learn"),
    ("pandas", "pandas"),
    ("matplotlib", "Matplotlib"),
    ("networkx", "NetworkX"),
    ("tqdm", "tqdm"),
    ("gudhi", "GUDHI"),
    ("joblib", "joblib"),
]


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    failed = []
    loaded = {}
    for module_name, display_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            loaded[module_name] = module
            print(f"[OK] {display_name}: {version}")
        except Exception as exc:  # pragma: no cover - diagnostic script
            failed.append(display_name)
            print(f"[MISSING] {display_name}: {exc}")

    torch = loaded.get("torch")
    if torch is not None:
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"PyTorch CUDA build: {torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    if failed:
        print("\nEnvironment check failed. Missing/broken: " + ", ".join(failed))
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
