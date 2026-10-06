from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


NUMERIC_COLUMNS = {
    "round",
    "mean_client_loss",
    "train_accuracy",
    "validation_accuracy",
    "test_accuracy",
    "clean_target_rate",
    "attack_success_rate",
    "attack_effect",
}


def _parse_value(key: str, raw: str) -> Any:
    text = raw.strip()

    if text == "":
        return None

    if key == "round":
        return int(float(text))

    if key in NUMERIC_COLUMNS:
        return float(text)

    try:
        return float(text)
    except ValueError:
        return text


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [
            {
                key: _parse_value(key, value or "")
                for key, value in row.items()
            }
            for row in reader
        ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_figure_all_formats(figure, output_base: Path) -> list[str]:
    """Save PNG 600 DPI plus vector PDF and SVG."""

    output_base.parent.mkdir(parents=True, exist_ok=True)

    paths = {
        "png": output_base.with_suffix(".png"),
        "pdf": output_base.with_suffix(".pdf"),
        "svg": output_base.with_suffix(".svg"),
    }

    figure.savefig(paths["png"], dpi=600, bbox_inches="tight")
    figure.savefig(paths["pdf"], bbox_inches="tight")
    figure.savefig(paths["svg"], bbox_inches="tight")

    return [str(path) for path in paths.values()]
