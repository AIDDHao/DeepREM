from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from deep_cluster_model import Encoder


def load_feature_table(path: Path) -> tuple[pd.DataFrame, torch.FloatTensor]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = [[name] + list(values.values()) for name, values in data.items()]
    df = pd.DataFrame(rows)
    matrix = df.iloc[:, 1:].values
    x_scaled = StandardScaler().fit_transform(matrix)
    return df, torch.FloatTensor(x_scaled)


def strip_pair_prefix(name: str) -> str:
    return name.split("_", 1)[1] if "_" in name else name


def encode_features(encoder: Encoder, data: torch.FloatTensor) -> np.ndarray:
    encoder.eval()
    with torch.no_grad():
        encoded, _ = encoder(data)
    return encoded.cpu().numpy()


def load_state_dict(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def rank_neighbors(
    encoded_features: np.ndarray,
    names: list[str],
    target_index: int,
    top_n: int,
) -> list[dict]:
    n_neighbors = min(len(names), top_n + 1)
    nn_model = NearestNeighbors(n_neighbors=n_neighbors)
    nn_model.fit(encoded_features)
    distances, indices = nn_model.kneighbors([encoded_features[target_index]])

    rows = []
    rank = 1
    for idx, distance in zip(indices[0], distances[0]):
        if idx == target_index:
            continue
        rows.append(
            {
                "rank": rank,
                "raw_name": names[idx],
                "ligand": strip_pair_prefix(names[idx]),
                "distance": float(distance),
            }
        )
        rank += 1
    return rows


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Load the deep-clustering encoder and rank nearest endogenous ligands."
    )
    parser.add_argument("--data", default=project_root / "data" / "Test.json", type=Path)
    parser.add_argument(
        "--encoder",
        default=project_root / "models" / "encoder_params_final.pth",
        type=Path,
        help="Path to the trained encoder parameters.",
    )
    parser.add_argument("--target-index", default=0, type=int)
    parser.add_argument("--target-name", help="Optional row name in the input data.")
    parser.add_argument("--top-n", default=20, type=int)
    parser.add_argument("--output", default=project_root / "results" / "top_neighbors.csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, tensor = load_feature_table(args.data)
    names = df.iloc[:, 0].astype(str).tolist()

    target_index = args.target_index
    if args.target_name:
        if args.target_name not in names:
            raise SystemExit(f"Target name not found: {args.target_name}")
        target_index = names.index(args.target_name)

    encoder = Encoder(n_features=tensor.shape[1], latent_dim=10, n_classes=10)
    encoder.load_state_dict(load_state_dict(args.encoder))

    encoded = encode_features(encoder, tensor)
    rows = rank_neighbors(encoded, names, target_index=target_index, top_n=args.top_n)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "raw_name", "ligand", "distance"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Target: {names[target_index]}")
    print(f"Encoder: {args.encoder}")
    print(f"Saved: {args.output}")
    for row in rows:
        print(f"{row['rank']:>2}. {row['ligand']}\t{row['distance']:.8f}")


if __name__ == "__main__":
    main()
