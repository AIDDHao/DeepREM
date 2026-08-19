from __future__ import annotations

import argparse
import csv
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from deep_cluster_model import Decoder, Encoder


PAIR_INDICES = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (12, 13),
    (14, 15),
    (16, 17),
    (18, 19),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_feature_tensor(path: Path) -> tuple[pd.DataFrame, torch.FloatTensor]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = [[name] + list(values.values()) for name, values in data.items()]
    df = pd.DataFrame(rows)
    matrix = df.iloc[:, 1:].values
    x_scaled = StandardScaler().fit_transform(matrix)
    return df, torch.FloatTensor(x_scaled)


def custom_distance_loss(encoded: torch.Tensor, pair_indices=PAIR_INDICES) -> torch.Tensor:
    losses = [torch.norm(encoded[a] - encoded[b], p=2) for a, b in pair_indices]
    return torch.stack(losses).sum()


def validate_model(
    encoder: Encoder,
    decoder: Decoder,
    valid_data: torch.Tensor,
    criterion_reconstruction: nn.Module,
    labels: torch.Tensor,
    kmeans_layer: KMeans,
    clustering_loss_weight: float,
    supervised_loss_weight: float = 1.0,
) -> float:
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        encoded_valid, encoded_classification_valid = encoder(valid_data)
        outputs_valid = decoder(encoded_valid)
        reconstruction_loss_valid = criterion_reconstruction(outputs_valid, valid_data)

        distances_to_centers = torch.cdist(
            encoded_valid, torch.tensor(kmeans_layer.cluster_centers_, dtype=encoded_valid.dtype)
        )
        nearest_center_distances, _ = torch.min(distances_to_centers, dim=1)
        clustering_loss_valid = torch.mean(nearest_center_distances)

        supervised_count = min(len(labels), len(encoded_classification_valid))
        supervised_loss_valid = F.cross_entropy(
            encoded_classification_valid[:supervised_count],
            labels[:supervised_count],
        )

        total_loss_valid = (
            reconstruction_loss_valid
            + clustering_loss_weight * clustering_loss_valid
            + supervised_loss_valid * supervised_loss_weight
        )
    encoder.train()
    decoder.train()
    return float(total_loss_valid.item())


def train(args: argparse.Namespace) -> None:
    warnings.filterwarnings(
        "ignore",
        message="KMeans is known to have a memory leak on Windows with MKL.*",
    )
    set_seed(args.seed)

    train_df, train_data = load_feature_tensor(args.train_json)
    valid_df, valid_data = load_feature_tensor(args.valid_json)
    n_features = train_data.shape[1]

    encoder = Encoder(n_features=n_features, latent_dim=args.latent_dim, n_classes=args.n_clusters)
    decoder = Decoder(n_features=n_features, latent_dim=args.latent_dim)
    kmeans_layer = KMeans(n_clusters=args.n_clusters, n_init=10, random_state=args.seed)

    criterion_reconstruction = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate)
    labels = torch.arange(args.n_clusters).repeat_interleave(2)

    history = []
    encoder.train()
    decoder.train()

    for epoch in range(1, args.epochs + 1):
        encoded, _ = encoder(train_data)
        outputs = decoder(encoded)
        reconstruction_loss = criterion_reconstruction(outputs, train_data)

        kmeans_layer.fit(encoded.detach().numpy())
        distances_to_centers = torch.cdist(
            encoded, torch.tensor(kmeans_layer.cluster_centers_, dtype=encoded.dtype)
        )
        nearest_center_distances, _ = torch.min(distances_to_centers, dim=1)
        clustering_loss = torch.mean(nearest_center_distances)

        labeled_count = min(len(labels), len(train_data))
        _, encoded_classification = encoder(train_data[:labeled_count])
        supervised_loss = F.cross_entropy(encoded_classification, labels[:labeled_count])
        distance_loss = custom_distance_loss(encoded, PAIR_INDICES[: args.n_clusters])

        total_loss = (
            args.reconstruction_loss_weight * reconstruction_loss
            + args.clustering_loss_weight * clustering_loss
            + supervised_loss
            + args.custom_loss_weight * distance_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        valid_loss = validate_model(
            encoder,
            decoder,
            valid_data,
            criterion_reconstruction,
            labels,
            kmeans_layer,
            args.clustering_loss_weight,
        )

        row = {
            "epoch": epoch,
            "reconstruction_loss": float(reconstruction_loss.item()),
            "clustering_loss": float(clustering_loss.item()),
            "supervised_loss": float(supervised_loss.item()),
            "custom_distance_loss": float(distance_loss.item()),
            "total_loss": float(total_loss.item()),
            "valid_loss": valid_loss,
        }
        history.append(row)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch}/{args.epochs} "
                f"loss={row['total_loss']:.4f} "
                f"recon={row['reconstruction_loss']:.4f} "
                f"cluster={row['clustering_loss']:.4f} "
                f"supervised={row['supervised_loss']:.4f} "
                f"custom={row['custom_distance_loss']:.4f} "
                f"valid_loss={valid_loss:.4f}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), args.output_dir / args.encoder_name)
    torch.save(decoder.state_dict(), args.output_dir / args.decoder_name)

    history_path = args.output_dir / "training_history.csv"
    with history_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    metadata = {
        "train_json": str(args.train_json),
        "valid_json": str(args.valid_json),
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "n_features": int(n_features),
        "latent_dim": args.latent_dim,
        "n_clusters": args.n_clusters,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "reconstruction_loss_weight": args.reconstruction_loss_weight,
        "clustering_loss_weight": args.clustering_loss_weight,
        "custom_loss_weight": args.custom_loss_weight,
        "encoder": args.encoder_name,
        "decoder": args.decoder_name,
        "history": history_path.name,
        "note": "Full Train.json is used for autoencoder reconstruction and KMeans clustering; the first 2*n_clusters rows are used for supervised pair guidance.",
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved encoder/decoder and training history to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train the deep clustering autoencoder.")
    parser.add_argument("--train-json", default=project_root / "data" / "Train.json", type=Path)
    parser.add_argument("--valid-json", default=project_root / "data" / "Valid.json", type=Path)
    parser.add_argument("--output-dir", default=project_root / "models" / "retrained", type=Path)
    parser.add_argument("--encoder-name", default="encoder_params_retrained.pth")
    parser.add_argument("--decoder-name", default="decoder_params_retrained.pth")
    parser.add_argument("--epochs", default=390, type=int)
    parser.add_argument("--learning-rate", default=0.0001, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--n-clusters", default=10, type=int)
    parser.add_argument("--latent-dim", default=10, type=int)
    parser.add_argument("--reconstruction-loss-weight", default=1.0, type=float)
    parser.add_argument("--clustering-loss-weight", default=1.0, type=float)
    parser.add_argument("--custom-loss-weight", default=10.0, type=float)
    parser.add_argument("--log-every", default=10, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
