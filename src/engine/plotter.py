"""Matplotlib figures for ANFIS training and journey diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def plot_anfis_training_loss(
    train_loss: list[float],
    val_losses: list | None,
    *,
    time_interval: int,
    output_path: str | Path,
    val_legend_label: str = "Val loss",
) -> None:
    """Save train (and optional val) loss vs epoch for one time interval."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    if train_loss:
        ax.plot(
            range(1, len(train_loss) + 1),
            train_loss,
            label="Train loss",
        )
    if (
        val_losses
        and len(val_losses) == len(train_loss)
        and all(np.isfinite(v) for v in val_losses)
    ):
        ax.plot(
            range(1, len(val_losses) + 1),
            list(val_losses),
            label=val_legend_label,
        )
    ax.set_title(f"ANFIS loss (time interval {time_interval})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.legend()
    ax.grid(True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_journey_inter_station_distances(
    distances_km: np.ndarray,
    output_path: str | Path,
    *,
    bin_km: float = 0.5,
) -> None:
    """Histogram (fixed bin width) + ECDF of per-journey inter-station distances (km)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    d = np.asarray(distances_km, dtype=np.float64)
    n = int(d.size)
    if n == 0:
        return

    d_max = float(d.max())
    right = np.ceil(d_max / bin_km) * bin_km + bin_km
    hist_edges = np.arange(0.0, right + 1e-9, bin_km)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), layout="tight")
    axes[0].hist(
        d,
        bins=hist_edges,
        color="steelblue",
        edgecolor="white",
        alpha=0.9,
        linewidth=0.3,
    )
    axes[0].set_xlabel(
        f"Geodesic distance origin → destination station (km) ({int(bin_km * 1000)} m bins)"
    )
    axes[0].set_ylabel("Journey count")
    axes[0].set_title(
        f"Journey inter-station distances (n={n}, bin width = {int(bin_km * 1000)} m)"
    )
    axes[0].grid(True, alpha=0.3)

    s = np.sort(d)
    axes[1].plot(
        s, (np.arange(1, n + 1) - 0.5) / n, color="darkgreen", drawstyle="steps-post"
    )
    axes[1].set_xlabel("Distance (km)")
    axes[1].set_ylabel("ECDF")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
