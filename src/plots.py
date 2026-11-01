"""Wykresy na żądanie, rysowane z danych zapisanych przez src/main.py.

Algorytmy niczego nie rysują. Ten moduł czyta points.txt, labels.csv, summary.csv
oraz tabele kandydatów i rysuje tylko to, o co poprosisz.

Przykłady (z katalogu głównego repo):
    uv run python -m src.plots partition results/circle_A/30/500 single_p1
    uv run python -m src.plots dbscan-map results/circle_A/30/500 p1
    uv run python -m src.plots linkage-curve results/circle_A/30/500 complete p1
    uv run python -m src.plots dendrogram results/circle_A/30/500 single
    uv run python -m src.plots accuracy results --criterion p1 --shape circle_A
Dodaj --out plik.png (albo .pdf/.svg), żeby zapisać zamiast wyświetlać.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram

GRANULE = "#1d4fb8"
PULLED = "#c4561b"
NOISE = "#a3adb9"
SAVE_DPI = 300


def _finish(fig, out):
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=SAVE_DPI)
        print(f"Zapisano {out}")
    else:
        plt.show()
    plt.close(fig)


def partition(path, column, out=None):
    """Podział punktów wybrany przez metodę i kryterium (kolumna z labels.csv, np. single_p1)."""
    df = pd.read_csv(f"{path}/labels.csv")
    if column not in df.columns:
        options = ", ".join(c for c in df.columns if c not in ("x", "y", "true"))
        raise SystemExit(f"Brak kolumny {column!r}. Dostępne: {options}")
    granule, true_cluster = df[column] == 1, df["true"] == 1
    groups = [
        (~granule & ~true_cluster, "szum poprawnie odrzucony", dict(c=NOISE, s=6)),
        (~granule & true_cluster, "z grupy, ale poza granulą",
         dict(facecolors="none", edgecolors=GRANULE, s=14, linewidths=0.8)),
        (granule & true_cluster, "w granuli, z prawdziwej grupy", dict(c=GRANULE, s=10)),
        (granule & ~true_cluster, "szum wciągnięty do granuli", dict(c=PULLED, s=14)),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for mask, label, style in groups:
        ax.scatter(df.x[mask], df.y[mask], label=f"{label} ({int(mask.sum())})", **style)
    accuracy = (granule == true_cluster).mean()
    ax.set_title(f"{column}   N={int(granule.sum())}   trafność={accuracy:.3f}")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    _finish(fig, out)


def dbscan_map(path, p_id, out=None):
    """Kryterium P na siatce eps x min_samples (tylko dopuszczalni kandydaci)."""
    df = pd.read_csv(f"{path}/dbScan/dbScanResults.csv")
    yes = df[df["condition"] == "yes"]
    if yes.empty:
        raise SystemExit("Brak dopuszczalnych kandydatów DBSCAN.")
    best = yes.loc[yes[p_id].idxmax()]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(yes["epsilon"], yes["min_samples"], c=yes[p_id], cmap="viridis", s=14)
    ax.scatter(best["epsilon"], best["min_samples"], marker="x", c="red", s=80,
               label=f"max {p_id}: eps={best['epsilon']}, minPts={int(best['min_samples'])}")
    fig.colorbar(sc, ax=ax, label=p_id)
    ax.set_xlabel("eps")
    ax.set_ylabel("min_samples")
    ax.set_title(f"DBSCAN, {p_id}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _finish(fig, out)


def linkage_curve(path, method, p_id, out=None):
    """Kryterium P i rozmiar granuli w zależności od d_max."""
    df = pd.read_csv(f"{path}/{method}/{method}LinkageCandidates.csv")
    yes = df[df["condition"] == "yes"]
    fig, (ax_p, ax_n) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    ax_p.plot(yes["max_d"], yes[p_id], color=GRANULE, marker=".", ms=3)
    if not yes.empty:
        best = yes.loc[yes[p_id].idxmax()]
        ax_p.axvline(best["max_d"], color="black", lw=0.8)
        ax_p.set_title(f"{method} linkage: max {p_id} przy d_max={best['max_d']:.2f}")
    ax_p.set_ylabel(p_id)
    ax_p.grid(True, alpha=0.3)
    ax_n.plot(df["max_d"], df["largest_cluster_points"], color=GRANULE, label="granula (N)",
              drawstyle="steps-post")
    ax_n.plot(df["max_d"], df["remaining_points"], color=PULLED, label="szum",
              drawstyle="steps-post")
    ax_n.set_xlabel("d_max")
    ax_n.set_ylabel("liczba punktów")
    ax_n.legend(fontsize=8)
    ax_n.grid(True, alpha=0.3)
    _finish(fig, out)


def dendrogram(path, method, out=None):
    """Dendrogram liczony na x, y z points.txt."""
    data = np.loadtxt(f"{path}/points.txt")[:, :2]
    fig, ax = plt.subplots(figsize=(11, 5))
    scipy_dendrogram(linkage(data, method=method), ax=ax, no_labels=True, color_threshold=0)
    ax.set_title(f"Dendrogram, {method} linkage")
    ax.set_ylabel("wysokość połączenia")
    _finish(fig, out)


def accuracy(results_root, criterion="p1", shape=None, out=None):
    """Trafność w zależności od liczby punktów szumu, osobno dla każdej metody."""
    df = pd.read_csv(os.path.join(results_root, "summary.csv"))
    df = df[df["criterion"] == criterion]
    if shape:
        df = df[df["shape"] == shape]
    if df.empty:
        raise SystemExit("Brak wierszy dla tych filtrów.")
    shapes = sorted(df["shape"].unique())
    fig, axes = plt.subplots(1, len(shapes), figsize=(4.5 * len(shapes), 4), squeeze=False)
    for ax, sh in zip(axes[0], shapes):
        sub = df[df["shape"] == sh]
        for method, grp in sub.groupby("method"):
            grp = grp.sort_values("noise_points")
            ax.plot(grp["noise_points"], grp["accuracy"], marker="o", label=method)
        ax.set_title(f"{sh}, {criterion}")
        ax.set_xlabel("punkty szumu (z 1000)")
        ax.set_ylabel("trafność")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    _finish(fig, out)


def main():
    parser = argparse.ArgumentParser(description="Wykresy na żądanie z zapisanych wyników")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("partition", help="podział punktów dla metody i kryterium")
    p.add_argument("path")
    p.add_argument("column", help="kolumna z labels.csv, np. single_p1, dbscan_p3")
    p = sub.add_parser("dbscan-map", help="P na siatce eps x min_samples")
    p.add_argument("path")
    p.add_argument("p_id")
    p = sub.add_parser("linkage-curve", help="P i rozmiar granuli w zależności od d_max")
    p.add_argument("path")
    p.add_argument("method", choices=["single", "complete"])
    p.add_argument("p_id")
    p = sub.add_parser("dendrogram", help="dendrogram dla metody")
    p.add_argument("path")
    p.add_argument("method", choices=["single", "complete"])
    p = sub.add_parser("accuracy", help="trafność a poziom szumu")
    p.add_argument("results_root")
    p.add_argument("--criterion", default="p1")
    p.add_argument("--shape")
    for p in sub.choices.values():
        p.add_argument("--out", help="zapisz do pliku zamiast wyświetlać")

    a = parser.parse_args()
    if a.cmd == "partition":
        partition(a.path, a.column, a.out)
    elif a.cmd == "dbscan-map":
        dbscan_map(a.path, a.p_id, a.out)
    elif a.cmd == "linkage-curve":
        linkage_curve(a.path, a.method, a.p_id, a.out)
    elif a.cmd == "dendrogram":
        dendrogram(a.path, a.method, a.out)
    else:
        accuracy(a.results_root, a.criterion, a.shape, a.out)


if __name__ == "__main__":
    main()
