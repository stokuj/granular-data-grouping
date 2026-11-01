"""Wspólne reguły granuli dla wszystkich metod (DBSCAN, single/complete linkage)."""

from math import sqrt

import numpy as np
from scipy.spatial.distance import pdist

from .KNN import knn_mean_distances

# Granula musi obejmować co najmniej tyle punktów (ułamek całego zbioru)
MIN_GRANULE_FRACTION = 0.1


def is_admissible(n_granule, n_total, min_fraction=MIN_GRANULE_FRACTION):
    """Czy kandydat może być granulą: ma >= min_fraction punktów i nie obejmuje wszystkich."""
    return n_granule >= min_fraction * n_total and n_granule < n_total


# p1..p15: N lub sqrt(N) razy 1/d lub 1/d^2, d = średnia odległość w granuli albo d_k
# (k = 3, 5, 7, 9); p16: znormalizowane kryterium Pedrycza (pokrycie x specyficzność)
P_IDS = [f"p{i}" for i in range(1, 17)]


def criteria(cluster_data, mean_d, n_total, mean_d_all):
    """Wartości p1..p16 dla granuli.

    mean_d: średnia odległość między parami punktów granuli,
    mean_d_all: to samo dla całego zbioru (granula + szum).
    p16 = (N / n_total) * max(0, 1 - mean_d / mean_d_all): pokrycie i specyficzność
    w przedziale [0, 1], więc wynik nie zależy od jednostek odległości.
    """
    N = len(cluster_data)
    p = [N / mean_d, N / mean_d**2, sqrt(N) / mean_d**2]
    for d_k in knn_mean_distances(cluster_data, (3, 5, 7, 9)):
        p += [N / d_k, N / d_k**2, sqrt(N) / d_k**2]
    p.append((N / n_total) * max(0.0, 1 - mean_d / mean_d_all))
    return p


def largest_cluster_mask(labels):
    """Maska punktów należących do największego klastra (granula); reszta to szum."""
    labels = np.asarray(labels)
    ids, counts = np.unique(labels, return_counts=True)
    return labels == ids[np.argmax(counts)]


def distance_stats(points):
    """Średnia i odchylenie standardowe odległości między wszystkimi parami punktów."""
    if len(points) < 2:
        return 0.0, 0.0
    distances = pdist(points)
    return float(np.mean(distances)), float(np.std(distances))
