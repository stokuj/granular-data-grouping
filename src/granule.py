"""Wspólne reguły granuli dla wszystkich metod (DBSCAN, single/complete linkage)."""

import numpy as np
from scipy.spatial.distance import pdist

# Granula musi obejmować co najmniej tyle punktów (ułamek całego zbioru)
MIN_GRANULE_FRACTION = 0.1


def is_admissible(n_granule, n_total, min_fraction=MIN_GRANULE_FRACTION):
    """Czy kandydat może być granulą: ma >= min_fraction punktów i nie obejmuje wszystkich."""
    return n_granule >= min_fraction * n_total and n_granule < n_total


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
