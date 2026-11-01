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


# Numery jak w pierwotnym zestawie p1..p16, żeby wcześniejsze wyniki dalej się zgadzały.
# Usunięte: p2 i p3 (zawsze dolna granica rozmiaru granuli) oraz p7..p15 (d_k dla
# k = 5, 7, 9 dawało te same wybory co k = 3).
# p1: N / d̄, gdzie d̄ = średnia odległość par punktów w granuli
# p4, p5, p6: N / d3, N / d3^2, sqrt(N) / d3^2, gdzie d3 = średnia odległość do 3 sąsiadów
# p16: znormalizowane kryterium Pedrycza (pokrycie x specyficzność)
# p17: p16 z drugą specyficznością, lokalną (d3 granuli wobec d3 całego zbioru)
# p18: stabilność, czyli jak szeroki zakres parametru daje tę samą granulę; liczy ją
#      metoda (linkage: przedział d_max, DBSCAN: liczba par eps x min_samples)
CRITERIA_IDS = ["p1", "p4", "p5", "p6", "p16", "p17"]
STABILITY_ID = "p18"
P_IDS = CRITERIA_IDS + [STABILITY_ID]


def criteria(cluster_data, mean_d, n_total, mean_d_all, d3_all):
    """Wartości kryteriów z CRITERIA_IDS dla granuli, w tej samej kolejności.

    mean_d, mean_d_all: średnia odległość między parami punktów granuli / całego zbioru,
    d3_all: średnia odległość do 3 najbliższych sąsiadów w całym zbiorze.
    p16 = (N / n_total) * (1 - mean_d / mean_d_all): pokrycie i specyficzność w [0, 1],
    więc wynik nie zależy od jednostek odległości.
    p17 = p16 * (1 - d3 / d3_all): granula ma być zwarta w całości i gęsta lokalnie.
    """
    N = len(cluster_data)
    (d3,) = knn_mean_distances(cluster_data, (3,))
    p16 = (N / n_total) * max(0.0, 1 - mean_d / mean_d_all)
    return [
        N / mean_d,
        N / d3,
        N / d3**2,
        sqrt(N) / d3**2,
        p16,
        p16 * max(0.0, 1 - d3 / d3_all),
    ]


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
