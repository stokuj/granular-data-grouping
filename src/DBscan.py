import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Wewnętrzna funkcja sklearn, którą DBSCAN.fit wywołuje po wyszukaniu sąsiadów.
# Używamy jej bezpośrednio, żeby szukać sąsiadów raz na każde eps; wersja sklearn
# jest przypięta w uv.lock. Przy zmianie wersji sprawdź, czy DBSCAN.fit nadal
# robi to samo (radius_neighbors -> n_neighbors >= min_samples -> dbscan_inner).
from sklearn.cluster._dbscan_inner import dbscan_inner
from .granule import P_IDS, criteria, is_admissible
from sklearn.metrics import accuracy_score


DBSCAN_COLUMNS = [
    "epsilon",
    "min_samples",
    "p_cluster",
    "no_noise",
    "condition",
    "mean_distance",
    "std_distance",
    "no_clusters",
] + P_IDS


def _granule_metrics(cluster_data, admissible, n_total, mean_d_all):
    """Średnia i odchylenie odległości w klastrze oraz p1..p16 (tylko dla dopuszczalnych)."""
    p = [0.0] * len(P_IDS)
    if len(cluster_data) < 2:
        return 0.0, 0.0, p
    distances = pdist(cluster_data)
    mean_distance, std_distance = np.mean(distances), np.std(distances)
    if admissible and mean_distance != 0 and std_distance != 0:
        p = criteria(cluster_data, mean_distance, n_total, mean_d_all)
    return mean_distance, std_distance, p


def DBscanAlgorithmLoop(
    min_epsilon, max_epsilon, epsilon_step, min_min_samples, max_min_samples, folder
):
    """Siatka eps x min_samples; jeden wiersz CSV na każdą parę parametrów.

    Przyspieszenia (wyniki takie same jak DBSCAN(eps, min_samples).fit):
    - sąsiedzi w promieniu eps liczeni raz na każde eps; dla każdego min_samples
      zostaje tylko wyznaczenie punktów rdzeniowych i dbscan_inner, czyli dokładnie
      to, co robi DBSCAN.fit po wyszukaniu sąsiadów,
    - metryki granuli liczone raz na każdy unikalny klaster (wiele par parametrów
      daje dokładnie ten sam klaster),
    - p1..p16 liczone tylko dla kandydatów dopuszczalnych (tylko te są wybierane).
    """
    data = np.loadtxt(f"{folder}/points.txt")
    # Tylko x, y: etykieta nie może wpływać na odległości
    dataForDBSCAN = data[:, :2]
    n_total = len(dataForDBSCAN)
    mean_d_all = pdist(dataForDBSCAN).mean()
    os.makedirs(f"{folder}/dbScan", exist_ok=True)

    results = []
    cache = {}

    epsilon = min_epsilon
    while epsilon <= max_epsilon + epsilon_step:
        neighborhoods = (
            NearestNeighbors(radius=epsilon)
            .fit(dataForDBSCAN)
            .radius_neighbors(dataForDBSCAN, return_distance=False)
        )
        n_neighbors = np.array([len(neighbors) for neighbors in neighborhoods])
        for min_samples in range(min_min_samples, max_min_samples + 1):
            labels = np.full(n_total, -1, dtype=np.intp)
            core_samples = np.asarray(n_neighbors >= min_samples, dtype=np.uint8)
            dbscan_inner(core_samples, neighborhoods, labels)

            no_clusters = len(np.unique(labels))
            no_noise = int(np.sum(labels == -1))
            cluster_idx = np.flatnonzero(labels == 0)
            p_cluster = len(cluster_idx)

            # Dokładnie jeden klaster (etykieta 0) plus szum, oraz wspólny warunek
            # dopuszczalności granuli (ten sam co w linkage)
            admissible = no_clusters - (no_noise > 0) == 1 and p_cluster > 0 \
                and is_admissible(p_cluster, n_total)

            key = (cluster_idx.tobytes(), admissible)
            if key not in cache:
                cache[key] = _granule_metrics(dataForDBSCAN[cluster_idx], admissible, n_total, mean_d_all)
            mean_distance, std_distance, p = cache[key]

            results.append(
                [round(epsilon, 2), min_samples, p_cluster, no_noise,
                 "yes" if admissible else "no", mean_distance, std_distance, no_clusters] + p
            )

        epsilon += epsilon_step
        epsilon = round(epsilon, 2)

    results_df = pd.DataFrame(results, columns=DBSCAN_COLUMNS)
    results_df.to_csv(f"{folder}/dbScan/dbScanResults.csv", index=False)
    results_df[results_df["condition"] == "yes"].to_csv(
        f"{folder}/dbScan/dbScanResultsYes.csv", index=False
    )


def DBscanBest(folder, p_id):
    """Parametry (eps, min_samples) maksymalizujące kryterium p_id wśród dopuszczalnych.

    Zwraca (epsilon, min_samples, p_value) albo None, gdy brak dopuszczalnych kandydatów.
    """
    data = pd.read_csv(f"{folder}/dbScan/dbScanResults.csv")
    data_yes = data[data["condition"] == "yes"]
    if data_yes.empty:
        return None
    best = data_yes.loc[data_yes[p_id].idxmax()]
    return float(best["epsilon"]), int(best["min_samples"]), float(best[p_id])


def DBscanFinal(file_path, epsilon, samples):
    """Końcowe grupowanie DBSCAN dla wybranych parametrów.

    Zwraca (labels, accuracy): labels to 1 = granula, 2 = szum (jak w points.txt).
    """
    data = np.loadtxt(file_path)
    dbscan = DBSCAN(eps=epsilon, min_samples=samples).fit(data[:, :2])
    labels = np.where(dbscan.labels_ == -1, 2, 1)
    accuracy = accuracy_score(data[:, 2].astype(int), labels)
    return labels, accuracy
