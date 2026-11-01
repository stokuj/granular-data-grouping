import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
from math import sqrt
from . import KNN as KNN
from .granule import is_admissible
from sklearn.metrics import accuracy_score


def DBscanAlgorithmLoop(
    min_epsilon, max_epsilon, epsilon_step, min_min_samples, max_min_samples, folder
):
    # Wczytanie danych
    filename = f"{folder}/points.txt"
    data = np.loadtxt(filename)

    # Wybranie tylko dwóch pierwszych kolumn (x i y), ignorując kolumnę z etykietami
    dataForDBSCAN = data[:, :2]

    if not os.path.exists(f"{folder}/dbScan"):
        os.makedirs(f"{folder}/dbScan")

    # Przygotowanie list na wyniki
    results = []
    results_yes = []

    # Pętla po różnych wartościach epsilon i min_samples
    epsilon = min_epsilon

    while epsilon <= max_epsilon + epsilon_step:
        for min_samples in range(min_min_samples, max_min_samples + 1):
            db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(dataForDBSCAN)
            labels = db.labels_

            # Obliczenie metryk
            no_clusters = len(np.unique(labels))
            no_noise = np.sum(labels == -1)
            p_cluster = np.sum(labels == 0)
            mean_distance = 0
            std_distance = 0

            # Obliczenie odległości między punktami
            if p_cluster > 0:
                distances = pdist(
                    dataForDBSCAN[labels == 0]
                )  # Oblicz odległości tylko dla punktów w klastrze
                mean_distance = np.mean(distances)
                std_distance = np.std(distances)

            # Zaokrąglenie wartości do 2 miejsc po przecinku
            epsilon_rounded = round(epsilon, 2)

            p1 = p2 = p3 = p4 = p5 = p6 = p7 = p8 = p9 = p10 = p11 = p12 = p13 = p14 = (
                p15
            ) = 0
            if mean_distance != 0 and std_distance != 0:
                cluster_data = dataForDBSCAN[labels == 0]

                k3 = KNN.kNNAlgorithm(cluster_data, k=3)
                k5 = KNN.kNNAlgorithm(cluster_data, k=5)
                k7 = KNN.kNNAlgorithm(cluster_data, k=7)
                k9 = KNN.kNNAlgorithm(cluster_data, k=9)

                p1 = p_cluster * (1 / mean_distance)
                p2 = p_cluster * (1 / mean_distance**2)
                p3 = sqrt(p_cluster) * (1 / mean_distance**2)

                p4 = p_cluster * (1 / k3)
                p5 = p_cluster * (1 / k3**2)
                p6 = sqrt(p_cluster) * (1 / k3**2)

                p7 = p_cluster * (1 / k5)
                p8 = p_cluster * (1 / k5**2)
                p9 = sqrt(p_cluster) * (1 / k5**2)

                p10 = p_cluster * (1 / k7)
                p11 = p_cluster * (1 / k7**2)
                p12 = sqrt(p_cluster) * (1 / k7**2)

                p13 = p_cluster * (1 / k9)
                p14 = p_cluster * (1 / k9**2)
                p15 = sqrt(p_cluster) * (1 / k9**2)

            result = [
                epsilon_rounded,
                min_samples,
                p_cluster,
                no_noise,
                mean_distance,
                std_distance,
                no_clusters,
                p1,
                p2,
                p3,
                p4,
                p5,
                p6,
                p7,
                p8,
                p9,
                p10,
                p11,
                p12,
                p13,
                p14,
                p15,
            ]
            # Dokładnie jeden klaster (etykieta 0) plus szum, oraz wspólny warunek
            # dopuszczalności granuli (ten sam co w linkage)
            found_clusters = set(np.unique(labels)) - {-1}
            if found_clusters == {0} and is_admissible(p_cluster, len(dataForDBSCAN)):
                result.insert(4, "yes")
                results_yes.append(result)
            else:
                result.insert(4, "no")
            results.append(result)

        epsilon += epsilon_step
        epsilon = round(epsilon, 2)

    # Zapisz wyniki do plików CSV
    columns = [
        "epsilon",
        "min_samples",
        "p_cluster",
        "no_noise",
        "condition",
        "mean_distance",
        "std_distance",
        "no_clusters",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
        "p8",
        "p9",
        "p10",
        "p11",
        "p12",
        "p13",
        "p14",
        "p15",
    ]
    results_df = pd.DataFrame(results, columns=columns)
    results_yes_df = pd.DataFrame(results_yes, columns=columns)

    results_df.to_csv(f"{folder}/dbScan/dbScanResults.csv", index=False)
    results_yes_df.to_csv(f"{folder}/dbScan/dbScanResultsYes.csv", index=False)


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
