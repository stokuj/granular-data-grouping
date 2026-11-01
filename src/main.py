import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from math import sqrt
import os
import csv
import time
from sklearn.metrics import accuracy_score

####
# importy własne
from . import point_generators as pg
from . import DBscan as db
from . import KNN as KNN
from .granule import is_admissible, largest_cluster_mask, distance_stats


#################################################################################################################################
### Single/complete linkage
#################################################################################################################################


def LinkageAlgorithmLoop(
    path, file_path, method, max_d, num_measurements, max_d_range, result_path
):
    # Wczytanie danych z pliku: tylko x, y (etykieta nie może wpływać na odległości)
    data = np.loadtxt(file_path)[:, :2]
    n_total = len(data)

    # Wykonanie algorytmu aglomeracyjnego (single / complete linkage)
    Z = linkage(data, method=method)
    max_d_step = (max_d_range - max_d) / (num_measurements - 1)

    with open(result_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        # Zapisz nagłówki kolumn
        csvwriter.writerow(
            [
                "max_d",
                "largest_cluster_points",
                "remaining_points",
                "num_clusters",
                "condition",
                "mean_cluster_distance",
                "std_dev_distance",
                "noise_mean_distance",
                "noise_std_dev_distance",
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
        )

        while max_d <= max_d_range:
            clusters = fcluster(Z, max_d, criterion="distance")
            num_clusters = len(np.unique(clusters))

            # Granula = największy klaster, cała reszta = szum
            granule_mask = largest_cluster_mask(clusters)
            largest_cluster_data = data[granule_mask]
            largest_cluster_points = len(largest_cluster_data)
            remaining_points = n_total - largest_cluster_points

            # Średnia i odchylenie odległości w granuli oraz w szumie
            mean_cluster_distance, std_dev_distance = distance_stats(largest_cluster_data)
            noise_mean_distance, noise_std_dev_distance = distance_stats(
                data[~granule_mask]
            )

            # Wspólny warunek dopuszczalności (ten sam co w DBSCAN)
            admissible = is_admissible(largest_cluster_points, n_total)

            p1, p2, p3, p4 = 0, 0, 0, 0
            p5, p6, p7, p8 = 0, 0, 0, 0
            p9, p10, p11, p12 = 0, 0, 0, 0
            p13, p14, p15 = 0, 0, 0

            if admissible:
                k3 = KNN.kNNAlgorithm(largest_cluster_data, k=3)
                k5 = KNN.kNNAlgorithm(largest_cluster_data, k=5)
                k7 = KNN.kNNAlgorithm(largest_cluster_data, k=7)
                k9 = KNN.kNNAlgorithm(largest_cluster_data, k=9)

                p1 = largest_cluster_points * (1 / mean_cluster_distance)
                p2 = largest_cluster_points * (1 / mean_cluster_distance**2)
                p3 = sqrt(largest_cluster_points) * (1 / mean_cluster_distance**2)

                p4 = largest_cluster_points * (1 / k3)
                p5 = largest_cluster_points * (1 / k3**2)
                p6 = sqrt(largest_cluster_points) * (1 / k3**2)

                p7 = largest_cluster_points * (1 / k5)
                p8 = largest_cluster_points * (1 / k5**2)
                p9 = sqrt(largest_cluster_points) * (1 / k5**2)

                p10 = largest_cluster_points * (1 / k7)
                p11 = largest_cluster_points * (1 / k7**2)
                p12 = sqrt(largest_cluster_points) * (1 / k7**2)

                p13 = largest_cluster_points * (1 / k9)
                p14 = largest_cluster_points * (1 / k9**2)
                p15 = sqrt(largest_cluster_points) * (1 / k9**2)

            # Zapisz dane do pliku CSV (kolumny granuli zawsze opisują granulę)
            csvwriter.writerow(
                [
                    max_d,
                    largest_cluster_points,
                    remaining_points,
                    num_clusters,
                    "yes" if admissible else "no",
                    mean_cluster_distance,
                    std_dev_distance,
                    noise_mean_distance,
                    noise_std_dev_distance,
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
            )

            # Zwiększenie wartości d_max o krok
            max_d = max_d + max_d_step


def LinkageAlgorithm(file_path, method, max_d):
    """Końcowe grupowanie linkage dla wybranego d_max.

    Zwraca (labels, accuracy): labels to 1 = granula (największy klaster),
    2 = szum (cała reszta), tak jak w points.txt.
    """
    data = np.loadtxt(file_path)
    Z = linkage(data[:, :2], method=method)
    clusters = fcluster(Z, max_d, criterion="distance")
    labels = np.where(largest_cluster_mask(clusters), 1, 2)
    accuracy = accuracy_score(data[:, 2].astype(int), labels)
    return labels, accuracy


def LinkageBest(file_path):
    """d_max maksymalizujące każde z kryteriów p1..p15 wśród dopuszczalnych kandydatów."""
    data = pd.read_csv(file_path)
    data = data[data["condition"] == "yes"]
    if data.empty:
        raise ValueError(f"Brak dopuszczalnych kandydatów na granulę w {file_path}")
    d_max_values = data["max_d"].to_numpy()
    return [d_max_values[np.argmax(data[p].to_numpy())] for p in P_IDS]


def dendrogramRange(file_path, method):
    """Najmniejsza i największa wysokość połączenia w drzewie (zakres szukania d_max)."""
    # Tylko x, y: etykieta nie może wpływać na odległości
    data = np.loadtxt(file_path)[:, :2]
    Z = linkage(data, method=method)
    # Wysokości połączeń w Z rosną: pierwsze połączenie = najmniejsza, ostatnie = największa
    return Z[0, 2], Z[-1, 2]


#################################################################################################################################
### Eksperymenty: algorytmy zapisują tylko dane, wykresy rysuje src/plots.py na żądanie
#################################################################################################################################

P_IDS = [f"p{i}" for i in range(1, 16)]
KNN_HEURISTIC_K = [3, 5, 7, 9]
LINKAGE_METHODS = ["single", "complete"]
LINKAGE_STEPS = 200

N_TOTAL = 1000
MAX_SIZE = 100
RADII = [30]

GENERATORS = {
    "circle_A": (pg.generate_points_circle, None),
    "circle_B": (pg.generate_points_mises, None),
    "circle_C": (pg.generate_points_normal, None),
    "ring": (pg.generate_points_ring, None),
    # "shape_A": (pg.generate_points_in_shape, "drawn_figure_A.txt"),
}
NOISE_PERCENTAGES = [30, 40, 50, 60, 70]
DBSCAN_GRID = dict(
    min_epsilon=2, max_epsilon=20.0, epsilon_step=0.2, min_min_samples=20, max_min_samples=150
)

# Szybki przebieg do sprawdzenia, że całość działa
QUICK_GENERATORS = ["circle_A", "ring"]
QUICK_NOISE_PERCENTAGES = [50]
QUICK_DBSCAN_GRID = dict(
    min_epsilon=2, max_epsilon=12.0, epsilon_step=1.0, min_min_samples=10, max_min_samples=40
)


def run_linkage(path, points_file, method):
    """Pętla d_max, wybór d_max dla każdego P i heurystyka d_max = d_k całego zbioru."""
    min_d, max_d = dendrogramRange(points_file, method)
    result_path = f"{path}/{method}/{method}LinkageLoopResults.csv"
    LinkageAlgorithmLoop(
        path=path,
        file_path=points_file,
        method=method,
        max_d=min_d,
        max_d_range=max_d,
        num_measurements=LINKAGE_STEPS,
        result_path=result_path,
    )

    candidates = []
    try:
        candidates += list(zip(P_IDS, LinkageBest(result_path)))
    except ValueError as e:
        print(e)
    xy = np.loadtxt(points_file)[:, :2]
    candidates += [(f"knn{k}", KNN.kNNAlgorithm(xy, k)) for k in KNN_HEURISTIC_K]

    rows, labels = [], {}
    for criterion, d_max in candidates:
        lab, acc = LinkageAlgorithm(points_file, method, d_max)
        rows.append(
            dict(method=method, criterion=criterion, d_max=d_max,
                 n_granule=int((lab == 1).sum()), accuracy=acc)
        )
        labels[f"{method}_{criterion}"] = lab
    return rows, labels


def run_dbscan(path, points_file, grid):
    """Siatka eps x min_samples i wybór parametrów dla każdego P."""
    db.DBscanAlgorithmLoop(**grid, folder=path)
    rows, labels = [], {}
    for p_id in P_IDS:
        best = db.DBscanBest(path, p_id)
        if best is None:
            print(f"Brak dopuszczalnych kandydatów DBSCAN dla {p_id} w {path}")
            continue
        epsilon, min_samples, _ = best
        lab, acc = db.DBscanFinal(points_file, epsilon, min_samples)
        rows.append(
            dict(method="dbscan", criterion=p_id, epsilon=epsilon, min_samples=min_samples,
                 n_granule=int((lab == 1).sum()), accuracy=acc)
        )
        labels[f"dbscan_{p_id}"] = lab
    return rows, labels


def run_experiments(
    results_root="results",
    generators=None,
    noise_percentages=None,
    dbscan_grid=None,
):
    generators = generators or list(GENERATORS)
    noise_percentages = noise_percentages or NOISE_PERCENTAGES
    dbscan_grid = dbscan_grid or DBSCAN_GRID
    os.makedirs(results_root, exist_ok=True)

    all_rows, time_rows = [], []
    for noise_percent in noise_percentages:
        noise_points = int((noise_percent / 100) * N_TOTAL)
        num_points = N_TOTAL - noise_points

        for shape in generators:
            generator_function, shape_name = GENERATORS[shape]
            for circle_radius in RADII:
                subpath = (
                    f"{shape}/{noise_points}"
                    if shape_name
                    else f"{shape}/{circle_radius}/{noise_points}"
                )
                path = os.path.join(results_root, subpath)
                for subfolder in [path] + [f"{path}/{m}" for m in LINKAGE_METHODS]:
                    os.makedirs(subfolder, exist_ok=True)

                points_file = f"{path}/points.txt"
                if not os.path.exists(points_file):
                    pg.generate_points(
                        generator_function, path, noise_points, circle_radius,
                        shape_name, num_points, MAX_SIZE,
                    )

                rows, labels, times = [], {}, {}
                for method in LINKAGE_METHODS:
                    start = time.time()
                    r, lab = run_linkage(path, points_file, method)
                    times[f"time_{method}"] = time.time() - start
                    rows += r
                    labels.update(lab)

                start = time.time()
                r, lab = run_dbscan(path, points_file, dbscan_grid)
                times["time_dbscan"] = time.time() - start
                rows += r
                labels.update(lab)

                # Dane tej konfiguracji: podsumowanie i wybrane podziały
                config = dict(shape=shape, radius=circle_radius, noise_points=noise_points)
                summary = pd.DataFrame([{**config, **row} for row in rows])
                summary.to_csv(f"{path}/summary.csv", index=False)

                points = np.loadtxt(points_file)
                labels_df = pd.DataFrame(
                    {"x": points[:, 0], "y": points[:, 1], "true": points[:, 2].astype(int)}
                )
                labels_df = pd.concat([labels_df, pd.DataFrame(labels)], axis=1)
                labels_df.to_csv(f"{path}/labels.csv", index=False)

                # Pliki zbiorcze nadpisywane po każdej konfiguracji (częściowy postęp zostaje)
                all_rows.append(summary)
                pd.concat(all_rows, ignore_index=True).to_csv(
                    os.path.join(results_root, "summary.csv"), index=False
                )
                time_rows.append({**config, **times})
                pd.DataFrame(time_rows).to_csv(
                    os.path.join(results_root, "wyniki_czasu_wykonania.csv"), index=False
                )
                print(f"{path}: gotowe ({', '.join(f'{k}={v:.1f}s' for k, v in times.items())})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Porównanie DBSCAN i single/complete linkage")
    parser.add_argument("--results", default="results", help="katalog wyników")
    parser.add_argument(
        "--quick", action="store_true",
        help="szybki test: circle_A i ring, szum 50%%, rzadka siatka DBSCAN",
    )
    args = parser.parse_args()

    if args.quick:
        run_experiments(args.results, QUICK_GENERATORS, QUICK_NOISE_PERCENTAGES, QUICK_DBSCAN_GRID)
    else:
        run_experiments(args.results)
