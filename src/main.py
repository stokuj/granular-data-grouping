import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from math import sqrt
from scipy.spatial.distance import pdist, squareform
import os
import time
from sklearn.metrics import accuracy_score

####
# importy własne
from . import point_generators as pg
from . import DBscan as db
from . import KNN as KNN
from .granule import is_admissible, largest_cluster_mask


#################################################################################################################################
### Single/complete linkage
#################################################################################################################################


LINKAGE_COLUMNS = [
    "max_d",
    "max_d_low",
    "max_d_high",
    "largest_cluster_points",
    "remaining_points",
    "num_clusters",
    "condition",
    "mean_cluster_distance",
    "std_dev_distance",
    "noise_mean_distance",
    "noise_std_dev_distance",
] + [f"p{i}" for i in range(1, 16)]


def _mean_std(total, total_sq, n_pairs):
    """Średnia i odchylenie standardowe z sumy i sumy kwadratów odległości."""
    if n_pairs <= 0:
        return 0.0, 0.0
    mean = total / n_pairs
    return mean, sqrt(max(total_sq / n_pairs - mean**2, 0.0))


def LinkageCandidates(file_path, method, result_path):
    """Wszyscy kandydaci na granulę z drzewa linkage, każdy dokładnie raz.

    Przechodzi po połączeniach drzewa od najniższego. Kandydat powstaje za każdym
    razem, gdy rośnie największy klaster. Granula jest taka sama dla każdego cięcia
    d_max z przedziału [max_d_low, max_d_high), od tego połączenia do następnego
    kandydata. Jako max_d zapisujemy środek przedziału, żeby fcluster(Z, max_d)
    odtwarzał granulę także po zaokrągleniu przy zapisie do CSV. Sumy odległości
    w klastrach są aktualizowane przy każdym połączeniu zamiast liczyć pdist od nowa.
    """
    # Tylko x, y: etykieta nie może wpływać na odległości
    data = np.loadtxt(file_path)[:, :2]
    n_total = len(data)
    Z = linkage(data, method=method)

    D = squareform(pdist(data))
    D_sq = D**2
    row_sum, row_sum_sq = D.sum(axis=1), D_sq.sum(axis=1)
    all_pairs_sum, all_pairs_sq = D.sum() / 2, D_sq.sum() / 2

    # Dla każdego klastra: punkty, suma (i suma kwadratów) odległości wewnątrz,
    # oraz suma wierszy macierzy odległości (potrzebna do statystyk szumu)
    members = {i: [i] for i in range(n_total)}
    inner = {i: (0.0, 0.0) for i in range(n_total)}
    rows_of = {i: (row_sum[i], row_sum_sq[i]) for i in range(n_total)}

    largest_id, largest_size, changed = None, 1, False
    rows = []
    for step, (a, b, height, _) in enumerate(Z):
        a, b, new = int(a), int(b), n_total + step
        ma, mb = members.pop(a), members.pop(b)
        block = np.ix_(ma, mb)
        (sa, qa), (sb, qb) = inner.pop(a), inner.pop(b)
        inner[new] = (sa + sb + D[block].sum(), qa + qb + D_sq[block].sum())
        (ra, rqa), (rb, rqb) = rows_of.pop(a), rows_of.pop(b)
        rows_of[new] = (ra + rb, rqa + rqb)
        members[new] = ma + mb

        if len(members[new]) > largest_size or largest_id in (a, b):
            largest_id, largest_size, changed = new, len(members[new]), True

        # Połączenia o tej samej wysokości traktujemy razem: fcluster przy tym
        # d_max wykona je wszystkie
        last_at_height = step == len(Z) - 1 or Z[step + 1, 2] > height
        if not (changed and last_at_height):
            continue
        changed = False

        granule = members[largest_id]
        N = len(granule)
        remaining = n_total - N
        g_sum, g_sq = inner[largest_id]
        g_rows, g_rows_sq = rows_of[largest_id]
        # Pary szum-szum = wszystkie pary - pary w granuli - pary granula-szum
        cross, cross_sq = g_rows - 2 * g_sum, g_rows_sq - 2 * g_sq
        noise_sum = all_pairs_sum - g_sum - cross
        noise_sq = all_pairs_sq - g_sq - cross_sq

        mean_d, std_d = _mean_std(g_sum, g_sq, N * (N - 1) / 2)
        noise_mean, noise_std = _mean_std(noise_sum, noise_sq, remaining * (remaining - 1) / 2)
        admissible = is_admissible(N, n_total)

        p = [0.0] * 15
        if admissible:
            cluster_data = data[granule]
            p[0:3] = [N / mean_d, N / mean_d**2, sqrt(N) / mean_d**2]
            for j, d_k in enumerate(KNN.knn_mean_distances(cluster_data, (3, 5, 7, 9))):
                p[3 + 3 * j : 6 + 3 * j] = [N / d_k, N / d_k**2, sqrt(N) / d_k**2]

        rows.append(
            [height, N, remaining, n_total - (step + 1), "yes" if admissible else "no",
             mean_d, std_d, noise_mean, noise_std] + p
        )

    # Przedział stałej granuli: od tego połączenia do następnego kandydata
    for i, row in enumerate(rows):
        low = row[0]
        high = rows[i + 1][0] if i + 1 < len(rows) else low
        row[0:1] = [(low + high) / 2, low, high]

    pd.DataFrame(rows, columns=LINKAGE_COLUMNS).to_csv(result_path, index=False)


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


#################################################################################################################################
### Eksperymenty: algorytmy zapisują tylko dane, wykresy rysuje src/plots.py na żądanie
#################################################################################################################################

P_IDS = [f"p{i}" for i in range(1, 16)]
KNN_HEURISTIC_K = [3, 5, 7, 9]
LINKAGE_METHODS = ["single", "complete"]

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
    """Kandydaci z drzewa, wybór d_max dla każdego P i heurystyka d_max = d_k całego zbioru."""
    result_path = f"{path}/{method}/{method}LinkageCandidates.csv"
    LinkageCandidates(file_path=points_file, method=method, result_path=result_path)

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


def run_config(results_root, noise_percent, shape, circle_radius, dbscan_grid):
    """Jedna konfiguracja (kształt, promień, szum): dane, linkage, DBSCAN, pliki wyników."""
    noise_points = int((noise_percent / 100) * N_TOTAL)
    num_points = N_TOTAL - noise_points
    generator_function, shape_name = GENERATORS[shape]
    subpath = (
        f"{shape}/{noise_points}" if shape_name else f"{shape}/{circle_radius}/{noise_points}"
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

    print(f"{path}: gotowe ({', '.join(f'{k}={v:.1f}s' for k, v in times.items())})", flush=True)
    return summary, {**config, **times}


def _init_worker():
    # Każdy proces liczy jedną konfigurację na jednym rdzeniu: bez tego numpy/scipy
    # w 16 procesach odpaliłyby po kilka wątków BLAS i walczyły o rdzenie.
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)
    # Procesy z fork dziedziczą stan generatora losowego rodzica, więc nowe punkty
    # byłyby takie same w każdej konfiguracji. Świeże ziarno z systemu w każdym procesie.
    np.random.seed()


def run_experiments(
    results_root="results",
    generators=None,
    noise_percentages=None,
    dbscan_grid=None,
    workers=None,
):
    """Wszystkie konfiguracje; workers > 1 liczy je równolegle w osobnych procesach."""
    generators = generators or list(GENERATORS)
    noise_percentages = noise_percentages or NOISE_PERCENTAGES
    dbscan_grid = dbscan_grid or DBSCAN_GRID
    workers = workers or os.cpu_count()
    os.makedirs(results_root, exist_ok=True)

    configs = [
        (noise_percent, shape, circle_radius)
        for noise_percent in noise_percentages
        for shape in generators
        for circle_radius in RADII
    ]
    results = {}

    def save_progress():
        # Pliki zbiorcze w stałej kolejności konfiguracji, nadpisywane po każdej
        # skończonej konfiguracji (częściowy postęp zostaje)
        done = [results[c] for c in configs if c in results]
        pd.concat([s for s, _ in done], ignore_index=True).to_csv(
            os.path.join(results_root, "summary.csv"), index=False
        )
        pd.DataFrame([t for _, t in done]).to_csv(
            os.path.join(results_root, "wyniki_czasu_wykonania.csv"), index=False
        )

    start = time.time()
    if workers == 1:
        for c in configs:
            results[c] = run_config(results_root, *c, dbscan_grid)
            save_progress()
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=min(workers, len(configs)),
                                 initializer=_init_worker) as pool:
            futures = {pool.submit(run_config, results_root, *c, dbscan_grid): c for c in configs}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                save_progress()
    print(f"Wszystko: {len(configs)} konfiguracji w {time.time() - start:.1f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Porównanie DBSCAN i single/complete linkage")
    parser.add_argument("--results", default="results", help="katalog wyników")
    parser.add_argument(
        "--quick", action="store_true",
        help="szybki test: circle_A i ring, szum 50%%, rzadka siatka DBSCAN",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="liczba procesów (domyślnie liczba rdzeni; 1 = po kolei, bez procesów)",
    )
    args = parser.parse_args()

    if args.quick:
        run_experiments(args.results, QUICK_GENERATORS, QUICK_NOISE_PERCENTAGES,
                        QUICK_DBSCAN_GRID, args.workers)
    else:
        run_experiments(args.results, workers=args.workers)
