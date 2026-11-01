import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
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
                k3 = KNN.kNNAlgorithm(
                    data=largest_cluster_data,
                    k=3,
                    folder=path + "/" + method + "/",
                    show_picture=False,
                    save_picture=False,
                    name=method + " " + str(max_d),
                )
                k5 = KNN.kNNAlgorithm(
                    data=largest_cluster_data,
                    k=5,
                    folder=path + "/" + method + "/",
                    show_picture=False,
                    save_picture=False,
                    name=method + " " + str(max_d),
                )
                k7 = KNN.kNNAlgorithm(
                    data=largest_cluster_data,
                    k=7,
                    folder=path + "/" + method + "/",
                    show_picture=False,
                    save_picture=False,
                    name=method + " " + str(max_d),
                )
                k9 = KNN.kNNAlgorithm(
                    data=largest_cluster_data,
                    k=9,
                    folder=path + "/" + method + "/",
                    show_picture=False,
                    save_picture=False,
                    name=method + " " + str(max_d),
                )

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


def LinkageAlgorithm(file_path, method, max_d, name, folder, show_picture=False):
    # Wczytanie danych z pliku
    data = np.loadtxt(file_path)
    # Wybranie tylko dwóch pierwszych kolumn (x i y), ignorując kolumnę z etykietami
    data_xy = data[:, :2]
    # Wykonanie algorytmu aglomeracyjnego single linkage
    Z = linkage(data_xy, method=method)

    # Wyznaczenie klastrów na podstawie dendrogramu
    clusters = fcluster(Z, max_d, criterion="distance")

    # Granula = największy klaster (etykieta 1), cała reszta = szum (etykieta 2),
    # tak samo jak w points.txt. Bez progów zależnych od prawdziwego podziału.
    clusters = np.where(largest_cluster_mask(clusters), 1, 2)

    # Obliczenie dokładności
    true_labels = data[:, 2].astype(int)

    accuracy = accuracy_score(true_labels, clusters)
    # print("Accuracy:", accuracy)
    #######################################################################
    # Wyświetlenie klastrów na płaszczyźnie 2D
    plt.figure()

    # Wyświetlenie punktów danych
    plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap="viridis", alpha=0.5, s=5)

    # Tworzenie tablicy Z dla konturów klastrów
    x = np.linspace(min(data[:, 0]), max(data[:, 0]), 100)
    y = np.linspace(min(data[:, 1]), max(data[:, 1]), 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            dists = np.sqrt((data[:, 0] - X[j, i]) ** 2 + (data[:, 1] - Y[j, i]) ** 2)
            if np.min(dists) < max_d:
                Z[j, i] = clusters[np.argmin(dists)]
            else:
                Z[j, i] = np.nan  # Ustawienie obszarów poza zasięgiem klastrów na NaN

    plt.contourf(X, Y, Z, alpha=0.1, cmap="viridis")

    # Ustawienie zakresu osi x i y
    plt.xlim(min(data[:, 0]) - 5, max(data[:, 0]) + 5)
    plt.ylim(min(data[:, 1]) - 5, max(data[:, 1]) + 5)

    plt.title(f"{method} -- {name}: {round(max_d, 2)} Accuracy: {accuracy}")
    plt.xlabel("Współrzędna X")
    plt.ylabel("Współrzędna Y")
    plt.colorbar(label="Numer klastra")

    output_file = f"{folder}/{method}/clustering_{name}.png"
    plt.savefig(output_file, dpi=1000)
    if show_picture:
        plt.show()
    plt.close()

    #######################################################################
    # Tworzenie listy punktów w formacie "x_punktu y_punktu id_clustra"
    points_with_clusters = []
    for i in range(len(data)):
        points_with_clusters.append((data[i][0], data[i][1], clusters[i]))

    # Zapisywanie punktów do pliku CSV
    output_file = f"{folder}/{method}/LinkageResults_{name}.csv"
    with open(output_file, mode="w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        # Zapisz nagłówki kolumn
        csvwriter.writerow(["x_punktu", "y_punktu", "id_klastra"])
        for point in points_with_clusters:
            csvwriter.writerow(point)

    # print(f"Punkty zapisano do pliku {output_file}")


def LinkageChart(file_path, draw):
    # Wczytanie wyników pętli; do wyboru d_max tylko dopuszczalni kandydaci
    data = pd.read_csv(file_path)
    data = data[data["condition"] == "yes"]
    if data.empty:
        raise ValueError(f"Brak dopuszczalnych kandydatów na granulę w {file_path}")

    d_max_values = data["max_d"].to_numpy()

    # d_max maksymalizujące każde z kryteriów p1..p15
    # (kolejność: p1, p2, p3, k3_p1, k3_p2, k3_p3, ..., k9_p1, k9_p2, k9_p3)
    p_columns = [f"p{i}" for i in range(1, 16)]
    best_d_max = [d_max_values[np.argmax(data[col].to_numpy())] for col in p_columns]

    if draw:
        # Liczba punktów w granuli i w szumie w zależności od d_max
        plt.figure(figsize=(15, 6))
        plt.plot(
            d_max_values,
            data["largest_cluster_points"],
            label="Liczba punktów w klastrze",
            color="blue",
        )
        plt.plot(
            d_max_values,
            data["remaining_points"],
            label="Liczba punktów w szumie",
            color="red",
        )
        plt.xlabel("d_max")
        plt.ylabel("Liczba punktów")
        plt.title("Liczba punktów w klastrze oraz szumie")
        plt.grid(True)
        plt.legend()

        # Kryterium p1 w zależności od d_max
        p1 = data["p1"].to_numpy()
        plt.figure(figsize=(15, 6))
        plt.plot(d_max_values, p1, label="p1", color="green")
        plt.scatter(
            best_d_max[0],
            p1.max(),
            color="black",
            label=f"D_max dla max ({best_d_max[0]})",
        )
        plt.xlabel("d_max")
        plt.ylabel("p1")
        plt.title("Wartość p1 dla parametru d_max")
        plt.grid(True)
        plt.legend()
        plt.show()
    return best_d_max


def makeDendrogram(file_path, method, draw):
    # Wczytanie danych z pliku: tylko x, y (etykieta nie może wpływać na odległości)
    data = np.loadtxt(file_path)[:, :2]

    # Wykonanie algorytmu aglomeracyjnego (single / complete linkage)
    Z = linkage(data, method=method)

    # Wysokości połączeń w Z rosną: pierwsze połączenie = najmniejsza, ostatnie = największa
    min_distance = Z[0, 2]
    max_distance = Z[-1, 2]

    if draw:
        plt.figure(figsize=(10, 5))
        dendrogram(Z)
        plt.title("Dendrogram - Algorytm aglomeracyjny")
        plt.xlabel("Indeksy danych")
        plt.ylabel("Odległość")
        plt.show()
    return min_distance, max_distance


#################################################################################################################################
#################################################################################################################################


def p_name(param):
    for name, value in globals().items():
        if value is param:
            return name
    return None


def run_experiments():
    maxSize = 100
    results_root = "results"
    os.makedirs(results_root, exist_ok=True)

    generators = {
        "circle_A": (pg.generate_points_circle, None),
        "circle_B": (pg.generate_points_mises, None),
        "circle_C": (pg.generate_points_normal, None),
        "ring": (pg.generate_points_ring, None),
        # "shape_A"   : (pg.generate_points_in_shape, "drawn_figure_A.txt"),
        # "shape_B"   : (pg.generate_points_in_shape, "drawn_figure_B.txt"),
        # "shape_C"   : (pg.generate_points_in_shape, "drawn_figure_C.txt"),
    }
    csv_file = os.path.join(results_root, "wyniki_czasu_wykonania.csv")

    # # Tworzenie nagłówków pliku CSV
    # with open(csv_file, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['generator', 'noisePoints', 'circle_radius', 'time_dbscan', 'time_single', 'time_complete'])

    # Pętla szumu 30% - 70%, skok 10%
    noise_percentages = [30, 40, 50, 60, 70]

    for noise_percent in noise_percentages:
        noisePoints = int((noise_percent / 100) * 1000)
        num_points = 1000 - noisePoints

        # Pętla generatora - kształu
        for folder, (generator_function, shape_name) in generators.items():
            # Pętlna rozmiaru koła jeżeli koło - aktulanie działą raz więc to NIE PĘTLA!
            for circle_radius in np.arange(30, 31, 20):
                #print(circle_radius, noisePoints, generator_function.__name__)

                ##############################################################
                ### GENEREATE POINTS
                ##############################################################
                experiment_subpath = (
                    f"{folder}/{circle_radius}/{noisePoints}"
                    if generator_function != pg.generate_points_in_shape
                    else f"{folder}/{noisePoints}"
                )
                path = os.path.join(results_root, experiment_subpath)

                for subfolder in [path, path + "/single", path + "/complete"]:
                    if not os.path.exists(subfolder):
                        os.makedirs(subfolder)

                points_file = f"{path}/points.txt"
                if not os.path.exists(points_file):
                    pg.generate_points(
                        generator_function,
                        path,
                        noisePoints,
                        circle_radius,
                        shape_name,
                        num_points,
                        maxSize,
                    )
                ##############################################################
                
                # # Wczytanie danych z pliku
                data = np.loadtxt(points_file)

                # ### KNN
                k_names = [3, 5, 7, 9]
                p_names = [
                    "p1",
                    "p2",
                    "p3",
                    "k3_p1",
                    "k3_p2",
                    "k3_p3",
                    "k5_p1",
                    "k5_p2",
                    "k5_p3",
                    "k7_p1",
                    "k7_p2",
                    "k7_p3",
                    "k9_p1",
                    "k9_p2",
                    "k9_p3",
                ]

                # ### Single i complete linkage
                # methods = ["single", "complete"]

                # for method in methods:
                #     start_time = time.time()

                #     # Tworzenie dendrogramu i uruchamianie pętli algorytmu Linkage
                #     min_distance, max_distance = makeDendrogram(
                #         file_path=f"{path}/points.txt", method=method, draw=False
                #     )
                #     result_path = f"{path}/{method}/{method}LinkageLoopResults.csv"
                #     LinkageAlgorithmLoop(
                #         path=path,
                #         file_path=f"{path}/points.txt",
                #         method=method,
                #         max_d=min_distance,
                #         max_d_range=max_distance,
                #         num_measurements=200,
                #         result_path=result_path,
                #     )
                #     p_val = LinkageChart(file_path=result_path, draw=False)

                #     # Uruchamianie algorytmu kNN i algorytmu Linkage
                #     for k in k_names:
                #         k_val = KNN.kNNAlgorithm(
                #             data,
                #             k=k,
                #             folder=path,
                #             show_picture=False,
                #             save_picture=False,
                #             name=None,
                #         )
                #         LinkageAlgorithm(
                #             file_path=f"{path}/points.txt",
                #             method=method,
                #             max_d=k_val,
                #             name=f"k{k}",
                #             folder=path,
                #             show_picture=False,
                #         )

                #     # Uruchamianie algorytmu Linkage dla każdego p w p_names
                #     for i, p in enumerate(p_names):
                #         LinkageAlgorithm(
                #             file_path=f"{path}/points.txt",
                #             method=method,
                #             max_d=p_val[i],
                #             name=p,
                #             folder=path,
                #             show_picture=False,
                #         )
                #     end_time = time.time()
                #     elapsed_time = end_time - start_time

                #     if method == "single":
                #         time_single = elapsed_time
                #     else:
                #         time_complete = elapsed_time

                # ### DBscan
                # start_time = time.time()

                # db.DBscanAlgorithmLoop(2, 20.0, 0.2, 20, 150, folder=path)

                # for i in range(1, 16):
                #     ep, samp, p_val = db.DBscanChart(
                #         folder=path, show_picture=False, p_id=f"p{i}"
                #     )
                #     db.DBscan(
                #         file_path=f"{path}/points.txt",
                #         folder=path,
                #         epsilon=ep,
                #         samples=samp,
                #         p_id=f"p{i}",
                #         p_val=p_val,
                #     )

                # end_time = time.time()
                # time_dbscan = end_time - start_time

                # ### Zapis wyników do pliku CSV
                # with open(csv_file, mode="a", newline="") as file:
                #     writer = csv.writer(file)
                #     writer.writerow(
                #         [
                #             folder,
                #             noisePoints,
                #             circle_radius,
                #             time_dbscan,
                #             time_single,
                #             time_complete,
                #         ]
                #     )

                # # ### Wykresy
                # pg.create_combined_plot(path)

                # if generator_function == pg.generate_points_in_shape:
                #     break


if __name__ == "__main__":
    run_experiments()
