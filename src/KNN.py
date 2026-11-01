import numpy as np
from sklearn.neighbors import NearestNeighbors


def kNNAlgorithm(data, k):
    """Średnia (po punktach) średniej odległości do k najbliższych sąsiadów, czyli d_k."""
    knn_model = NearestNeighbors(n_neighbors=k + 1)
    knn_model.fit(data)

    # Pierwszy sąsiad to sam punkt (odległość 0), więc pomijamy kolumnę 0
    distances, _ = knn_model.kneighbors(data)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    return np.mean(mean_distances)


def knn_mean_distances(data, ks=(3, 5, 7, 9)):
    """d_k dla kilku k z jednego zapytania o max(ks) sąsiadów (wynik jak kNNAlgorithm dla każdego k)."""
    k_max = max(ks)
    distances, _ = NearestNeighbors(n_neighbors=k_max + 1).fit(data).kneighbors(data)
    return [np.mean(distances[:, 1 : k + 1]) for k in ks]
