# point_generators.py

import numpy as np
from collections.abc import Callable
from typing import TypeAlias, cast
from matplotlib.path import Path
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
PointSeries: TypeAlias = FloatArray | tuple[float, ...]
CirclePoints: TypeAlias = tuple[FloatArray, FloatArray, FloatArray, FloatArray]
ShapePoints: TypeAlias = tuple[
    tuple[float, ...], tuple[float, ...], FloatArray, FloatArray
]
CircleGenerator: TypeAlias = Callable[[int, float, int, float], CirclePoints]
ShapeGenerator: TypeAlias = Callable[[int, str, int, float], ShapePoints]
GeneratorFunction: TypeAlias = CircleGenerator | ShapeGenerator


def generate_points_circle(
    noisePoints: int, circle_radius: float, num_points: int, maxSize: float
) -> CirclePoints:
    """
    Generate points inside a circle and uniform noise points.

    Parameters
    ----------
    noisePoints : int
        Number of uniformly random noise points.
    circle_radius : float
        Radius of the generated circle.
    num_points : int
        Number of points generated for the circle distribution.
    maxSize : float
        Half-range of the square domain ``[-maxSize, maxSize]``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(x_circle, y_circle, x_random, y_random)`` arrays.
    """
    offsetX = np.random.uniform(-(maxSize - circle_radius), maxSize - circle_radius)
    offsetY = np.random.uniform(-(maxSize - circle_radius), maxSize - circle_radius)

    theta = np.random.uniform(0.0, 2.0 * np.pi, num_points)
    zero_to_one = np.random.uniform(0.0, 1.0, num_points)

    x_circle = circle_radius * np.sqrt(zero_to_one) * np.cos(theta) + offsetX
    y_circle = circle_radius * np.sqrt(zero_to_one) * np.sin(theta) + offsetY

    # Maskowanie punktów znajdujących się poza obszarem
    inside_circle_mask = (
        (x_circle >= -maxSize)
        & (x_circle <= maxSize)
        & (y_circle >= -maxSize)
        & (y_circle <= maxSize)
    )
    x_circle = x_circle[inside_circle_mask]
    y_circle = y_circle[inside_circle_mask]

    x_random = np.random.uniform(-maxSize, maxSize, noisePoints)
    y_random = np.random.uniform(-maxSize, maxSize, noisePoints)

    print(
        f"Liczba wszystkich punktów: {noisePoints + num_points}, Procentowy udział szumu: {(noisePoints / (noisePoints + num_points)) * 100}%, Rozmiar koła: {circle_radius}"
    )
    return x_circle, y_circle, x_random, y_random


def generate_points_mises(
    noisePoints: int, circle_radius: float, num_points: int, maxSize: float
) -> CirclePoints:
    """
    Generate points using a Von Mises-based radial distribution and noise points.

    Parameters
    ----------
    noisePoints : int
        Number of uniformly random noise points.
    circle_radius : float
        Radius scaling factor for generated points.
    num_points : int
        Number of points generated for the main distribution.
    maxSize : float
        Half-range of the square domain ``[-maxSize, maxSize]``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(x_circle, y_circle, x_random, y_random)`` arrays.
    """
    offsetX = np.random.uniform(-(maxSize - circle_radius), maxSize - circle_radius)
    offsetY = np.random.uniform(-(maxSize - circle_radius), maxSize - circle_radius)

    # Generowanie punktów na okręgu z rozkładem Von Misesa
    theta = np.random.uniform(0.0, 2.0 * np.pi, num_points)
    r = np.random.vonmises(0, 10, num_points) * circle_radius
    x_circle = r * np.cos(theta) + offsetX
    y_circle = r * np.sin(theta) + offsetY

    # Maskowanie punktów znajdujących się poza obszarem
    inside_circle_mask = (
        (x_circle >= -maxSize)
        & (x_circle <= maxSize)
        & (y_circle >= -maxSize)
        & (y_circle <= maxSize)
    )
    x_circle = x_circle[inside_circle_mask]
    y_circle = y_circle[inside_circle_mask]

    x_random = np.random.uniform(-maxSize, maxSize, noisePoints)
    y_random = np.random.uniform(-maxSize, maxSize, noisePoints)

    print(
        f"Liczba wszystkich punktów: {noisePoints + num_points}, Procentowy udział szumu: {(noisePoints / (noisePoints + num_points)) * 100}%, Rozmiar koła: {circle_radius}"
    )
    return x_circle, y_circle, x_random, y_random


def generate_points_normal(
    noisePoints: int, circle_radius: float, num_points: int, maxSize: float
) -> CirclePoints:
    """
    Generate points using a normal radial distribution and noise points.

    Parameters
    ----------
    noisePoints : int
        Number of uniformly random noise points.
    circle_radius : float
        Standard deviation used for radial sampling.
    num_points : int
        Number of points generated for the main distribution.
    maxSize : float
        Half-range of the square domain ``[-maxSize, maxSize]``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(x_circle, y_circle, x_random, y_random)`` arrays.
    """
    offsetX = np.random.uniform(-(maxSize - circle_radius), maxSize - circle_radius)
    offsetY = np.random.uniform(-(maxSize - circle_radius), maxSize - circle_radius)

    # Generowanie punktów na okręgu z rozkładem normalnym
    theta = np.random.uniform(0.0, 2.0 * np.pi, num_points)
    r = np.random.normal(0, circle_radius, num_points)
    x_circle = r * np.cos(theta) + offsetX
    y_circle = r * np.sin(theta) + offsetY

    # Maskowanie punktów znajdujących się poza obszarem
    inside_circle_mask = (
        (x_circle >= -maxSize)
        & (x_circle <= maxSize)
        & (y_circle >= -maxSize)
        & (y_circle <= maxSize)
    )
    x_circle = x_circle[inside_circle_mask]
    y_circle = y_circle[inside_circle_mask]

    x_random = np.random.uniform(-maxSize, maxSize, noisePoints)
    y_random = np.random.uniform(-maxSize, maxSize, noisePoints)

    print(
        f"Liczba wszystkich punktów: {noisePoints + num_points}, Procentowy udział szumu: {(noisePoints / (noisePoints + num_points)) * 100}%, Rozmiar koła: {circle_radius}"
    )
    return x_circle, y_circle, x_random, y_random


def generate_points_ring(
    noisePoints: int, circle_radius: float, num_points: int, maxSize: float
) -> CirclePoints:
    """
    Generate points in a ring-like region and uniform noise points.

    Parameters
    ----------
    noisePoints : int
        Number of uniformly random noise points.
    circle_radius : float
        Outer ring radius parameter.
    num_points : int
        Number of points generated for the ring distribution.
    maxSize : float
        Half-range of the square domain ``[-maxSize, maxSize]``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(x_circle, y_circle, x_random, y_random)`` arrays.
    """
    offsetX = np.random.uniform(-(maxSize - circle_radius), maxSize - circle_radius)
    offsetY = np.random.uniform(-(maxSize - circle_radius), maxSize - circle_radius)

    theta = np.random.uniform(0, 2.0 * np.pi, num_points)
    zero_to_one = np.random.uniform(circle_radius / 2, circle_radius, num_points)

    x_circle = 1 * (zero_to_one) * np.cos(theta) + offsetX
    y_circle = 1 * (zero_to_one) * np.sin(theta) + offsetY

    # Maskowanie punktów znajdujących się poza obszarem
    inside_circle_mask = (
        (x_circle >= -maxSize)
        & (x_circle <= maxSize)
        & (y_circle >= -maxSize)
        & (y_circle <= maxSize)
    )
    x_circle = x_circle[inside_circle_mask]
    y_circle = y_circle[inside_circle_mask]

    x_random = np.random.uniform(-maxSize, maxSize, noisePoints)
    y_random = np.random.uniform(-maxSize, maxSize, noisePoints)

    print(
        f"Liczba wszystkich punktów: {noisePoints + num_points}, Procentowy udział szumu: {(noisePoints / (noisePoints + num_points)) * 100}%, Rozmiar koła: {circle_radius}"
    )
    return x_circle, y_circle, x_random, y_random


def generate_points_in_shape(
    noisePoints: int, shape_name: str, num_points: int, maxSize: float
) -> ShapePoints:
    """
    Generate points inside a custom shape and uniform noise points.

    Parameters
    ----------
    noisePoints : int
        Number of uniformly random noise points.
    shape_name : str
        Path to a text file describing shape vertices.
    num_points : int
        Number of points to sample inside the shape.
    maxSize : float
        Half-range of the square domain ``[-maxSize, maxSize]``.

    Returns
    -------
    tuple[tuple[float, ...], tuple[float, ...], np.ndarray, np.ndarray]
        ``(x_points, y_points, x_random, y_random)`` where shape points are tuples.
    """
    x_random = np.random.uniform(-maxSize, maxSize, noisePoints)
    y_random = np.random.uniform(-maxSize, maxSize, noisePoints)

    shape = Path(read_shape_from_file(shape_name))
    bounds = [(-maxSize, maxSize), (-maxSize, maxSize)]
    points_inside: list[tuple[float, float]] = []
    while len(points_inside) < num_points:
        x = np.random.uniform(*bounds[0])
        y = np.random.uniform(*bounds[1])
        if shape.contains_point((x, y)):
            points_inside.append((x, y))

    print(
        f"Liczba wszystkich punktów: {noisePoints + num_points}, Procentowy udział szumu: {(noisePoints / (noisePoints + num_points)) * 100}%, Kształt: {shape_name}"
    )
    x_points, y_points = zip(*points_inside)
    return x_points, y_points, x_random, y_random

def save_points_to_file(
    x_circle: PointSeries,
    y_circle: PointSeries,
    x_random: PointSeries,
    y_random: PointSeries,
    folder: str,
) -> None:
    """
    Save generated points with labels to ``points.txt``.

    Parameters
    ----------
    x_circle : np.ndarray | tuple
        X coordinates of main distribution points.
    y_circle : np.ndarray | tuple
        Y coordinates of main distribution points.
    x_random : np.ndarray | tuple
        X coordinates of noise points.
    y_random : np.ndarray | tuple
        Y coordinates of noise points.
    folder : str
        Output directory where ``points.txt`` is saved.
    """
    # Convert tuples to NumPy arrays if not already arrays
    x_circle = np.array(x_circle) if isinstance(x_circle, tuple) else x_circle
    y_circle = np.array(y_circle) if isinstance(y_circle, tuple) else y_circle
    x_random = np.array(x_random) if isinstance(x_random, tuple) else x_random
    y_random = np.array(y_random) if isinstance(y_random, tuple) else y_random

    # Tworzenie etykiet dla danych: 1 dla danych z circle, 2 dla danych random
    labels_circle = np.ones(
        x_circle.shape[0]
    )  # Wszystkie punkty z circle mają etykietę 1
    labels_random = np.full(
        x_random.shape[0], 2
    )  # Wszystkie punkty z random mają etykietę 2

    # Łączenie danych x, y oraz etykiet
    data_circle = np.column_stack((x_circle, y_circle, labels_circle))
    data_random = np.column_stack((x_random, y_random, labels_random))

    # Łączenie wszystkich danych w jedną macierz
    data = np.vstack((data_circle, data_random))

    # Określenie ścieżki do pliku
    file_path = f"{folder}/points.txt"

    # Zapisywanie danych do pliku z użyciem formatu z 4 miejscami po przecinku dla wartości x i y
    np.savetxt(file_path, data, fmt="%.4f %.4f %d")


def read_shape_from_file(file_path: str) -> list[tuple[float, float]]:
    """
    Read shape vertices from a text file.

    Parameters
    ----------
    file_path : str
        Path to a file where each line has ``x,y`` coordinates.

    Returns
    -------
    list[tuple[float, float]]
        List of 2D vertices parsed from the file.
    """
    points: list[tuple[float, float]] = []
    with open(file_path, "r") as file:
        for line in file:
            # Rozdziel linie na części używając przecinka jako separatora
            x, y = line.strip().split(",")
            # Dodaj punkt do listy, przekonwertujając x i y na liczby zmiennoprzecinkowe
            points.append((float(x), float(y)))
    return points


def generate_points(
    generator_function: GeneratorFunction,
    folder: str,
    noisePoints: int,
    circle_radius: float,
    shape_name: str | None,
    num_points: int,
    maxSize: float,
) -> None:
    """
    Generate point sets and save them to disk.

    Parameters
    ----------
    generator_function : callable
        Point generator to call.
    folder : str
        Output directory for generated files.
    noisePoints : int
        Number of uniformly random noise points.
    circle_radius : float
        Radius argument for circle-based generators.
    shape_name : str | None
        Shape file path for shape-based generation.
    num_points : int
        Number of main distribution points.
    maxSize : float
        Half-range of the square domain ``[-maxSize, maxSize]``.
    """
    if shape_name:
        shape_generator = cast(ShapeGenerator, generator_function)
        x_circle, y_circle, x_random, y_random = shape_generator(
            noisePoints, shape_name, num_points, maxSize
        )
    else:
        circle_generator = cast(CircleGenerator, generator_function)
        x_circle, y_circle, x_random, y_random = circle_generator(
            noisePoints, circle_radius, num_points, maxSize
        )


    save_points_to_file(x_circle, y_circle, x_random, y_random, folder=folder)
