# Granular Data Clustering in Python

This project explores **granular one-cluster extraction**: instead of partitioning data into many groups, we identify one dominant, internally coherent cluster and treat the remaining points as noise. The implementation combines and adapts DBSCAN, Single Linkage, and Complete Linkage, then evaluates them on synthetic datasets with controlled noise levels.

## Project Structure

```text
project_root/
├── pyproject.toml
├── uv.lock
├── src/
│   ├── point_generators.py  # Data generation
│   ├── DBscan.py            # DBSCAN implementation
│   ├── KNN.py               # mean k-NN distance (d_k)
│   ├── granule.py           # shared admissibility rule and granule helpers
│   ├── main.py              # linkage + experiment runner (data only)
│   └── plots.py             # plots on demand from saved data
└── results/
    ├── summary.csv
    ├── wyniki_czasu_wykonania.csv
    └── <shape>/<radius>/<noise_points>/...
```

## Problem Domain

1. **Cluster Analysis**
    - Techniques of unsupervised classification of unlabeled data.
2. **Granular Computing**
    - Information management by creating data granules at different levels of detail.
3. **Cluster Detection in Noise**
    - Identification of significant structures even in the presence of a large number of interfering points.

## Research Objectives

- Design and implementation of granular clustering methods, enabling the selection of **one dominant cluster**.
- Development of quantitative criteria for evaluating cluster boundaries.
- Comparison of three clustering approaches: DBSCAN, Single Linkage, Complete Linkage.
- Conducting experiments on synthetic datasets with different shapes and noise levels.
- Evaluation of computational efficiency of all algorithms.

## Why One Group?

Typical clustering algorithms divide data into multiple clusters. In our approach, we are **only** interested in identifying one main cluster:

- Focus on the most significant structure in the data.
- Avoiding excessive segmentation leading to artifacts.
- Unambiguous criteria defining cluster boundaries.

## Principle of Justified Granulation

Granular clustering is based on combining two complementary measures:

1. **Cluster Size (N)**
2. **Internal Homogeneity**:
   - Average distance between points (d̄)
   - Average distance to the k-th neighbor (d_k)

Criteria (numbering kept from the original set P1–P16, so earlier results stay comparable):

$$
P1 = \frac{N}{\bar{d}}, \quad
P4 = \frac{N}{d_3}, \quad
P5 = \frac{N}{d_3^2}, \quad
P6 = \frac{\sqrt{N}}{d_3^2}
$$

where \(\bar{d}\) is the average pairwise distance in the granule and \(d_3\) the average
distance to the 3 nearest neighbours. Normalized criteria following Pedrycz's principle of
justifiable granularity (coverage × specificity, all factors in [0, 1], independent of units):

$$
P16 = \frac{N}{N_{all}} \times \left(1 - \frac{\bar{d}}{\bar{d}_{all}}\right), \quad
P17 = P16 \times \left(1 - \frac{d_3}{d_{3,all}}\right)
$$

where \(\bar{d}_{all}\) and \(d_{3,all}\) are computed on the whole data set (negative factors
are clipped to 0). P18 is a stability criterion: how wide a parameter range yields exactly the
same granule — \(\ln(d_{max,high} / d_{max,low})\) for linkage, the number of
(eps, min_samples) grid cells for DBSCAN.

Removed: P2, P3 (always pick the smallest admissible granule) and P7–P15 (\(d_k\) for
k = 5, 7, 9 gave the same choices as k = 3).

## Theoretical Foundations of Algorithms

### DBSCAN
- Complexity: \(O(n \log n)\) when using spatial indexes; otherwise \(O(n^2)\).
- Two key parameters: radius ε and minimum number of points MinPts.

### Single Linkage
- Principle: distance between clusters = minimum distance between points.
- Disturbances: may connect long chains.

### Complete Linkage
- Principle: distance between clusters = maximum distance between points.
- Advantage: creates more compact clusters; disadvantage: sensitive to isolated points.

## Requirements
- Python 3.9+
- Libraries:
  - `numpy` (numerical calculations)
  - `matplotlib` (visualization)
  - `scikit-learn` (DBSCAN and KNN implementation)
  - `pandas` (data processing)
  - `scipy` (hierarchical calculations)

## Installation

```bash
git clone https://github.com/user/clustering-project.git
cd clustering-project
uv sync
```

## Example Usage

Algorithms only compute and save data. Plots are drawn on demand afterwards.

### Run the comparison
```bash
uv run python -m src.main --quick   # circle_A + ring, 50% noise, coarse DBSCAN grid (~10 s)
uv run python -m src.main           # full grid: 4 shapes x 5 noise levels
```

Per configuration (`results/<shape>/<radius>/<noise_points>/`):
- `points.txt`: x, y, true label (1 = cluster, 2 = noise)
- `single/`, `complete/`: `*LinkageCandidates.csv`, one row per granule candidate from the tree
- `dbScan/dbScanResults.csv`: one row per (eps, min_samples) candidate
- `summary.csv`: best parameters, granule size and accuracy per method x criterion
- `labels.csv`: selected partitions, one column per method x criterion (e.g. `single_p1`, `dbscan_p3`)

Across all configurations: `results/summary.csv` and `results/wyniki_czasu_wykonania.csv` (run times).

All methods use the same admissibility rule (`src/granule.py`): the granule has at least 10% of
the points and does not cover all of them.

### Plots on demand
```bash
uv run python -m src.plots partition results/circle_A/30/500 single_p1
uv run python -m src.plots dbscan-map results/circle_A/30/500 p1
uv run python -m src.plots linkage-curve results/circle_A/30/500 complete p1
uv run python -m src.plots dendrogram results/circle_A/30/500 single
uv run python -m src.plots accuracy results --criterion p1 --shape circle_A
```
Add `--out file.png` (or `.pdf` / `.svg`) to save instead of showing.

## Experimental Results

| Algorithm       | Accuracy (average) | Relative Time | Notes                             |
|-----------------|-------------------|--------------|-----------------------------------|
| DBSCAN          | 0.89              | 1.0          | Sensitive to ε, MinPts            |
| Single Linkage  | 0.82              | 0.2          | Good for chain-like shapes        |
| Complete Linkage| 0.75              | 0.25         | Worst with high noise             |

## Conclusions

- **Best** algorithm for granular detection of a single cluster: DBSCAN, but requires parameter optimization.
- **Single Linkage** performs better with atypical shapes (von Mises).
- **Complete Linkage** not recommended in the presence of strong noise.
- Increase in noise share above 60% significantly reduces the quality of all algorithms.

## Development Possibilities

- Automatic parameter selection using global optimization techniques or learning heuristics.
- Application of hybrid algorithms (DBSCAN + hierarchical).
- Extrapolation to multidimensional data and dynamic data streams.

## License

This project is released under the MIT License. See the `LICENSE` file.
