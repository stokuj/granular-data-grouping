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
│   ├── KNN.py               # kNN algorithm implementation
│   └── main.py              # Experimental script
└── results/
    ├── time_of_exec.csv
    ├── circle/...
    ├── ring/...
    └── normal/...
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

Criteria formulas:

$$
P1 = N \times \frac{1}{\bar{d}}
$$

$$
P2 = N \times \left(\frac{1}{\bar{d}}\right)^2
$$

$$
P3 = \sqrt{N} \times \left(\frac{1}{\bar{d}}\right)^2
$$

and variants:

$$
P1_k = N \times \frac{1}{d_k}, \quad
P2_k = N \times \left(\frac{1}{d_k}\right)^2, \quad
P3_k = \sqrt{N} \times \left(\frac{1}{d_k}\right)^2
$$

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

### Data Generation
```python
from src import point_generators as pg
# generate 800 points in a circle with radius 50 + 200 noise points
pg.generate_points_circle(
    output_folder='results/circle',
    noise_points=200,
    circle_radius=50,
    num_points=800,
    max_size=100
)
```

### DBSCAN
```python
from src.DBscan import DBscanAlgorithmLoop, DBscanChart, DBscan
# iterate over eps ∈ [2,20], step=0.5, MinPts ∈ [5,50]
DBscanAlgorithmLoop(2,20,0.5,5,50,'results/circle')
# analyze charts and select optimal parameters
eps, min_pts, p_val = DBscanChart('results/circle', show_picture=True, p_id='p1')
# final execution
DBscan(
    file_path='results/circle/points.txt',
    folder='results/circle',
    epsilon=eps,
    samples=min_pts,
    p_id='p1',
    p_val=p_val
)
```

### Single / Complete Linkage
```python
from src.main import makeDendrogram, LinkageAlgorithmLoop, LinkageAlgorithm
# calculate d_min, d_max range
d_min, d_max = makeDendrogram('results/circle/points.txt', method='single', draw=True)
# iterate over 200 steps
LinkageAlgorithmLoop(
    path='results/circle',
    file_path='results/circle/points.txt',
    method='single',
    max_d=d_min,
    max_d_range=d_max,
    num_measurements=200,
    result_path='results/circle/single/results_loop.csv'
)
# final run for optimal d_max
opt_d = 10.5  # result from chart
LinkageAlgorithm(
    file_path='results/circle/points.txt',
    method='single',
    max_d=opt_d,
    name='opt',
    folder='results/circle',
    show_picture=True
)
```

## Implementation Details

1. `src/point_generators.py`:
    - Function `generate_points`: wrapper selecting the appropriate generator.
    - Output formats: `points.txt` with three columns (x, y, label).

2. `src/DBscan.py`:
    - `DBscanAlgorithmLoop`: saves P1–P15 results to CSV.
    - `DBscanChart`: draws charts of measures vs eps / MinPts.
    - `DBscan`: generates final clusters and saves `DBscanResults.csv`.

3. `src/KNN.py`:
    - Easy integration with `LinkageAlgorithmLoop` for calculating P1–P15 metrics.

4. `src/main.py`:
    - Import of all modules.
    - Configuration of experiment parameters.
    - Loops over noise levels, radii, and shapes.
    - Saving results to `wyniki_czasu_wykonania.csv`.

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
