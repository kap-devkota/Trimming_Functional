from .computations import compute_degree_mat, compute_laplacian, compute_X_normalized
from denoise.graph.operations import densify

import itertools
import scipy.spatial.distance as spatial

def predict_links(A, lm = 1):
    """Predicts the most likely links in a graph using the DSD diffusion
    process.

    Returns a ranked list of (edges, distances) sorted from closest to
    furthest.
    """
    D = compute_degree_mat(A)
    X = compute_X_normalized(A, D, -1, lm=lm)

    distances = spatial.pdist(X)
    n = A.shape[0]
    edges = itertools.combinations(range(n), 2) # generator expression doesn't actualize list :)
    edges_and_distances = list(zip(edges, distances))
    edges_and_distances.sort(key=lambda x: x[1])
    return edges_and_distances
