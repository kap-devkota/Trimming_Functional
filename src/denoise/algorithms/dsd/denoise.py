from .computations import compute_embedding, compute_degree_mat, compute_laplacian, compute_X_normalized
from denoise.graph.operations import densify
from numpy.linalg  import norm

import numpy as np
import itertools
import scipy.spatial.distance as spatial

def rank_edges(edgelist, X):
    """Ranks the edges in the edgelist according to their L2 distance from
    each other in the embedding.
    """
    existing_edgelist = []
    best_known_list    = []
    for ed in edgelist:
        p, q, wt = ed
        _norm    = norm(X[p] - X[q])
        best_known_list.append((p, q, wt, _norm))
    return sorted(best_known_list, key = lambda l : l[3])
    
def predict_links(X, metric="euclidean"):
    """Predicts the most likely links in a graph given an embedding X
    of a graph.

    Returns a ranked list of (edges, distances) sorted from closest to
    furthest.
    """
    distances = spatial.pdist(X, metric=metric)
    n = X.shape[0]
    edges = itertools.combinations(range(n), 2) # generator expression doesn't actualize list :)
    edges_and_distances = list(zip(edges, distances))
    edges_and_distances.sort(key=lambda x: x[1])
    return edges_and_distances

def glide_predict_links(edgelist, X, params={})
    """Predicts the most likely links in a graph given an embedding X
    of a graph.

    Returns a ranked list of (edges, distances) sorted from closest to 
    furthest.
    """
    pass
