from .dsd.graph_operations import get_graph_from_file, densify
from .dsd.dse_computations import compute_degree_mat, compute_laplacian, compute_X_normalized

def compute_embedding(edge_list, lm = 1):
    A     = densify(edge_list)
    D     = compute_degree_mat(A)
    X     = compute_X_normalized(A, D, lm = lm)
    return X
