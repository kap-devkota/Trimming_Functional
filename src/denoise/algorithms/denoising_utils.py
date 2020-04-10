from .dsd.graph_operations import get_graph_from_file, densify
from .dsd.dse_computations import compute_degree_mat, compute_laplacian, compute_X_normalized
from .dsd.r_projections_and_dist_computations import compute_pairwise_distance_l2

def compute_embedding(edge_list, lm = 1):
    A     = densify(edge_list)
    D     = compute_degree_mat(A)
    X     = compute_X_normalized(A, D, lm = lm)
    return X

def get_best_scoring_unknown_links(edge_list, embedding = None, no_links = 100, lm = 1):
    emb_mat = embedding
    if (emb_mat == None):
        emb_mat  = compute_embedding(edge_list, lm = lm)

    existing_edge_dict = {}
    for ed in edge_list:
        p, q, wt = ed
        existing_edge_dict[(p, q)] = wt

    return compute_pairwise_distance(emb_mat, no_pairs = no_pairs, existing_edge_dict = existing_edge_dict)

    
    
