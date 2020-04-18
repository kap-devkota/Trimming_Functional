from .computations import compute_degree_mat, compute_laplacian, compute_X_normalized
from numpy.linalg  import norm

def get_best_scoring_unknown_links(edge_list, embedding = None, no_links = 100, lm = 1):
    emb_mat = embedding
    if emb_mat is None:
        emb_mat  = compute_embedding(edge_list, lm = lm)

    existing_edge_dict = {}
    for ed in edge_list:
        p, q, wt = ed
        existing_edge_dict[(p, q)] = wt

    return compute_pairwise_distance(emb_mat, no_pairs = no_pairs, existing_edge_dict = existing_edge_dict)

def get_from_known_links(edge_list, embedding = None, no_links = 100, lm = 1, top_scoring = True):
    emb_mat = embedding
    if emb_mat is None:
        emb_mat  = compute_embedding(edge_list, lm = lm)
    existing_edge_list = []
    best_known_list    = []
    for ed in edge_list:
        p, q, wt = ed
        _norm    = norm(emb_mat[p] - emb_mat[q])
        best_known_list.append((p, q, wt, _norm))
    best_known_list    = sorted(best_known_list, key = lambda l : l[3])
    if top_scoring:
        return best_known_list[:no_links]
    else:
        length         = len(best_known_list)
        return best_known_list[(best_known_list - no_links):best_known_list]
    
        

    
