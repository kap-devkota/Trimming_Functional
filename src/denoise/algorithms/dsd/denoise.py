from .computations import compute_degree_mat, compute_laplacian, compute_X_normalized

def get_best_scoring_unknown_links(edge_list, embedding = None, no_links = 100, lm = 1):
    emb_mat = embedding
    if emb_mat is None:
        emb_mat  = compute_embedding(edge_list, lm = lm)

    existing_edge_dict = {}
    for ed in edge_list:
        p, q, wt = ed
        existing_edge_dict[(p, q)] = wt

    return compute_pairwise_distance(emb_mat, no_pairs = no_pairs, existing_edge_dict = existing_edge_dict)
