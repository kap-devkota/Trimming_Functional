from .dsd.graph_operations import get_graph_from_file, densify
from .dsd.dse_computations import compute_degree_mat, compute_laplacian, compute_X_normalized


def parse_and_annotate(filename):
    edge_list   = get_graph_from_file(filename)
    node_dict   = {}
    r_node_dict = {}
    a_edge_list = []
    count_node  = 0
    for edge in edge_list:
        e1, e2, wt = edge
        wt         = float(wt)
        if e1 not in node_dict:
            node_dict[e1]           = count_node
            r_node_dict[count_node] = e1
            count                  += 1
        if e2 not in node_dict:
            node_dict[e2]           = count_node
            r_node_dict[count_node] = e2
            count                  += 1
        a_edge_list.append((node_dict[e1], node_dict[e2], wt))
    return a_edge_list, r_node_dict


def compute_embedding(edge_list, lm = 1):
    A     = densify(edge_list)
    D     = compute_degree_mat(A)
    X     = compute_X_normalized(A, D, lm = lm)
    return X
