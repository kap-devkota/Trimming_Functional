import numpy as np


def compute_adjacency_dictionary(edge_list):
    """
    Given a list of edges of form `list[(p1, q1, w1), ...]`, create a dictionary which has every node as a key 
    . The dictionary has two entries: `size` stores the total sum of all the edge weights containing the node 
    `e1`, and `connections` is another dictionary storing the weights of edges that are connected to `e1`
    """
    edge_dict = {}
    for edge in edge_list:
        # print(edge)
        e1, e2, w = edge[: 3]
        if e1 not in edge_dict:
            edge_dict[e1] = {"size": 0,
                             "connections": {}
                            }
        
        if e2 not in edge_dict:
            edge_dict[e2] = {"size": 0,
                             "connections": {}
                             }

        edge_dict[e1]["size"] += w
        edge_dict[e2]["size"] += w

        edge_dict[e1]["connections"][e2] = w
        edge_dict[e2]["connections"][e1] = w

    return edge_dict


def jaccard_index_u(node1, node2, edge_dict):    
    """
    Given two nodes and an edge_dict created from the `compute_adjacency_dictionary` datastructure, returns the
    unweighted jaccard score.
    """
    no_intersections = 0

    if edge_dict[node2]["size"] < edge_dict[node1]["size"]:
        temp = node1
        node1 = node2
        node2 = temp

    for key in edge_dict[node1]["connections"]:
        if key in edge_dict[node2]["connections"]:
            no_intersections += 1

    no_union = len(edge_dict[node1]["connections"]) + len(edge_dict[node2]["connections"]) - no_intersections

    return float(no_intersections) / float(no_union)


def common_neighbors_u(node1, node2, edge_dict):
    """
    Given two nodes and an edge_dict created from the `compute_adjacency_dictionary` datastructure, returns the
    unweighted common neighbors score.
    """
    no_intersections = 0

    if edge_dict[node2]["size"] < edge_dict[node1]["size"]:
        temp = node1
        node1 = node2
        node2 = temp

    for key in edge_dict[node1]["connections"]:
        if key in edge_dict[node2]["connections"]:
            no_intersections += 1

    return no_intersections


def adamic_adar_u(node1, node2, edge_dict):
    """
    Given two nodes and an edge_dict created from the `compute_adjacency_dictionary` datastructure, returns the
    unweighted adamic adar score.
    """
    aa_index = 0
    
    if edge_dict[node2]["size"] < edge_dict[node1]["size"]:
        temp = node1
        node1 = node2
        node2 = temp

    for key in edge_dict[node1]["connections"]:
        if key in edge_dict[node2]["connections"]:

            no_conn = len(edge_dict[key]["connections"])

            if no_conn == 1:
                no_conn += 0.5

            aa_index += 1 / np.log2(no_conn)

    return aa_index


def jaccard_index(node1, node2, edge_dict):    
    """
    Given two nodes and an edge_dict created from the `compute_adjacency_dictionary` datastructure, returns the
    weighted jaccard score.
    """
    no_intersections = 0

    if edge_dict[node2]["size"] < edge_dict[node1]["size"]:
        temp = node1
        node1 = node2
        node2 = temp

    for key in edge_dict[node1]["connections"]:
        if key in edge_dict[node2]["connections"]:
            no_intersections += (edge_dict[node1]["connections"][key] + edge_dict[node2]["connections"][key])/ 2

    no_union = edge_dict[node1]["size"] + edge_dict[node2]["size"] - no_intersections

    if no_union == 0:
        return 0

    return float(no_intersections) / float(no_union)


def common_neighbors(node1, node2, edge_dict):
    """
    Given two nodes and an edge_dict created from the `compute_adjacency_dictionary` datastructure, returns the
    weighted common neighbors score.
    """
    no_intersections = 0

    if edge_dict[node2]["size"] < edge_dict[node1]["size"]:
        temp = node1
        node1 = node2
        node2 = temp

    for key in edge_dict[node1]["connections"]:
        if key in edge_dict[node2]["connections"]:
            no_intersections += edge_dict[node2]["connections"][key] + edge_dict[node1]["connections"][key]

    return no_intersections


def adamic_adar(node1, node2, edge_dict):
    """
    Given two nodes and an edge_dict created from the `compute_adjacency_dictionary` datastructure, returns the
    weighted adamic adar score.
    """
    aa_index = 0
    
    if edge_dict[node2]["size"] < edge_dict[node1]["size"]:
        temp = node1
        node1 = node2
        nodee2 = temp

    for key in edge_dict[node1]["connections"]:
        if key in edge_dict[node2]["connections"]:

            no_conn = edge_dict[key]["size"]
            
            if no_conn <= 1:
                no_conn = 1 + no_conn

            aa_index += 1 / np.log2(no_conn)

    return aa_index


def predict_edges(edge_dict, metric = "common_neighbors", no_pred = -1):
    """
    Given the edge dictionary datastructure constructed from `compute_adjacency_dictionary` function, return a
    list of edges ranked by the metric given in arguments.
    """

    edges = list(edge_dict.keys())

    total_edges = len(edges)
    
    pred_edge_list = []
    for i in range(total_edges):
        for j in range(i):
            if edges[j] not in edge_dict[edges[i]]["connections"]:
                if metric == "common_neighbors":
                    index = common_neighbors(edges[i], edges[j], edge_dict)
                elif metric == "adamic_adar":
                    index = adamic_adar(edges[i], edges[j], edge_dict)
                elif metric == "jaccard_index":
                    index = jaccard_index(edges[i], edges[j], edge_dict)
                elif metric == "common_neighbors_u":
                    index =  jaccard_index_u(edges[i], edges[j], edge_dict)
                elif metric == "adamic_adar_u":
                    index = adamic_adar_u(edges[i], edges[j], edge_dict)
                else:
                    index = common_neighbors_u(edges[i], edges[j], edge_dict)
                    
                pred_edge_list.append((edges[i], edges[j], index))

    sorted_pred_edge = sorted(pred_edge_list, key = lambda x: x[2], reverse = True)

    if no_pred < 0:
        return sorted_pred_edge
    else:
        return sorted_pred_edge[ : no_pred]
                

def generate_heuristic_rank(edge_list, metric = "common_neighbors", no_pred = -1):
    """
    Given a list of graph edges, metric and an integer representing the number of edges to output, returns 
    a list of graph edges ranked based on the metric score.
    """
    edge_dict = compute_adjacency_dictionary(edge_list)
    return predict_edges(edge_dict, metric = metric, no_pred = no_pred)
