import numpy as np
import json


def create_expected_association_dict(expected_edges_list):
    """
    Given a association list of a graph, create a new datastructure, which is a dictionary with nodes as key.
    Each entry is also a dictionary with nodes as key. The entry at p contains the key q if p, q is a valid edge
    in the expected_edges_list.
    """
    assoc_dict = {}
    for edge in expected_edges_list:
        p, q, wt = edge
        
        if p not in assoc_dict:
            assoc_dict[p] = {}

        assoc_dict[p][q] = True

        if q not in assoc_dict:
            assoc_dict[q] = {}

        assoc_dict[q][p] = True

    return assoc_dict


def compute_ranking_recall_fp_prec(expected_dict, predicted_rank_list, no_sample_pts = 100):
    """
    Given a datastructure produced by `create_expected_association_dict`, and a predicted association list, 
    compute the auc by generating `tpr, fpr` values at `no_sample_pts` points.
    """
    n_expected_dict = len(expected_dict)
    n_predicted_rank_list = len(predicted_rank_list)
    
    spacing = int(n_predicted_rank_list / (no_sample_pts - 2))
    recall = [0.0]
    fp     = [0.0]
    prec   = [0.0]

    true_positives = 0
    false_positives = 0

    for i in range(no_sample_pts - 2):        
        start = i * spacing
        
        for j in range(spacing):
            q = predicted_rank_list[start + j]

            if (q in expected_dict):
                true_positives += 1
            else:
                false_positives += 1

        tpr = float(true_positives) / n_expected_dict
        fpr = float(false_positives) / (n_predicted_rank_list - n_expected_dict)
        pr  = float(true_positives) / ((i + 1) * spacing)

        recall.append(tpr)
        fp.append(fpr)
        prec.append(pr)

    recall.append(1.0)
    fp.append(1.0)
    prec.append(float(true_positives) / n_predicted_rank_list)
 
    return (recall, fp, prec)


def read_nodes_to_check(nlist_file):
    """
    Given a file containing all the nodes to sample for nodebased ranking, return the empty nodes dictionary.
    """
    nodes_dict = {}
    with open(nlist_file, "r") as n_file:
        nodes_dict = json.load(n_file)

    empty_nodes_dict_return = {}
    for node in nodes_dict["nodes_to_check"]:
        empty_nodes_dict_return[node] = None

    return empty_nodes_dict_return



def get_nodes_entries_from_predicted_list(nodes_dict, predicted_list):
    """
    Given a empty nodes dictionary created from read_nodes_to_check and predicted association list, populate
    empty node dictionary by 
    1. Seeing if the node is a key in the dictionary
       If it is, then it is selected for nodebased ranking
    2. If it is in the dictionary, add q to the list at nodes_dict[<key>]
    """
    for edge in predicted_list:
        p, q, wt = edge

        if p in nodes_dict:
            if nodes_dict[p] is None:
                nodes_dict[p] = []
            nodes_dict[p].append(q)

        if q in nodes_dict:
            if nodes_dict[q] is None:
                nodes_dict[q] = []

            nodes_dict[q].append(p)

    return nodes_dict


def select_random_nodes_to_test(nodes_list, numbers_to_check = 1000, out_filename = None):
    """
    Randomly selectes `numbers_to_check` samples from nodes_list
    """
    indices = np.random.permutation(len(nodes_list))[ : numbers_to_check]
    nodes_to_write = [nodes_list[x] for x in indices]
    dict_to_return = {"nodes_to_check" : nodes_to_write}

    if out_filename != None:
        with open(out_filename, "w") as ofp:
            json.dump(dict_to_return, ofp)

    return dict_to_return


def compute_average_roc_across_nodes(nodes_filename, expected_list, predicted_list, no_sample_pts = 100):
    """
    Given a nodes_filename containing all the nodes selected for nodebased ranking, association list of 
    edges,, and an association list of predicted edges, computes the roc parameters by generating 
    `no_sample_pts` number of <tpr, fpr> datapoints at uniform spacing.
    """
    empty_node_dict = read_nodes_to_check(nodes_filename)
    predicted_node_dict = get_nodes_entries_from_predicted_list(empty_node_dict, predicted_list)

    expected_node_dict = create_expected_association_dict(expected_list)

    n_nodes = len(predicted_node_dict)
    average_recall = None
    average_fp     = None
    average_prec   = None

    for node in predicted_node_dict:
        p_n_list = predicted_node_dict[node]
        
        if node in expected_node_dict:
            e_n_dict = expected_node_dict[node]        
        else:
            n_nodes -= 1
            continue

        recall, fp, prec = compute_ranking_recall_fp_prec(e_n_dict, p_n_list, no_sample_pts = no_sample_pts)

        if average_recall == None:
            average_recall    = list(recall)
            average_prec      = list(prec)
            average_fp        = list(fp)
        else:
            for i in range(no_sample_pts):
                average_recall[i] += recall[i]
                average_fp[i]     += fp[i]
                average_prec[i]   += prec[i]
                
    
    for i in range(no_sample_pts):
        average_recall[i] = average_recall[i] / n_nodes
        average_fp[i]     = average_fp[i] / n_nodes
        average_prec[i]   = average_prec[i] / n_nodes

    return average_recall, average_fp, average_prec


