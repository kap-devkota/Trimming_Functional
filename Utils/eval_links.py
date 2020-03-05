def create_expected_dict(expected_list):
    """
    This function takes in a list of edges for form `list[(p_i, q_i, w_i), ...]` and converts into a dictionary of form 
    `dict[(p_i, q_i)] = w_i`
    """
    expected_dict = {}

    for edge in expected_list:
        p1, p2, wt = edge[: 3]
        expected_dict[(p1, p2)] = wt

    return expected_dict


def count_unique_vertices(edges):
    """
    Given a list of form `list[(p_i, q_i, w_i)]` representing the edges, this function returns the number of unique vertices in the list (unique p_i and q_i's)
    """
    v_dict = {}
    for edge in edges:
        p1, p2 = edge[: 2]
        if p1 not in v_dict:
            v_dict[p1] = True
        if p2 not in v_dict:
            v_dict[p2] = True

    return len(v_dict)


def compute_metric(expected_list, predicted_list, no_iteration, include_weights = False):
    """
    This function takes in a expected_list (representing the missing edges that is supposed to be correctly
    predicted; form = `list[(p_i, q_i, w_i), ...]`) and predicted_list (representing the list of predicted
    edges, form = `list[(p_i, q_i, w_i), ...]`), and an integer value `no_iteration` representing the number
    of datapoints to be needed and returns the dictionary containing the list of accuracy datapoints. 
    """
    spacing = int(len(predicted_list) / no_iteration)
    e_dict = create_expected_dict(expected_list)
    
    no_accurate = []

    for i in range(no_iteration):
        count = 0
        index = spacing * i

        for j in range(spacing):
            p1, p2, wt = predicted_list[index + j][: 3]
                        
            key = (p1, p2) if (p1, p2) in e_dict else (p2, p1)
            if key in e_dict:
                if include_weights == False:
                    count += 1
                else:
                    count += e_dict[key]

        if len(no_accurate) != 0:
            count += no_accurate[-1]

        no_accurate.append(count)

    total_vertices = count_unique_vertices(predicted_list)

    return { "expected_size"  : len(expected_list),                        # size of expected list
             "predicted_size" : len(predicted_list),                       # size of predicted list
             "total_edges"    : total_vertices * (total_vertices - 1) / 2, # total number of edges
             "spacing"        : spacing,                                   # spacing between the acc datapoints
             "ac_list"        : no_accurate                                # acc datapoints list
        }
    


def compute_full_roc_params(rank_list, relevant_list, no_points = 10000):
    """
    Takes in a predicted list, and a expected list, and generates the roc parameters at `no_points` datapoints.
    """
    n_rank_list = len(rank_list)
    n_relevant_list = len(relevant_list)

    # no of negatives in the rank list
    n_negatives = n_rank_list - n_relevant_list
    
    spacing = int(n_rank_list / no_points)
    
    relevant_dict = {}
    for elem in relevant_list:
        p, q = elem[ : 2]
        relevant_dict[(p, q)] = True
        
    no_relevant_obtained = 0
    no_false_positives = 0

    roc_list = []
    index = 0

    for elem in rank_list:
        p, q = elem[ : 2]
    
        pair = (p, q) if (p, q) in relevant_dict else (q, p)
               
        if pair in relevant_dict:
            no_relevant_obtained += 1
            # print(pair)
        else:
            no_false_positives += 1

        index += 1
        
        if (index % spacing) == 0:
            recall = float(no_relevant_obtained) / float(n_relevant_list)
            fp = float(no_false_positives) / float(n_negatives)

            roc_list.append((fp, recall))

    return roc_list


def compute_auc(xy_list):
    """
    Given a list of points (x, y), returns the area under the curve.
    """
    prev_pair = (0.0, 0.0)
    auc = 0.0
    for next_pair in xy_list:

        auc += (next_pair[0] - prev_pair[0]) * next_pair[1]
        prev_pair = next_pair

    return auc
