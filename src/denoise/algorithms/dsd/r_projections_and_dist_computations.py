from sklearn import random_projection
from sklearn.random_projection import johnson_lindenstrauss_min_dim
import numpy as np

def r_projection(input_data, no_components = None, e = 0.1):
    if no_components == None:
        no_components = johnson_lindenstrauss_min_dim(n_samples = input_data.shape[0], eps = e)

    projected_data = random_projection.GaussianRandomProjection(n_components = no_components).fit_transform(input_data)

    return projected_data


def compute_pairwise_distance_list(projected_x, no_pairs = None):
    
    rows = projected_x.shape[0]
    
    distance_list = []
    for i in range(rows):
        for j in range(i):
            r_i = projected_x[i]
            r_j = projected_x[j]
            dist = np.linalg.norm(r_i - r_j)
            distance_list.append((i, j, dist))

    s_dist = sorted(distance_list, key = lambda x: x[2])
    
    if no_pairs is not None:
        s_dist = s_dist[: no_pairs]

    return s_dist

##################################### ALL PAIRWISE DISTANCE ############################################
def compute_all_pairwise_distance_l2(projected_x, existing_edge_dict = None):
    prod = np.matmul(projected_x, projected_x.T)   
    diag = np.diagonal(prod)

    dim = diag.shape[0]

    diag = diag.reshape((dim, 1))

    e = np.ones((1, dim))
    
    diag_m = np.matmul(diag, e)

    diff = diag_m + diag_m.T - 2 * prod
    diff = diff * 1000

    # Matrix computations done

    distance_list = []

    edgelist = []

    for i in range(dim):
        for j in range(i):
            
            if existing_edge_dict is not None:
                if (i, j) in existing_edge_dict or (j, i) in existing_edge_dict:
                    continue

            edgelist.append((i, j, diff[i, j]))

    updated_edgelist = sorted(edgelist, key = lambda x: x[2], reverse = False)
    return updated_edgelist


def compute_all_pairwise_distance_l1(projected_x, existing_edge_dict = None):
    edgelist = []
    dim = projected_x.shape[0]

    for i in range(dim):
        for j in range(i):
            
            if existing_edge_dict is not None:
                if (i, j) in existing_edge_dict or (j, i) in existing_edge_dict:
                    continue

            diff =  np.sum(np.absolute(projected_x[i] - projected_x[j]))
            
            edgelist.append((i, j, diff))

    updated_edgelist = sorted(edgelist, key = lambda x: x[2], reverse = False)
    return updated_edgelist
    

def compute_all_alignment(projected_x, existing_edge_dict = None):
    diff = np.matmul(projected_x, projected_x.T)
    dim = projected_x.shape[0]
    dag = np.diag(diff)

    edgelist = []

    for i in range(dim):
        for j in range(i):

            if existing_edge_dict is not None:
                if (i, j) in existing_edge_dict or (j, i) in existing_edge_dict:
                    continue

            corr = diff[i, j] / np.sqrt(dag[i] * dag[j])
            edgelist.append((i, j, corr))

    updated_edgelist = sorted(edgelist, key = lambda x: x[2], reverse = True)
    return updated_edgelist
                    

####################################### SOME PAIRWISE DISTANCE ###############################
def compute_pairwise_distance_l2(projected_x, no_pairs = 100, existing_edge_dict = None):
    prod = np.matmul(projected_x, projected_x.T)   
    diag = np.diagonal(prod)

    dim = diag.shape[0]

    diag = diag.reshape((dim, 1))

    e = np.ones((1, dim))
    
    diag_m = np.matmul(diag, e)

    diff = diag_m + diag_m.T - 2 * prod
    diff = diff * 1000

    # Matrix computations done

    distance_list = []
    already_added = {}

    for p in range(no_pairs):
        
        min_dist = -1
        min_i = -1
        min_j = -1

        for i in range(dim):
            for j in range(i):

                if (i, j) in already_added:
                    continue

                if existing_edge_dict is not None:
                    if (i, j) in existing_edge_dict or (j, i) in existing_edge_dict:
                        continue

                if min_dist == -1 or diff[i, j] < min_dist:
                    min_i = i
                    min_j = j
                    min_dist = diff[i, j]
        
        distance_list.append((min_i, min_j, min_dist))
        already_added[(min_i, min_j)] = True

    return distance_list


def compute_pairwise_distance_l1(projected_x, no_pairs = 100, existing_edge_dict = None):

    already_added = {}
    distance_list = []

    dim = projected_x.shape[0]

    for p in range(no_pairs):
        
        min_dist = -1
        min_i = -1
        min_j = -1

        for i in range(dim):
            for j in range(i):

                if (i, j) in already_added:
                    continue

                if existing_edge_dict is not None:
                    if (i, j) in existing_edge_dict or (j, i) in existing_edge_dict:
                        continue

                diff =  np.sum(np.absolute(projected_x[i] - projected_x[j]))

                if min_dist == -1 or diff < min_dist:
                    min_i = i
                    min_j = j
                    min_dist = diff
        
        distance_list.append((min_i, min_j, min_dist))
        already_added[(min_i, min_j)] = True

    return distance_list


def compute_pairwise_alignment(projected_dist, no_pairs = 100, existing_edge_dict = None):
    
    diff = np.matmul(projected_dist, projected_dist.T)
    diag = np.diag(diff)

    dim = diff.shape[0]
    
    # Matrix computations done
    distance_list = []
    already_added = {}

    for p in range(no_pairs):
        
        ma_dist = None
        ma_i = None
        ma_j = None

        for i in range(dim):
            for j in range(i):

                if (i, j) in already_added:
                    continue

                if existing_edge_dict is not None:
                    if (i, j) in existing_edge_dict or (j, i) in existing_edge_dict:
                        continue
                
                corr = diff[i, j] / np.sqrt(diag[i] * diag[j])

                if ma_dist is None or corr > ma_dist:
                    ma_i = i
                    ma_j = j
                    ma_dist = corr 
                    
        distance_list.append((ma_i, ma_j, ma_dist))
        already_added[(ma_i, ma_j)] = True

    return distance_list

    


