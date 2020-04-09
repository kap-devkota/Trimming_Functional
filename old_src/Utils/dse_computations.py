import numpy as np

import sys
sys.path.append("../")
from Utils.graph_operations import densify


def get_chi_ij(i, j, size):
    """
    Get the vector (e_i - e_j), where e_i is a vector of size `size` which is zero at all positions except position `i`
    @param i: `i`th index of (e_i - e_j)
    @param j: `j`th index of (e_i - e_j)
    @param size: Size of the vector.

    @return: The vector (e_i - e_j)
    """
    x = np.zeros((size, 1))
    x[i] = 1
    x[j] = -1
    return x


def get_updated_laplacian(L, L_dagger, i, j, weight):
    """
    Update the laplacian function after removing the edges between the nodes `i` and `j`.
    @param L: Laplacian matrix before removing the edges
    @param L_dagger: Pseudo inverse of the Laplacian matrix
    @param i: The first node index
    @param j: The second node index
    @param weight: The weight of the edge e_{ij}

    @return: Updated laplacian matrix
    """
    chi_ij = get_chi_ij(i,j, L_dagger.shape[0])
    psi_ij = - weight * chi_ij

    k = np.matmul(L_dagger, psi_ij)
    Ld_k = np.matmul(L_dagger , k)
    v = np.matmul(np.identity(L_dagger.shape[0]) - np.matmul(L, L_dagger), chi_ij)
    h = np.matmul(L_dagger, chi_ij)
    beta = 1 + np.matmul(np.matmul(chi_ij.T, L_dagger), psi_ij)[0,0]
    k_normsq = np.matmul(k.T, k)[0,0]
    v_normsq = np.matmul(v.T, v)[0,0]

    p_1 = -np.multiply(np.multiply(k_normsq, v), 1 / beta) - k
    q_1 = -np.multiply(np.multiply(v_normsq, Ld_k), 1 / beta) - h

    sigma_1 = k_normsq * v_normsq + beta ** 2
    L_tilda_dagger = L_dagger + np.multiply(1/beta , np.matmul(v, Ld_k.T)) - np.multiply(beta/sigma_1 , np.matmul(p_1, q_1.T))
    return L_tilda_dagger


def compute_X(D, L_dagger):
    """
    Computing the state diffusion matrix
    @param D: The diagonal matrix of degrees
    @param L_dagger: The pseudo inverse of the Laplacian matrix
    @return: The state diffusion matrix X
    """
    e = np.ones((D.shape[0], 1))
    delta_total = np.matmul(e.T, np.matmul(D, e))[0, 0]

    x = np.matmul(D, e)
    w = np.matmul(L_dagger, x)
    y = np.matmul(D, w)
    z = np.matmul(w.T, x)[0, 0]

    X = np.matmul(D, L_dagger) - np.multiply(1 / delta_total,  np.matmul(x, w.T)) - np.multiply(1/delta_total, np.matmul(y, e.T)) + np.multiply(1/(delta_total ** 2) * (z + delta_total) , np.matmul(x, e.T))
    return X


def compute_degree_mat(A, dim = None, directed = False, is_sparse = False, return_flattened = True):
    if is_sparse == True:
        A = densify(A, dim = dim, directed = directed)
    e = np.ones((A.shape[0], 1))
    deg = np.matmul(A, e)
    if not return_flattened:
        deg = np.diag(deg.flatten())
    return deg


def compute_laplacian(A, dim = None, directed = False, deg = None, is_deg_flattened = True, is_sparse = False):
    if is_sparse:
        A = densify(A, dim = dim, directed = directed)
    if deg is None:
        deg = compute_degree_mat(A, dim = dim, directed = directed, is_sparse = False)
    if is_deg_flattened:
        deg = np.diag(deg.flatten())
    L = deg - A
    return L


def compute_pinverse_diagonal(diag):
    i_diag = np.zeros((diag.shape[0], diag.shape[0]))
    for i in range(diag.shape[0]):
        di = diag[i, i]
        if di != 0.0:
            i_diag[i, i] = 1 / float(di)

    return i_diag
    

def compute_X_n(L, D):
    D_i = compute_pinverse_diagonal(D)
    X_i = np.matmul(D_i, L)
    
    e = np.ones((L.shape[0], 1))
    scale = np.matmul(e.T, np.matmul(D, e))[0, 0]
    
    W = np.matmul(e, np.matmul(e.T, D))
    W = np.multiply(1 / scale, W)
    X_i = X_i + W

    ### Normalize with steady state
    P_ = np.matmul(D, e)
    P_ = np.sqrt(P_)
    P_ = np.diag(P_.flatten())
    X_i = np.matmul(X_i, P_)
    ###
    return np.linalg.pinv(X_i)
    

def compute_X_normalized(A, D, t, lm = 1, is_normalized = True):
    D_i = compute_pinverse_diagonal(D)
    P = np.matmul(D_i, A)
    Identity = np.identity(A.shape[0])
    e = np.ones((A.shape[0], 1))

    # Compute W
    scale = np.matmul(e.T, np.matmul(D, e))[0, 0]
    W = np.multiply(1 / scale, np.matmul(e, np.matmul(e.T, D)))

    up_P = np.multiply(lm, P - W)
    X_ = Identity - up_P
    X_i = np.linalg.pinv(X_)

    if t > 0:
        LP_t = Identity - np.linalg.matrix_power(up_P, t)
        X_i = np.matmul(X_i, LP_t)
    
    if is_normalized == False:
        return X_i
    
    # Normalize with steady state
    SS = np.sqrt(np.matmul(D, e))
    SS = compute_pinverse_diagonal(np.diag(SS.flatten()))

    return np.matmul(X_i, SS)


def compute_X_t_normalized(A, D, t, lm = 1, is_normalized = True):
    D_i = compute_pinverse_diagonal(D)
    P = np.matmul(D_i, A)
    e = np.ones((A.shape[0], 1))

    # Compute W
    scale = np.matmul(e.T, np.matmul(D, e))[0, 0]
    W = np.multiply(1 / scale, np.matmul(e, np.matmul(e.T, D)))

    P_t = np.linalg.matrix_power(np.multiply(lm, P - W), t)

    if is_normalized == False:
        return P_t

    # Compute the Inverse steady state
    SS = np.sqrt(np.matmul(D, e))
    SS = compute_pinverse_diagonal(np.diag(SS.flatten()))

    return np.matmul(P_t, SS)

