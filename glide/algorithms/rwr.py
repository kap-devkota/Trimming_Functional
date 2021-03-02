import numpy as np
from numpy.linalg import pinv

def create_ranked_prob(A, starting_nodes, c):
    def i_deg(deg):
        i_deg = deg.copy()
        for i in range(deg.shape[0]):
            i_deg[i][i] = 1 / i_deg[i][i] if deg[i][i] != 0 else 0
        return i_deg
    e   = np.ones(A.shape[0])
    e_s = np.zeros(A.shape[0])
    deg = np.diag((A @ e).flatten())
    i_d   = i_deg(deg) 
    P     = i_d @ A
    print("Should be 5008:")
    print(np.sum(P @ np.ones((A.shape[0], 1))))
    for s in starting_nodes:
        e_s[s] = 1
    sc  = (1 - c) * (pinv(np.identity(A.shape[0]) - c * P) @ e_s)
    r   = np.argsort(sc)[::-1]
    return r, sc
    
