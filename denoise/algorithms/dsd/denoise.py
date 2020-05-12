from .computations import compute_embedding, compute_degree_mat, compute_laplacian, compute_X_normalized
from denoise.graph.operations import densify
from numpy.linalg  import norm

import numpy as np
import itertools
import scipy.spatial.distance as spatial

def rank_edges(edgelist, X):
    """Ranks the edges in the edgelist according to their L2 distance from
    each other in the embedding.
    """
    existing_edgelist = []
    best_known_list    = []
    for ed in edgelist:
        p, q, wt = ed
        _norm    = norm(X[p] - X[q])
        best_known_list.append((p, q, wt, _norm))
    return sorted(best_known_list, key = lambda l : l[3])
    
def predict_links(X, metric="euclidean"):
    """Predicts the most likely links in a graph given an embedding X
    of a graph.

    Returns a ranked list of (edges, distances) sorted from closest to
    furthest.
    """
    distances = spatial.pdist(X, metric=metric)
    n = X.shape[0]
    edges = itertools.combinations(range(n), 2) # generator expression doesn't actualize list :)
    edges_and_distances = list(zip(edges, distances))
    edges_and_distances.sort(key=lambda x: x[1])
    return edges_and_distances

def glide_predict_links(edgelist, X, params={}):
    """Predicts the most likely links in a graph given an embedding X
    of a graph.
    Returns a ranked list of (edges, distances) sorted from closest to 
    furthest.
    @param edgelist -> A list with elements of type `(p, q, wt)`
    @param X        -> A nxk embedding matrix
    @param params   -> A dictionary with entries 
    {
    alpha => real number
    beta  => real number
    delta => real number
    loc   => String, can be `cw` for common weighted, `l3` for l3 local scoring
    }
    """

    def create_edge_dict(edgelist):
        """
        Creates an edge dictionary with the edge `(p, q)` as the key, and weight `w` as the value.
        @param  edgelist -> A list with elements of form `(p, q, w)`
        @return edgedict -> A dictionary with key `(p, q)` and value `w`.
        """
        edgedict             = {}
        for (p, q, w) in edgelist:
            edgedict[(p, q)] = w
        return edgedict
    
    def create_neighborhood_dict(edgelist):
        """
        Create a dictionary with nodes as key and a list of neighborhood nodes as the value
        @param edgelist          -> A list with elements of form `(p, q, w)`
        @param neighborhood_dict -> A dictionary with key `p` and value, a set `{p1, p2, p3, ...}`
        """
        ndict                = {}
        for ed in edgelist:
            p, q, _          = ed
            if p not in ndict:
                ndict[p]     = set()
            if q not in ndict:
                ndict[q]     = set()
            ndict[p].add(q)
            ndict[q].add(p)
        return ndict

    def compute_cw_score(p, q, edgedict, ndict):
        """
        Computes the common weighted score between p and q
        @param p        -> A node of the graph
        @param q        -> Another node in the graph
        @param edgedict -> A dictionary with key `(p, q)` and value `w`.
        @param ndict    -> A dictionary with key `p` and the value a set `{p1, p2, ...}`
        @return         -> A real value representing the score
        """
        if (len(ndict[p]) > len(ndict[q])):
            temp  = p
            p     = q
            q     = temp            
        score     = 0
        for elem in ndict[p]:
            if elem in ndict[q]:
                p_elem  = edgedict[(p, elem)] if (p, elem) in edgedict else edgedict[(elem, p)]
                q_elem  = edgedict[(q, elem)] if (q, elem) in edgedict else edgedict[(elem, q)]
                score  += p_elem + q_elem
        return score

    def compute_l3_score(p, q, edgedict, ndict):
        """
        Compute the l3 score between p and q
            L3 metric proposed by \citet{kovacs2019network} is computed as 
            $\displaystyle{L3(p,q) = \sum_{u,v} \frac{a_{p,u}\cdot a_{u,v}\cdot a_{v,q}}{\sqrt{k_u k_v}}},$ 
            where $u$ and $v$ represent all distinct pair of nodes in $G$. Here, $a_{m,n} = 1$ if there is a 
            link between nodes $m,n\in G$, and $k_m$ represents the degree of the node $m$.
        @param p        -> A node of the graph
        @param q        -> Another node in the graph
        @param edgedict -> A dictionary with key `(p, q)` and value `w`.
        @param ndict    -> A dictionary with key `p` and the value a set `{p1, p2, ...}`
        @return         -> A real value representing the score
        """
        score = 0
        for e1 in ndict[p]:
            for e2 in ndict[q]:
                a_e1_e2 = 1 if ((e1, e2) in edgedict or (e2, e1) in edgedict) else 0
                k_e1    = len(ndict[e1])
                k_e2    = len(ndict[e2])
                score  += a_e1_e2 / np.sqrt(k_e1 * k_e2)
        return score

    edgedict      = create_edge_dict(edgelist)
    ndict         = create_neighborhood_dict(edgelist)

    # Embedding
    pairwise_dist = spatial.squareform(spatial.pdist(X))
    N             = X.shape[0]
    alpha         = params["alpha"]
    local_metric  = params["loc"]

    if local_metric == "l3":
        local_metric = compute_l3_score
    elif local_metric == "cw":
        local_metric = compute_cw_score
    else:
        raise Exception("[x] The local scoring metric is not available.")

    edgelist_with_scores = []
    for i in range(N):
        for j in range(i):
            local_score = local_metric(i, j, edgedict, ndict)
            dsed_dist   = pairwise_dist[i, j]
            glide_score = (np.exp(alpha / (1 + beta * dsed_dist)) * local_score
                                + delta * 1 / dsed_dist)
            edgelist_with_scores.append((i, j, glide_score))
    return sorted(edgelist_with_scores, key = lambda l : l[2], reverse = True)    
    
