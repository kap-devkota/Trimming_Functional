import numpy as np
import argparse
import json
import sys

sys.path.append('../')

from Utils.graph_operations import densify
from Utils.dse_computations import compute_degree_mat, compute_pinverse_diagonal
from numpy import linalg as LA

parser = argparse.ArgumentParser()
parser.add_argument("ppi_network", help = "PPI network to be converted")
parser.add_argument("out_file", help = "Output X-State network with changed label")
parser.add_argument("-j", "--json", default = None, help = "The output nodelist dictionary")
parser.add_argument("-r", "--reduced_dims", type = int, default = -1, help = "The reduced dimensions size")
args          = parser.parse_args()
red_dims      = args.reduced_dims
node_dict     = {}
r_node_dict   = {}
updated_edges = []

## Creating the adjacency matrix, from input links, with labels given  ##
with open(args.ppi_network, "r") as file:
   node_counter = 0
   for elem in file:
      words          = elem.rstrip().lstrip().split()
      n1, n2, weight = words        
      weight         = float(weight)
      if n1 not in node_dict:
         node_dict[n1]             = node_counter
         r_node_dict[node_counter] = n1
         node_counter             += 1
      if n2 not in node_dict:
         node_dict[n2]             = node_counter
         r_node_dict[node_counter] = n2 
         node_counter             += 1
      n1 = node_dict[n1]
      n2 = node_dict[n2]
      updated_edges.append((n1, n2, weight))

## Storing the matrix index to label dictionary ##
if args.json is not None:
   with open(args.json, "w") as outj:
      json.dump(r_node_dict, outj)

A   = densify(updated_edges)
D   = compute_degree_mat(A, dim = None, directed = False, is_sparse = False, return_flattened = False)
D_p = np.sqrt(D)
D_n = np.sqrt(compute_pinverse_diagonal(D))
N   = np.matmul(np.matmul(D_n, A), D_n)
L   = np.identity(A.shape[0]) - N

spec, X_spec = LA.eig(L + np.identity(A.shape[0]))
index_chosen = np.argsort(spec)[1 : red_dims + 1]
spec_mat     = spec[index_chosen] - 1
spec_mat     = np.reciprocal(spec_mat)
spec_mat     = np.diag(spec_mat)
X_spec_r     = X_spec[: , index_chosen]
X_spec_r     = np.matmul(D_n, X_spec_r)
reduced_mat  = np.matmul(X_spec_r, spec_mat)

np.save(args.out_file, reduced_mat)
