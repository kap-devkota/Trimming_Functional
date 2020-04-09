import numpy as np
from numpy import linalg as LA
import argparse
import json

import sys
sys.path.append('../')

from Utils.graph_operations import densify
from Utils.dse_computations import compute_degree_mat, compute_X_normalized, compute_X_t_normalized

parser = argparse.ArgumentParser()
parser.add_argument("ppi_network", help = "PPI network to be converted")
parser.add_argument("out_file", help = "Output X-State network with changed label")
parser.add_argument("-l", "--laplacian", default = None, help = "The output laplacian")
parser.add_argument("-d", "--degree", default = None, help = "The output degree file")
parser.add_argument("-n", "--no_randwalk", default = None, help = "Number of Random Walk")
parser.add_argument("-j", "--json", default = None, help = "The output nodelist dictionary")
parser.add_argument("-p", "--pval", type = float, default = 1.0, help = "The lambda value")
parser.add_argument("-i", "--is_tstp", action = "store_true", default = False)
parser.add_argument("-z", "--z_normalized", action = "store_false", default = True)
parser.add_argument("-r", "--reduced_dims", type = int, default = -1, help = "The reduced dimensions size")
args = parser.parse_args()

is_normalized = args.z_normalized

# A dictionary that converts the node label "string" from a file to its corresponding integer index in 
# the adjacency matrix
node_dict = {}

# A dictionary that converts the integer index of the adjacency matrix to its corresponding integer label
r_node_dict = {}

# Edges obtained from the files are stored here, with integer indexes
updated_edges = []

with open(args.ppi_network, "r") as file:
   node_counter = 0
   for elem in file:
      words = elem.rstrip().lstrip().split()
      n1, n2, weight = words        
      weight = float(weight)
        
      # If a given node is not in dictionary, add it to node_dict and get the corresponding integer id, 
      # represented by node_counter
      if n1 not in node_dict:
         node_dict[n1] = node_counter
         r_node_dict[node_counter] = n1
         node_counter += 1

      if n2 not in node_dict:
         node_dict[n2] = node_counter
         r_node_dict[node_counter] = n2 
         node_counter += 1
          
      n1 = node_dict[n1]
      n2 = node_dict[n2]
      updated_edges.append((n1, n2, weight))

# Save the integer index to node label dictionary as json
if args.json is not None:
   with open(args.json, "w") as outj:
      json.dump(r_node_dict, outj)

# Convert adjacency list to numpy matrix
A = densify(updated_edges)

deg = compute_degree_mat(A, dim = None, directed = False, is_sparse = False, return_flattened = False)

if args.degree is not None:
   np.save(args.degree, np.diagonal(deg))

rand_walk = 0

if args.no_randwalk is not None:
   rand_walk = int(args.no_randwalk)

# Is the matrix Coifmann or DSD?
if args.is_tstp is False:
   X = compute_X_normalized(A, deg, rand_walk, args.pval, is_normalized = is_normalized)
else:
   if rand_walk == 0:
      rand_walk = 3
   X = compute_X_t_normalized(A, deg, rand_walk, args.pval, is_normalized = is_normalized)

if args.reduced_dims > 0:
   spec, X_spec = LA.eig(X)
   X            = X_spec[:, :args.reduced_dims]

np.save(args.out_file, X)
