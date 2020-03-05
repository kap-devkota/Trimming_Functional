import argparse
import json
import numpy as np
from numpy.linalg import norm
import sys
sys.path.append("../")
from Utils.graph_operations import get_graph_from_file

parser = argparse.ArgumentParser();

parser.add_argument("graph_file", help = "Graph file")
parser.add_argument("state_file", help = "State file")
parser.add_argument("output_file", help = "Output file")
parser.add_argument("-t", "--to_trim", type = float, default = 0.1, 
                    help = "Percentage of edges to remove")
parser.add_argument("-j", "--json_file", help = "JSON file")

args = parser.parse_args();

graph_file     = args.graph_file
state_file     = args.state_file
index_to_label = args.json_file
p_to_trim      = args.to_trim

# SETUP json dictionary
index_to_label_dict = None
with open(index_to_label, "r") as jf:
    index_to_label_dict = json.load(jf)

label_to_index_dict = {}
for key in index_to_label_dict:
    label_to_index_dict[index_to_label_dict[key]] = key

# SETUP Embedding matrix
state_mat    = np.load(state_file)

# SETUP graph adjacency list
adjGraph     = get_graph_from_file(graph_file)
l2_distances = []

# Get l2 norm of every edges based on the embedding
for ed in adjGraph:
    n1, n2, w = ed
    i1        = int(label_to_index_dict[n1])
    i2        = int(label_to_index_dict[n2])
    l2dist    = norm(state_mat[i1] - state_mat[i2])
    l2_distances.append((n1, n2, l2dist, w))

l2_distances = sorted(l2_distances,
                      key = lambda x : x[2])[ : int((1 - p_to_trim) * len(adjGraph))]

str_to_wr = ""
for e in l2_distances:
    n1, n2, d, w = e
    str_to_wr   += "{} {} {}\n".format(n1, n2, w)

str_to_wr = str_to_wr.lstrip().rstrip()

with open(args.output_file, "w") as op:
    op.write(str_to_wr)



