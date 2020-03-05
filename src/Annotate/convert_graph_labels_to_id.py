"""
Purpose: This file takes in a ppi file, with edges having string labels (like <GENE1 GENE2 0.2134>) and converts
         it to a ppi file with integer labels (like <1 2 0.2134>). It also returns a json mapping of integer 
         labels to its corresponding string labels (like {"1" : "GENE1", "2" : "GENE2"})
"""

import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()

"""
ppi_file    : The original PPI file with string labels.
--node-dict : The output node dictionary that maps a integer label of a ppi graph to corresponding string label.
--output-file : Output PPI file with integer labels
"""
parser.add_argument("ppi_file", help = "PPI file")
parser.add_argument("-n", "--node_dict", help = "Output Nodelist dictionary")
parser.add_argument("-o", "--output_file", help = "Output PPI file")

args = parser.parse_args()

ppi_file = args.ppi_file
node_file = args.node_dict
output_file = args.output_file

node_dict = {}
r_node_dict = {}
edge_list = []
count_node = 0

with open(ppi_file, "r") as fp:
    for line in fp:
        e1, e2, wt = line.lstrip().rstrip().split()[: 3]
        wt = float(wt)

        if e1 not in node_dict:
            node_dict[e1] = count_node
            r_node_dict[count_node] = e1
            count_node += 1

        if e2 not in node_dict:
            node_dict[e2] = count_node
            r_node_dict[count_node] = e2
            count_node += 1

        edge_list.append((node_dict[e1], node_dict[e2], wt))

with open(output_file, "w") as opf:
    str_wrt = ""
    for edge in edge_list:
        wrt = "{} {} {}\n".format(edge[0], edge[1], edge[2])
        str_wrt += wrt
        
    str_wrt = str_wrt.lstrip().rstrip()

    opf.write(str_wrt)

with open(node_file, "w") as jf:
    json.dump(r_node_dict, jf)
