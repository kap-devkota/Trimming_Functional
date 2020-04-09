import networkit as nk
import numpy as np

"""
Parses a GO label file.

Outputs two dicts:
 - One associating GO labels to proteins
 - One associating proteins to GO labels
"""
def parse_go_label_file(fname):
    with open(fname, "r") as f:
        rows = f.readlines()
        go_to_proteins = {}
        proteins_to_go = {}
        for row in rows:
            words = row.split()
            golabel, proteins = words[0], words[1:]
            go_to_proteins[golabel] = proteins
            for protein in proteins:
                if protein in proteins_to_go:
                    proteins_to_go[protein].append(golabel)
                else:
                    proteins_to_go[protein] = [golabel]

        return go_to_proteins, proteins_to_go

"""
Parses a DREAM network file.

Outputs a triple:
  - A weighted graph G
  - A list mapping node indices to names
  - A dictionary from node names to node indices
"""
def parse_dream_network_file(fname):
    with open(fname, "r") as f:
        G, node_map = nk.Graph(weighted=True), {}
        edges = f.readlines()
        for edge in edges:
            edge = edge.split()
            u, v, weight = edge[0], edge[1], float(edge[2])
            if u not in node_map:
                node_map[u] = G.addNode()
            if v not in node_map:
                node_map[v] = G.addNode()
            G.addEdge(node_map[u], node_map[v], weight)

        n, m = G.size
        node_list = np.empty(n, dtype=object)
        for name, index in node_map.items():
            node_list[index] = name

        return G, node_list, node_map

"""
Start with the identity for now.
"""
def denoise(g):
    return g
