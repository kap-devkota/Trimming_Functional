import numpy as np

def parse_go_label_file(fname):
    """Parses a GO label file.

    Outputs two dicts:
    - One associating GO labels to proteins
    - One associating proteins to GO labels
    """
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

def parse_graph_file(fname):
    """Parses a graph represented as an adjacency list. Works on either
    directed or undirected graphs.

    Outputs a triple:
    - An edgelist for the weighted graph G
    - A list mapping node indices to names
    - A dictionary mapping node names to node indices
    """
    with open(fname, "r") as f:
        edgelist, node_map = [], {}

        counter = 0
        edges = f.readlines()
        for edge in edges:
            edge = edge.split()
            u, v, weight = edge[0], edge[1], float(edge[2])

            for x in [u, v]:
                if x not in node_map:
                    node_map[x] = counter
                    counter += 1

            edgelist.append((node_map[u], node_map[v], weight))

        n, m = len(node_map), len(edgelist)
        node_list = np.empty(n, dtype=object)
        for name, index in node_map.items():
            node_list[index] = name

        return edgelist, node_list, node_map

def write_graph_to_file(filename, edgelist, node_map=None):
    """Given an output filename and a list of edges, writes the contents
    of the list to the file.
    """
    list_str = ""
    for edge in lst:
        p, q, wt = edge
        if node_map is not None:
            p = node_map[p]
            q = node_map[q]
            
        e_str = str(p) + " " + str(q) + " " + str(wt) + "\n"
        list_str += e_str

    list_str = list_str.lstrip().rstrip()
    
    with open(filename, "w") as fp:
        fp.write(list_str)

