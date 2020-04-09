from collections import defaultdict

"""
Votes for the most popular label among the voters,
optionally weighted by their significance. Returns
none if no voters have labels.
"""
def vote(G, node, voters, labels_f):
    label_counts = defaultdict(int)
    for voter in voters:
        for label in labels_f(voter):
            label_counts[label] += G.weight(node, voter)

    if not label_counts:
        return None

    return max(label_counts.keys(), key=lambda k: label_counts[k])

"""
Weighted majority vote algorithm for an undirected graph.

Input:
  - An undirected graph G.
  - A function mapping node IDs to a list of labels. An
    empty list represents no known label.
  - A label to give when no label is predicted
Output:
  - A dictionary mapping node IDs to a label. If the label
    is already known, the first label in the list is picked.
"""
def wmv(G, labels_f, default_label="????"):
    predicted_labels = {}

    for node in G.iterNodes():
        labels = labels_f(node)
        if labels:
            predicted_labels[node] = labels[0]
            continue
        prediction = vote(G, node, G.neighbors(node), labels_f)
        if prediction is not None:
            predicted_labels[node] = prediction
        else:
            predicted_labels[node] = default_label

    return predicted_labels
