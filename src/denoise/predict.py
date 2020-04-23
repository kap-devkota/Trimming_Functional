import itertools
import multiprocessing as mp
import numpy as np
import scipy.spatial.distance as spatial
from collections import defaultdict

def vote(voters, labels_f, weight_f):
    """
    Votes for the most popular label among the voters,
    weighted by their significance. 
    
    Input:
      - A list of voters.
      - A function mapping each voter to a list of labels.
      - A function mapping each voter to its weight.
    Output:
      - The most popoular label or none if no voters have labels.
    """
    label_counts = defaultdict(int)
    for voter in voters:
        for label in labels_f(voter):
            label_counts[label] += weight_f(voter)

    if not label_counts:
        return None

    return max(label_counts.keys(), key=lambda k: label_counts[k])

def wmv(A, labels_f, default_label="????"):
    """
    Weighted majority vote algorithm for an undirected graph.

    Input:
      - An adjacency matrix for a graph.
      - A function mapping node IDs to a list of labels. An
      empty list represents no known label.
      - A label to give when no label is predicted
    Output:
      - A dictionary mapping node IDs to a label. If the label
      is already known, the first label in the list is picked.
    """
    predicted_labels = {}

    n = A.shape[0]
    for i in range(n):
        labels = labels_f(i)
        if labels:
            predicted_labels[i] = labels[0]
            continue
        voters = filter(lambda j: A[i, j] != 0, itertools.chain(range(0, i), range(i + 1, n)))
        weight_f = lambda voter: A[node, voter]
        prediction = vote(voters, labels_f, weight_f)
        if prediction is not None:
            predicted_labels[i] = prediction
        else:
            predicted_labels[i] = default_label

    return predicted_labels

def knn(X, labels_f, k, metric="euclidean", default_label="????"):
    """Performs k-nearest neighors voting algorithm using the passed in
    distance metric.

    Input:
      - An embedding matrix where each row corresponds to point.
      - A function mapping node IDs to a list of labels. An
      empty list represents no known label.
      - A label to give when no label is predicted
    Output:
      - A dictionary mapping node IDs to a label. If the label
      is already known, the first label in the list is picked."""

    distances = spatial.squareform(spatial.pdist(X, metric=metric))
    predicted_labels = {}

    n = X.shape[0] # number of nodes
    for i in range(n):
        labels = labels_f(i)
        if labels:
            predicted_labels[i] = labels[0]
            continue

        voters = np.argsort(distances[i, :])[1:k+1]
        weight_f = lambda voter: 1 / distances[voter, i]
        prediction = vote(voters, labels_f, weight_f)
        if prediction is not None:
            predicted_labels[i] = prediction
        else:
            predicted_labels[i] = default_label

    return predicted_labels
           
