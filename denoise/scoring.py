import random

def score_cv(test_nodes, test_labelling, real_labelling):
    """Scores cross validation by counting the number of test nodes that
    were accurately labeled after their removal from the true
    labelling.
    """
    correct = 0
    total = 0
    for node in test_nodes:
        if node not in test_labelling:
            continue

        test_label = test_labelling[node]
        if test_label in real_labelling[node]:
            correct += 1
        total += 1

    return float(correct) / float(total)

def kfoldcv(k, labels, prediction_algorithm, randomized=False):
    """Performs k-fold cross validation.

    Args:
      - A number of folds k
      - A labeling for the nodes.
      - An algorithm that takes the training labels
      and outputs a predicted labelling.

    Output: 
      - A list where each element is the accuracy of the
      learning algorithm holding out one fold of the data.
    """
    nodes = list(labels.keys())
    if randomized:
        random.shuffle(nodes)

    accuracies = []
    for i in range(0, k):
        inc = int(len(nodes) / k)
        x = inc * i
        y = inc * (i + 1)
        if i + 1 == k:
            y = len(nodes)

        training_nodes = nodes[:x] + nodes[y:]
        training_labels = {n: labels[n] for n in training_nodes}
        test_nodes = nodes[x:y]

        test_labelling = prediction_algorithm(training_labels)
        accuracy = score_cv(test_nodes, test_labelling, labels)
        accuracies.append(accuracy)

    return accuracies
