from collections import defaultdict
from gtnlplib.clf_base import predict,make_feature_vector

# deliverable 4.1
def perceptron_update(x,y,weights,labels):
    '''
    compute the perceptron update for a single instance

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param weights: a weight vector, represented as a dict
    :param labels: set of possible labels
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    '''
    updated_weights = defaultdict(float)
    y_pred, _ = predict(x, weights, labels)
    fxy = make_feature_vector(x, y)
    fxy_pred = make_feature_vector(x, y_pred)

    wrong_predictions = set(fxy.keys()).symmetric_difference(set(fxy_pred.keys()))
    for prediction in wrong_predictions:
        if prediction in fxy:
            updated_weights[prediction] = fxy.get(prediction)
        else:
            updated_weights[prediction] = - fxy_pred.get(prediction)

    return updated_weights

# deliverable 4.2
def estimate_perceptron(x,y,N_its):
    '''
    estimate perceptron weights for N_its iterations over the dataset (x,y)

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    '''

    labels = set(y)
    weights = defaultdict(float)
    weight_history = []
    for it in range(N_its):
        for x_i,y_i in zip(x,y):
            updated_weight = perceptron_update(x_i, y_i, weights, labels)
            for label, word in list(updated_weight.keys()):
                weights[(label, word)] += updated_weight[(label, word)]

        weight_history.append(weights.copy())
    return weights, weight_history

