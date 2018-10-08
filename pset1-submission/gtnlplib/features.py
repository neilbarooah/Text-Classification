from gtnlplib.constants import OFFSET
import numpy as np
import torch
import math

# deliverable 6.1
def get_top_features_for_label_numpy(weights,label,k=5):
    '''
    Return the five features with the highest weight for a given label.

    :param weights: the weight dictionary
    :param label: the label you are interested in 
    :returns: list of tuples of features and weights
    :rtype: list
    '''
    top_features = []
    for weight in list(weights.keys()):
        if weight[0] == label:
            top_features.append((weight, weights[weight]))

    top_features_sorted = sorted(top_features, key=lambda x : x[1], reverse=True)
    return top_features_sorted[:k]


# deliverable 6.2
def get_top_features_for_label_torch(model,vocab,label_set,label,k=5):
    '''
    Return the five words with the highest weight for a given label.

    :param model: PyTorch model
    :param vocab: vocabulary used when features were converted
    :param label_set: set of ordered labels
    :param label: the label you are interested in 
    :returns: list of words
    :rtype: list
    '''
    vocab = sorted(vocab)
    weights = []
    for name, param in model.state_dict().items():
        if name == "Linear.weight":
            weights = param.numpy()

    # get weights for the desired label
    label_index = 0
    for i in range(len(label_set)):
        if label_set[i] == label:
            label_index = i

    label_weights = weights[label_index]

    # gets the indices of the top k weights
    # negate weights to obtain them in descending order for argsort
    top_weights = (-label_weights).argsort()[:k]
    return [vocab[i] for i in top_weights]


# deliverable 7.1
def get_token_type_ratio(counts):
    '''
    compute the ratio of tokens to types

    :param counts: bag of words feature for a song, as a numpy array
    :returns: ratio of tokens to types
    :rtype: float

    '''
    length_of_song = 0
    distinct_words = 0
    for count in counts:
        if count > 0:
            length_of_song += count
            distinct_words += 1
    
    if distinct_words > 0 and length_of_song > 0:
        return length_of_song / distinct_words
    else:
        return 0


# deliverable 7.2
def concat_ttr_binned_features(data):
    '''
    Discretize your token-type ratio feature into bins.
    Then concatenate your result to the variable data

    :param data: Bag of words features (e.g. X_tr)
    :returns: Concatenated feature array [Nx(V+7)]
    :rtype: numpy array

    '''
    binned_feature_matrix = np.zeros(shape=(len(data), len(data[0]) + 7))
    for i in range(len(data)):
        token_type_ratio = get_token_type_ratio(data[i])
        if token_type_ratio is None:
            continue
        else:
            token_type_ratio = math.floor(token_type_ratio)
            if (token_type_ratio >= 6):
                token_type_ratio = 6
            binned_feature = np.zeros(7)
            binned_feature[token_type_ratio] = 1
            binned_feature_matrix[i] = np.concatenate([data[i], binned_feature])

    return binned_feature_matrix
