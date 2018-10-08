from gtnlplib.constants import OFFSET
from gtnlplib import clf_base, evaluation
import math

import numpy as np
from collections import defaultdict

# deliverable 3.1
def get_corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    corpus_counts = defaultdict(float)
    for i in range(len(y)):
        if y[i] == label:
            counter = x[i]
            for word in counter:
                corpus_counts[word] += counter[word]

    return corpus_counts

# deliverable 3.2
def estimate_pxy(x,y,label,smoothing,vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    '''
    pxy = defaultdict(float)
    corpus_counts = get_corpus_counts(x, y, label)
    total_words = 0
    for word in corpus_counts:
        total_words += corpus_counts[word]

    for word in vocab:
        pxy[word] = math.log((smoothing + corpus_counts[word]) / (len(vocab) * smoothing + total_words))

    return pxy

# deliverable 3.3
def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights 
    :rtype: defaultdict
    """
    
    labels = set(y)
    counts = defaultdict(float)
    doc_counts = defaultdict(float)
    num_instances = len(x)

    for genre in y:
        doc_counts[genre] += 1

    words = set()
    for counter in x:
        words.update(list(counter.keys()))

    vocab = list(words)

    for label in labels:
        pxy = estimate_pxy(x, y, label, smoothing, vocab)
        for word in pxy:
            counts[(label, word)] = pxy[word]

    for genre in doc_counts:
        counts[(genre, OFFSET)] = math.log(doc_counts[genre] / num_instances)

    return counts


# deliverable 3.4
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values
    :returns: best smoothing value
    :rtype: float

    '''
    accuracy = {}
    genres = set(y_dv)
    for smoother in smoothers:
        accuracy[smoother] = evaluation.acc(clf_base.predict_all(x_dv,
            estimate_nb(x_tr, y_tr, smoother), genres), y_dv)

    best_smoother = clf_base.argmax(accuracy)
    return best_smoother, accuracy






