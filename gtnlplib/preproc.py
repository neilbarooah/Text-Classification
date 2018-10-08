from collections import Counter

import pandas as pd
import torch
import numpy as np
from scipy.misc import logsumexp
import matplotlib.pyplot as plt
from torch import optim

# deliverable 1.1
def bag_of_words(text):
    '''
    Count the number of word occurences for each document in the corpus

    :param text: a document, as a single string
    :returns: a Counter for a single document
    :rtype: Counter
    '''
    counter = Counter()
    words = text.split(" ")
    for word in words:
    	counter[word] += 1

    del counter[""]
    
    return counter

# deliverable 1.2
def aggregate_counts(bags_of_words):
    '''
    Aggregate word counts for individual documents into a single bag of words representation

    :param bags_of_words: a list of bags of words as Counters from the bag_of_words method
    :returns: an aggregated bag of words for the whole corpus
    :rtype: Counter
    '''

    counts = Counter()
    # YOUR CODE GOES HERE
    for counter in bags_of_words:
    	for word in counter:
    		counts[word] += counter[word]
    
    return counts

# deliverable 1.3
def compute_oov(bow1, bow2):
    '''
    Return a set of words that appears in bow1, but not bow2

    :param bow1: a bag of words
    :param bow2: a bag of words
    :returns: the set of words in bow1, but not in bow2
    :rtype: set
    '''
    return list(set(bow1) - set(bow2))

# deliverable 1.4
def prune_vocabulary(training_counts, target_data, min_counts):
    '''
    prune target_data to only words that appear at least min_counts times in training_counts

    :param training_counts: aggregated Counter for training data
    :param target_data: list of Counters containing dev bow's
    :returns: new list of Counters, with pruned vocabulary
    :returns: list of words in pruned vocabulary
    :rtype: list of Counters, set
    '''
    vocab = []
    for word in training_counts:
    	if training_counts[word] >= min_counts:
    		vocab.append(word)

    pruned_target = []
    for counter in target_data:
    	cnt = Counter()
    	for word in vocab:
    		if counter[word]:
    			cnt[word] = counter[word]
    	pruned_target.append(cnt)

    return pruned_target, vocab

# deliverable 5.1
def make_numpy(bags_of_words, vocab):
    '''
    Convert the bags of words into a 2D numpy array

    :param bags_of_words: list of Counters
    :param vocab: pruned vocabulary
    :returns: the bags of words as a matrix
    :rtype: numpy array
    '''
    vocab = sorted(vocab)
    matrix = np.zeros((len(bags_of_words), len(vocab)))

    for i in range(len(bags_of_words)):
    	bag_of_words = bags_of_words[i]
    	for j in range(len(vocab)):
    		word = vocab[j]
    		if word in bag_of_words:
    			matrix[i][j] = bag_of_words[word]

    return matrix


### helper code

def read_data(filename,label='Era',preprocessor=bag_of_words):
    df = pd.read_csv(filename)
    return df[label].values,[preprocessor(string) for string in df['Lyrics'].values]

def oov_rate(bow1,bow2):
    return len(compute_oov(bow1,bow2)) / len(bow1.keys())
