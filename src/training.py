import numpy as np
import data_division as dd
import preprocessing as pp

'''Function: train_naive_bayes
Parameters: freqs - dictionary where keys are word,label pairs and values are the corresponding number of occurrences
            train_x - a list of reviews
            train_y - a list of labels (0 for positive, 1 for negative) corresponding to those reviews
Outputs: logprior: the log prior (logarithmic form for the Naive Bayesian prior)
        loglikelihood: the log likelihood (logarithmic form for the Naive Bayesian likelihood
Both are probabilities, that will be used to help predict which sentiment a review should be classified as'''

def train_naive_bayes(freqs, train_x, train_y):

    loglikelihood = {}
    logprior = 0


    # calculate V, the number of unique words in the vocabulary
    vocab = set(pair[0] for pair in freqs.keys())
    V = len(vocab)

    # calculate num_pos and num_neg - the total number of positive and negative words for all documents
    num_pos = num_neg = 0
    for pair in freqs.keys():
        # if the label is positive (greater than zero)
        if pair[1]<1:

            # Increment the number of positive words by the count for this (word, label) pair
            num_pos += freqs[(pair)]

        # else, the label is negative
        else:

            # increment the number of negative words by the count for this (word,label) pair
            num_neg += freqs[(pair)]

    # Calculate num_doc, the number of documents
    num_doc = len(dd.y_train)

    # Calculate D_pos, the number of positive documents 
    pos_num_docs = train_y.value_counts()[0]

    # Calculate D_neg, the number of negative documents 
    neg_num_docs = train_y.value_counts()[1]

    # Calculate logprior
    logprior = np.log(pos_num_docs)-np.log(neg_num_docs)

    # For each word in the vocabulary...
    for word in vocab:
        # get the positive and negative frequency of the word
        freq_pos = pp.find_occurrence(freqs, word, 0)
        freq_neg = pp.find_occurrence(freqs, word, 1)

        # calculate the probability that each word is positive, and negative
        p_w_pos = (freq_pos + 1) / (num_pos + V)
        p_w_neg = (freq_neg + 1) / (num_neg + V)

        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos/p_w_neg)


    return logprior, loglikelihood
