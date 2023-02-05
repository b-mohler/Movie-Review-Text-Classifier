import preprocessing as pp
'''Function: naive_bayes_predict
Parameters: review - a string (of a review)
        logprior - a probability
        loglikelihood - a dictionary of words mapping to probabilities
Outputs: total_prob - the sum of all the likelihoods of each word in the review (if the word is present in the loglikelihood dict) added to logprior'''
def naive_bayes_predict(review, logprior, loglikelihood):
    
      # process the review to get a list of words
    word_l = pp.clean_review(review)

    # initialize probability to zero
    total_prob = 0

    # add the logprior
    total_prob += logprior
    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            total_prob += loglikelihood[word]
            #print("new total_prob is:",total_prob, "word was:", word)
    if total_prob > 0:
        total_prob = 0
    else:
        total_prob = 1
    return total_prob
