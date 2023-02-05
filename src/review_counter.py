import preprocessing as clean
'''Function: review_counter
Parameters: output_occurrence - an empty dictionary to which each word label pair will be mapped to its frequency
            reviews - a list of reviews
            positive_or_negative - a list of the sentiments (positive or negative) assigned to each review
Outputs: a dictionary where the keys are word label pairs (where the label is the sentiment) and the values are the frequencies of those pairs
'''

def review_counter(output_occurrence, reviews, positive_or_negative):

    for label, review in zip(positive_or_negative, reviews):
        for token in clean.clean_review(review):
            key = (token,label)
            if key in output_occurrence:
                output_occurrence[key] += 1
            else:
                output_occurrence.update({key:1})
   
    return output_occurrence
