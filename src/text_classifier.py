'''The purpose of this program is to modularize the sentiment analysis classifier so that a user can input reviews which will
be cleaned and then processed, providing an output of the probabilities for each token of the review as well as the ultimate classification. The program
will continue to ask for and process reviews until the user inputs x to quit the program.'''

#importing all the packages involved
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

#loading in the frequency dictionary containing the words and their log likelihoods from the training model found in NLPAssignment1Final.ipynb
def load_model_params(file = "data/model_params.csv"):
    loglikelihood = pd.read_csv(file)
    return loglikelihood
load_model_params()

#cleaning the reviews
def clean_review(user_input):
    review = user_input.lower() #changes all capital letters to lower case
    review = re.sub(r"<br ?/>", " ", review) # removes <br>s
    review = re.sub(r"http\S+", " ", review) # replaces URLs starting with http 
    review = re.sub(r"www.\S+", " ", review) # replaces URLs starting with www
    review = re.sub(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", "",review) # replaces email addresses
    review = re.sub(r'[\'!@#?()*+_::<=>[\]^~{|}~"&,.-]', ' ', review) #removes the majority of punctuation outright
    review = re.sub(r'\/[^0-9/0-9]', ' ', review) #removes slashes unless directly preceding digits (to preserve fractions, like 4/10 stars)
    stopword = nltk.corpus.stopwords.words('english') #defines stopwords
    review = " ".join([word for word in re.split(' ', review) #removes those stopwords (modified to remove at a break ' ' insted of a word /W+ to allow for the retention of ' as explained above)
            if word not in stopword])
    review = word_tokenize(review)
    stemmer = PorterStemmer()
    review_cleaned = [stemmer.stem(word) for word in review]
    #print(review_cleaned)
    return review_cleaned

#the prediction function to add up the probabilites of each word and classify the review accordingly
def naive_bayes_predict(review, logprior, loglikelihood):
    loglikelihood = pd.DataFrame(loglikelihood)
      # process the review to get a list of words
    word_l = clean_review(review)

    # initialize probability to zero
    total_prob = 0

    # add the logprior
    total_prob += logprior
    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if (loglikelihood['Words'].eq(word)).any():
            #float(loglikelihood.loc[loglikelihood['Words'] == word, 'Likelihoods'])
            # add the log likelihood of that word to the probability
            word_prob = float(loglikelihood.loc[loglikelihood['Words'] == word, 'Likelihoods'])
            print("The probability of",word,"is:",word_prob)
            total_prob += word_prob
            #print(total_prob)
    if total_prob < 0:
        total_prob = 1
        print("The predicted classification is: Negative")
    else:
        total_prob = 0
        print("The predicted classification is: Positive")
    return total_prob
logprior = 0.0

#Asking for initial input
review = input("Please enter your review to be classified:                    (To quit, enter x)")

#if that input was x, quit immediately. Otherwise, run the prediction function on the review and ask for a new input, continuing until the input is x
while review != "x":
    naive_bayes_predict(review, logprior, load_model_params())
    review = input("Please enter your review to be classified:                    (To quit, enter x)")
