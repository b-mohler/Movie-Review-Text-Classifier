import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import data_division as dd
'''Function: clean_review
Parameters: review (the review to be cleaned)
Output: the review once it has been stripped of websites, line break markers, email addresses, punctuation and stopwords, as well as being tokenized and those tokens stemmed,
the ultimate format will be a list of tokens'''

def clean_review(review):
    review = review.lower() #changes all capital letters to lower case
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
    return review_cleaned

#testing on a randomly selected review from the training dataset by printing the initial review as well as the cleaned result
test_review = dd.X_train.iloc[10403]
print(test_review)
clean_review(test_review)
