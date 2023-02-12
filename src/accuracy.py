import random
import data_division as dd
import preprocessing as pp
import pandas as pd
import test
import training
'''Function: split_data
Parameters: seed - a randomizing seed to allow the data to be randomized in the same way repeatedly
Outputs: split_data.random_X_test - randomized test data consisting of reviews
        split_data.random_X_train - randomized train data consisting of reviews
        split_data.random_y_test - randomized test data consisting of sentiment labels that correspond to the reviews in random_X_test
        split_data.random_y_train - randomized train data consisting of sentiment labels that correspond to the reviews in random_y_test'''

def split_data(seed):
    #setting data up to be randomized (need to be lists for random shuffle)
    X = dd.df_upsampled["review"]
    X = X.tolist()
    y = dd.df_upsampled['sentiment']
    y = y.map(pp.output_map)
    y = y.tolist()

    #randomizing the data
    random.Random(seed).shuffle(X)
    random.Random(seed).shuffle(y)

    #splitting the data so 1/4 of it goes to test and 3/4 to train
    #reorganizing the data as series instead of lists
    split_data.random_X_test = X[:len(X)//4]
    split_data.random_X_test = pd.Series(split_data.random_X_test)
    split_data.random_y_test = y[:len(y)//4]
    split_data.random_y_test = pd.Series(split_data.random_y_test)
    split_data.random_X_train = X[len(X)//4:]
    split_data.random_X_train = pd.Series(split_data.random_X_train)
    split_data.random_y_train = y[len(X)//4:]
    split_data.random_y_train = pd.Series(split_data.random_y_train)

    return split_data.random_X_test, split_data.random_X_train, split_data.random_y_test, split_data.random_y_train
split_data(5)