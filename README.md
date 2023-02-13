# Movie Review Sentiment Analysis Text Classifier
## Overview
The purpose of this project is to create a text classifier to sort movie reviews into positive and negative reviews using a naive Bayesian model. Using a set of reviews which have already been tagged as either positive or negative, words are assigned frequency values for how often they appear in either type of review. The model is then trained using these frequency values to generate a loglikelihood and logprior for each word. Once trained the model can be applied to reviews whose sentiment has not been categorized to predict whether they are positive or negative.

## Data
This project used a dataset of IMDB movie reviews evaluated (and made publicly available) by Stanford University's Maas, Daly, Pham, Huang, Ng, and Potts for their paper Learning Word Vectors for Sentiment Analysis (2011). The particular file used can now be found, in csv format, in the data folder of this repository as movie_reviews.csv. A second file, model_params.csv was generated as part of this project, containing the likelihoods (positive or negative) associated with each word as generated in training.py. The csv file was created using the model_params.py file. 

## Cleaning the Data
Before being fed into the model the data was cleaned up. In data_division.py it was first upsampled so that there would be the same number of positive and negative reviews (12474 of each) and then divided so that there would be 10,000 positive and 10,000 negative reviews in the training data and the remaining 2,472 positive and 2,472 negative reviews would be the test data. 

Once this was done, preprocessing.py cleaned each review, making all letters lowercase, eliminating elements like urls, line break markers, email addresses, punctuation, and stopwords. The data was then tokenized and stemmed to create clean, workable reviews. A frequency dictionary was then created with keys for each token and label (with the labels being 0 or 1, depending on whether the positive or negative instances of the token were being accessed). 

## Generating Frequency Values
Using review_counter.py the frequency dictionary created during preprocessing was populated with the frequencies with which each token/label pairing occurred in the training dataset, to then be used to train the model.

## Training the Model
The model is fed a list of reviews and their associated labels from the training dataset as well as the frequency dictionary in order to generate a logprior and loglikelihood in training.py

## Testing the Model
The logprior and loglikelihood from the training are taken and used to create a test function (test.py) which makes predictions about each review in the test data and then compares those predictions to the actual sentiment labels associated with those reviews to test the model. 

## Accuracy of the Model
The accuracy of the model is tested in accuracy.py by randomizing the data, splitting it into new test and training sets (with 1/4 of it going to test and 3/4 to train). Once this has been done, confusion_matrix.py uses it to create a confusion matrix of the rate of true negatives, false positives, false negatives, and true positives:

![Confusion Matrix]()

## Predicting Sentiment of New Reviews

## Discussion of Results

## Running the Code
