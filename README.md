# Movie Review Sentiment Analysis Text Classifier
## Overview
The purpose of this project is to create a text classifier to sort movie reviews into positive and negative reviews using a naive Bayesian model. Using a set of reviews which have already been tagged as either positive or negative, words are assigned frequency values for how often they appear in either type of review. The model is then trained using these frequency values to generate a loglikelihood and logprior for each word. Once trained the model can be applied to reviews whose sentiment has not been categorized to predict whether they are positive or negative.

## Data
This project used a dataset of IMDB movie reviews evaluated (and made publicly available) by Stanford University's Maas, Daly, Pham, Huang, Ng, and Potts for their paper Learning Word Vectors for Sentiment Analysis (2011). The particular file used can now be found, in csv format, in the data folder of this repository as movie_reviews.csv. A second file, model_params.csv was generated as part of this project, containing the likelihoods (positive or negative) associated with each word as generated in training.py. The csv file was created using the model_params.py file. 

## Cleaning the Data
Before being fed into the model the data was cleaned up. In data_division.py it was first upsampled so that there would be the same number of positive and negative reviews (12474 of each) and then divided so that there would be 10,000 positive and 10,000 negative reviews in the training data and the remaining 2,472 positive and 2,472 negative reviews would be the test data. 

Once this was done, preprocessing.py cleaned each review, making all letters lowercase, eliminating elements like urls, line break markers, email addresses, punctuation, and stopwords. The data was then tokenized and stemmed to create clean, workable reviews. A frequency dictionary was then created with keys for each token and label (with the labels being 0 or 1, depending on whether the positive or negative instances of the token were being accessed). 

## Generating Frequency Values


## Training the Model

## Implementing the Predict Function

## Implementing the Test Function

## Evaluating Accuracy

## Discussion of Results

## Running the Code
