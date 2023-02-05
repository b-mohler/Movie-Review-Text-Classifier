import accuracy as acc
import predict
import preprocessing as pp
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.ioff()

#renaming random_y_test for clarity's sake (random_y_test is the accurate list of sentiment labels)
actual_results = acc.split_data.random_y_test

#sorting into two lists, one for the true positives, one for true negatives
actual_pos = actual_results.tolist().count(0)
actual_neg = actual_results.tolist().count(1)

#empty list for the predicted labels
predicted_results = []

#for each review in random_X_test calculate the predicted sentiment and add to the predicted_results list
for review in acc.split_data.random_X_test:
    prediction = predict.naive_bayes_predict(review, acc.logprior, acc.loglikelihood)
    predicted_results.append(prediction)

#empty lists for indices of the false positives, false negatives, true positives and true negatives 

false_neg_index = []
false_pos_index = []
true_pos_index = []
true_neg_index = []

#set initial index to -1 so when we add 1 to each index as the first part of our loop we actually start with 0 as we want
index = -1
#for the corresponding values in actual results and predicted results
#if the actual result is positive and the prediction is negative add the index to false_neg_index
#if the actual result is negative and the prediction is positive add the index to false_pos_index
#if the actual result is positive and the prediction is positive add the index to true_pos_index
#if the actual result is negative and the prediction is negative add the index to true_neg_index
#increase the index and repeat until done
for a, b in zip(actual_results, predicted_results):
    index += 1
    if a==0 and b==1:   
        false_neg_index.append(index)
    elif a==1 and b==0:
        false_pos_index.append(index)
    elif a==0 and b==0:
        true_pos_index.append(index)
    elif a==1 and b==1:
        true_neg_index.append(index)

#empty lists for the actual false negative and false positive reviews 
#(prior to being cleanned so they can be read and interpreted on a human level for context to better understand why they were misclassified)
false_neg_reviews_unclean =[]
false_pos_reviews_unclean =[]

#empty lists for the false negative reviews, false positive reviews, true negative reviews, and true positive reviews
#(after being cleaned so each review will be a tokenized list)
false_neg_reviews = []
false_pos_reviews = []
true_neg_reviews = []
true_pos_reviews = []

#using each index add the corresponding review to the appropriate reviews list
#false predictions are added both in cleaned form to _reviews and uncleaned form _reviews_unclean
#I want to actually be able to read the unaltered false reviews to get a sense of where the ambiguity is that led to them being misclassified
#with the correctly classified reviews there is no need
for index in false_neg_index:
    false_neg_reviews_unclean.append(acc.split_data.random_X_test[index])
    false_neg_reviews.append(pp.clean_review(acc.split_data.random_X_test[index]))
for index in false_pos_index:
    false_pos_reviews_unclean.append(acc.split_data.random_X_test[index])
    false_pos_reviews.append(pp.clean_review(acc.split_data.random_X_test[index]))
for index in true_neg_index:
    true_neg_reviews.append(pp.clean_review(acc.split_data.random_X_test[index]))
for index in true_pos_index:
    true_pos_reviews.append(pp.clean_review(acc.split_data.random_X_test[index]))

#the number of false positive and false negatives
num_false_neg = len(false_neg_reviews)
num_false_pos = len(false_pos_reviews)

#the rate of false positives and false negatives
rate_false_neg = num_false_neg/len(acc.split_data.random_X_test)
rate_false_pos = num_false_pos/len(acc.split_data.random_X_test)

#the percent of false positives and false negatives in relation to the total
percent_false_neg = round((rate_false_neg * 100),2)
percent_false_pos = round((rate_false_pos * 100),2)

#the number of true negatives and true positives
num_true_neg = actual_neg-len(false_neg_reviews)
num_true_pos = actual_pos-len(false_pos_reviews)

#the rate of true negatives and true positives
rate_true_neg = num_true_neg/len(acc.split_data.random_X_test)
rate_true_pos = num_true_pos/len(acc.split_data.random_X_test)

#the percent of true negatives and true positives in relation to the total
percent_true_neg = round((rate_true_neg * 100),2)
percent_true_pos = round((rate_true_pos * 100),2)

#creating a visualization of the confusion matrix using seaborn
matrix_array = np.array([[rate_true_neg, rate_false_pos], [rate_false_neg, rate_true_pos]])
#printing the basic array
print(matrix_array)
#classing it up a bit
cm = sns.heatmap(matrix_array, cmap="icefire", fmt=".2%", annot=True)
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()
