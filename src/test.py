import numpy as np
import predict
'''Function: test_naive_bayes
Parameters: test_x - a list of reviews
            test_y - the corresponding labels for the list of reviews
            logprior - the logprior
            loglikelihood - a dictionary of the loglikelihoods for each word
Outputs: accuracy - the number of reviews classified correctly divided by the total number of reviews'''
def test_naive_bayes(test_x, test_y, logprior, loglikelihood):

    accuracy = 0  

    y_hats = []
    for review in test_x:
        # if the prediction is > 0
        if predict.naive_bayes_predict(review, logprior, loglikelihood) > 0:
            y_hat_i  = 1

        else:
            # otherwise the predicted class is 0
            y_hat_i = 0

        # append the predicted class to the list y_hats
        y_hats.append(y_hat_i)

    # error is the average of the absolute values of the differences between y_hats and test_y
    error_rate = np.mean(np.absolute([a-b for a, b in zip(y_hats, test_y)]))
    num_errors = (len(test_x))*error_rate

    #accuracy = (len(test_x))
    accuracy = (len(test_x)-num_errors)/len(test_x)

    #providing printed feedback on the results
    print("error is:",error_rate)
    print("number of errors is:", num_errors)
    print("number of reviews:",len(test_x))
    print("accuracy",accuracy)
    return accuracy
