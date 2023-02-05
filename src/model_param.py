#creating a csv file of the loglikelihoods for each word to be used with text_classifier.py)
words = list(loglikelihood.keys())
likelihoods = list(loglikelihood.values())
df = pd.DataFrame(words, columns=['Words'])
df["Likelihoods"] = likelihoods
df.to_csv("model_params.csv")
