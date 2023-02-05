import pandas as pd
from sklearn.utils import resample
import os

#creating a dataframe from the movie review data
filename = "D:\data"
df = pd.read_csv(os.path.join(filename, "movie_reviews.csv.zip"))

#breaking up the dataframe into separate dfs for positive and negative
pos = df[df['sentiment'] == "positive"]
neg = df[df['sentiment'] == "negative"]

#assigning the review dataframes to the minority and majority class based on which there are more of, upsampling to account for this discrepancy
df_majority = pos
df_minority = neg

negative_upsample = resample(df_minority, replace = True, 
                        n_samples = df_majority.shape[0],
                        random_state = 101)

#concatenating the majority class data set and upsampled minority class data set
df_upsampled = pd.concat([pos, negative_upsample])  
df_upsampled = df_upsampled.sample(frac = 1)

# Dividing the data into training and testing points

# Considering 10000 positive and 10000 negative data points
negative_data_points_train = df_upsampled[df_upsampled["sentiment"]=="positive"].iloc[:10000]
positive_data_points_train = df_upsampled[df_upsampled["sentiment"]=="negative"].iloc[:10000]

# Considering the remaining data points for test
negative_data_points_test = df_upsampled[df_upsampled["sentiment"]=="positive"].iloc[10000:]
positive_data_points_test = df_upsampled[df_upsampled["sentiment"]=="negative"].iloc[10000:]

# Concatenate the training positive and negative reviews
X_train = pd.concat([negative_data_points_train["review"], positive_data_points_train["review"]])
# Concatenating the training positive and negative outputs
y_train = pd.concat([negative_data_points_train["sentiment"], positive_data_points_train["sentiment"]])

# Concatenating the test positive and negative reviews
X_test = pd.concat([negative_data_points_test["review"], positive_data_points_test["review"]])
# Concatenating the test positive and negative outputs
y_test = pd.concat([negative_data_points_test["sentiment"], positive_data_points_test["sentiment"]])

#making sure there are the correct number of positive and negative reviews (10000 of each) 
print("The number of training values is:\n", y_train.value_counts() + "\n The number of testing values is:", y_test.value_counts())
