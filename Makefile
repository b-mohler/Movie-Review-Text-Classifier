#Accessing and preprocessing the data
clean:
  python -B src/data_division.py
  python -B src/preprocessing.py

#Building frequency dictionary
counter: src/data_division.py, src/preprocessing.py
  python -B src/review_counter.py

#Training the model
train: src/data_division.py, src/preprocessing.py, src/review_counter.py
  python -B src/training.py

#Testing the model
test: src/data_division.py, src/preprocessing.py, src/review_counter.py, src/training.py
  python -B src/test.py

#Checking the accuracy of the model
acc: src/data_division.py, src/preprocessing.py, src/review_counter.py, src/training.py, src/test.py
  python -B src/accuracy.py

#Generating a confusion matrix for the model
conmax: src/data_division.py, src/preprocessing.py, src/review_counter.py, src/training.py, src/test.py, src/accuracy.py
  python -B src/confusion_matrix.py

#Allowing for user generated reviews to be entered and predicted
tryit: src/data_division.py, src/preprocessing.py, src/review_counter.py, src/training.py
  python -B src/predict.py
  python -B src/text_classifier.py
