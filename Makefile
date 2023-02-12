#Accessing and cleaning data
clean:
	python -B src/data_division.py
	python -B src/preprocessing.py

# Building the frequency dictionary
counter: clean
	python -B src/review_counter.py
	
# Training the model
train: clean counter
	python -B src/training.py
	
# Testing the model
test: clean counter train
	python -B src/test.py
  
# Checking the accuracy of the model
acc: clean counter train test
	python -B src/accuracy.py
  
# Generating a confusion matrix for the model
conmax: 
	python -B src/data_division.py
	python -B src/preprocessing.py
	python -B src/training.py
	python -B src/test.py
	python -B src/accuracy.py
	python -B src/confusion_matrix.py
  
# Generating a confusion matrix for the model
tryit: clean counter train
	python -B src/predict.py
	python -B src/text_classifier.py
