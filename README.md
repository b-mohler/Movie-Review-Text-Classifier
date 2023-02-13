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

![Confusion Matrix](https://github.com/b-mohler/Movie-Review-Text-Classifier/blob/main/figs/conmax.png)

## Predicting Sentiment of New Reviews
Once the model has been trained and tested (and found to be reasonably accurate) new reviews which haven't yet been tagged as either positive or negative can be fed into text_classifier.py which (using the function from predict.py) will output the values (positive or negative) associated with each token in the review and a prediction as to whether the review should be classified as positive or negative. The program will continue asking for additional reviews to be input until the user terminates it by entering x.  

## Discussion of Results
The test function is found to have an accuracy exceeding 85% which is reasonably satisfactory but it is worth taking a look at where the misclassifications are coming from and how the model can be improved going forward. 

Having printed a list of tokens that appear in false negatives and false positives and sorted those tokens by how often they occur, some patterns begin to emerge that could contribute to the inaccurate prediction. First, for both the false negative reviews and the false positive reviews, the four most common words (though the ordering among these four differs between false positives and false negatives) are (with the stemming ignored for clarity), "movie", "film", "one", and "like". Since these words appear commonly across postive and negative reviews and none of them carry a particularly heavy weight as being either positive or negative ("movie" has a logprior value of -0.274537885142484, "film" has 0.108335405549288, "one" has 0.0220387487129296, and "like" has -0.197440371967943) these most common words aren't especially helpful for providing a definitive classification. Looking at a smattering of the actual falsely classified reviews, to get a more in depth sense of what is actually happenig contextually. For example, some of the false positives spent a great deal of time talking about positive things that were lacking from the movie in question (be that comparing it to other, better movies, talking about things they wanted the movie to do or ways it could have been improved, etc.) and all of those positive words were being counted towards a prediction of positive even though they weren't actually being used in reference to the movie being reviewed. As an example of one such false positive review:

"Sometimes laughter in the middle of a horror film is a signal of its greatness. I remember the nervous laughter from the audience in the re-release of The Excorcist really nervous laughter. It punctuated just how freaked out we all were watching the voice of Satan coming out of a 12 year old girl. In the case of the 2006 remake of the 1972 cult classic The Wicker Man however, it made me think that this new Wickerman is about as scary as the South Park character, Scuzzlebut, the friendly forest monster with TV's Patrick Duffy for a leg and a celery stalk for an arm who's favorite hobby is weaving wicker baskets. 3 years ago a friend of mine in Hollywood told me that he heard that Nicolas Cage was going to do a remake of the film. I started laughing and my friend (Keith) got mad at me touting Nicolas Cage as a great actor. I just didn't think that he could pull it off and unfortunately for moviegoers I was right. Gone is the realness, the outstanding original music, the originality, the creepiness and the wonderfully powerful dialogue. Instead we have horror movie cliches, affected acting and changes to the storyline that make any believability fall apart. Like many of the countless Hollywood remakes we have been inundated with lately this feels like we are watching 4th graders on a playground "playing Wickerman". The original film takes place on a remote Scottish Isle where a Scottish police officer is lured there to find a missing young girl named Rowan Morrison. In the new spin a California cop (Cage) is lured to an island of the coast of Washington state by his ex-girlfriend to find her missing daughter. She sends a photo and the missing daughter looks exactly like a young girl he tried to save in a fiery crash not long ago. The crash still haunts him in part because the girl's body was never found. Yet even after he gets a letter with her picture in it that connection is completely cast aside as he heads north, alone, to help his ex-girlfriend find her daughter. He arrives to find an island full of actors pretending to be the descendants of Wiccans, many of whom seem like they didn't get call backs for roles in The Village. And like The Village it isn't long before you realize there is nothing to be afraid of here. Not even the cloudy eyed blind sisters who speak in unison. I think that the opportunity in Hollywood to make great amounts of money on a film often comes at great expense to the artistry. I think someone like Nicolas Cage who is in so many films these days loses touch with the magic that film can be when it gets to the point where he has a personal chef on the set preparing his snacks. We needed a bad re-make of the Wickerman like we needed yet another '9-11' movie. I'm starting to wonder if Nicolas changed his surname from Coppola because he wanted to or because he was pleaded with to do so."

Above is the review in its uncleaned form, but even once it had been stemmed, stripped of punctuation and excess words, etc. for prediction purposes, positives like "greatness" (0.813614590909551 once stemmed to "great"), "classic" (0.645512802257026), "outstanding" (1.58236075640738, stemmed to "outstand"), "wonderfully" (0.576647762959286, stemmed to "wonder"), etc. would still remain (albeit in stemmed forms) even though all of these positive words are actually being used as comparisons to describe how negative the movie is.

Another potential factor in inaccurate classifications seems to be genre, with a cursory overview of the reviews themselves seeming to include a disproportionate number of horror movies (though given the lack of genre data included with the initial dataset, it is hard to be definitive about this). Consider for example the following false negative, which is illustrative of a number of others found in that category (again, before being cleaned):

"I caught this movie on the Sci-Fi channel recently. It actually turned out to be pretty decent as far as B-list horror/suspense films go. Two guys (one naive and one loud mouthed a**) take a road trip to stop a wedding but have the worst possible luck when a maniac in a freaky, make-shift tank/truck hybrid decides to play cat-and-mouse with them. Things are further complicated when they pick up a ridiculously whorish hitchhiker. What makes this film unique is that the combination of comedy and terror actually work in this movie, unlike so many others. The two guys are likable enough and there are some good chase/suspense scenes. Nice pacing and comic timing make this movie more than passable for the horror/slasher buff. Definitely worth checking out."

The viewer clearly enjoyed the movie but is honest about having done so more as a bit of stupid fun than a work of cinematic art, and it is easy to see how, even though the review is ultimately positive, terms like "B-list" (-0.521459736246108 stemmed to "b", and -0.242829497318239, stemmed to "list"), "worst" (-2.36141396089183), "maniac" (-0.522377931886275), "freaky" (-0.425783765300653, stemmed to "freak"), etc. slide it into being falsely classified as negative even though within the context of the review they are actually being used pretty positively.

Another problem seems to be the ambiguity of the initial classification. Some of the reviews are lukewarm and arguably could be classified as either positive, negative, or (perhaps more accurately), neither. Take for example:

"Simon Wests pg-13 thriller about a babysitter who gets disturbing prank calls while sitting at a mansion is neither original nor exciting enough to be called a good film. Although there are some elements of suspense, good eye candy and decent characters, the film is just another I know what you did last summer, as it falls short of being taken seriously. The performances were alright, but nothing special with this flick, i say skip it, unless you are looking for a mediocre movie, you can find better films than this on lifetime sometimes, okay maybe not lifetime but at least USA or somethin, haha....7/10"

This review is ultimately classified as positive in the original dataset and the 7/10 rating is, empirically, positive, but the words leading up to that final numeric rating are very lackluster meaning, while this is technically a false negative, it is difficult to equivocally call it a positive review even when that is how it is oficially classified.

While these are specific examples based on a cursory (and decidedly human) overview of the data they speak to the subjectivity of language and suggest areas where the current programming is falling short and ways in which it might potentially be improved in future.

Returning briefly to empirical CS for a moment, I wanted to check whether the length of a review had any relation to the likelihood of it being incorrectly classified, but a comparison of the range of lengths (in number of tokens) and average length for reviews in a given category yielded the following results:

False Neg Max: 622 False Neg Min: 16 False Neg Avg: 106.36605316973416

False Pos Max: 533 False Pos Min: 6 False Pos Avg: 134.57692307692307

True Neg Max: 556 True Neg Min:6 Ture Neg Avg: 116.99605451936873

True Pos Max: 585 True Pos Min: 14 True Pos Avg: 120.25743707093821

suggesting that review length is not a deciding factor in prediction accuracy (there are some differences but not significant ones, especailly in light of how comparatively small the falsely classified datasets are).

## Running the Code
To generate the confusion matrix run the following command:

```
make conmax
```

To run the text_classifier and get predictions for user input reviews, run the command:

``` 
make tryit
```

The rest of the code were used to build these two files, but if you want to pop the hood and see how everything works, each file is included in the src folder. Many of them include print statements which have been commented out but could be uncommented to provide further insight into how the code is wokring. If you opt to do this, just make sure that you are running any prerequisite files first. For example, data_division.py doesn't need any other files to have been run in order to work, but training.py will only work if data_division.py, preprocessing.py, and review_counter.py have already been run, in that order. The crucial ordering of the files is as follows (when there is a make command that gets you to that step of the process it has been included):
1. data_division.py
2. preprocessing.py (make clean)
3. review_counter.py (make counter)
4. training.py (make train)
5. test.py (make test)
6. accuracy.py (make acc)
7. confusion_matrix.py (make conmax)

The predict.py file is necessary for the text classifier (make tryit) but not for the confusion matrix (make conmax). If you want to run the predict file without taking the next step to the text classifier file the crucial order is: 

1. data_division.py
2. preprocessing.py
3. review_counter.py
4. training.py
5. predict.py
