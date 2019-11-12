# Sentiment Analysis of Amazon Fine Food Reviews

In this project I worked with a dataset of ~500,000 reviews of fine foods from Amazon. More information about it can 
be found on Stanford's [website](http://snap.stanford.edu/data/web-FineFoods.html). 
Each product review contains a text summary, text body and a numerical rating 
of the product in terms of (1, 2, 3, 4 or 5) stars.

The task is to train a model which receives a text review as input and predicts whether it is positive or negative 
(for simplicity's sake, I treated reviews with 2 or more stars as being positive).

## Summary
- The solution uses the fastai library for training the model.
- The entire dataset is used. The classifier was trained with a 60/20/20 train/validation/test ratio.
- The **final model achieves an accuracy of 95%** on the held out test set (20% of the data).

## Data Exploration
- Unique products: 74258
- Unique users: 256047

#### Review Ratings
![](https://github.com/ddrevicky/amazon-fine-foods/blob/master/images/reviews.png)

Most reviews are positive - the dataset is unbalanced. It is also important to be aware that most people who write reviews are either very disappointed or very satisfied so there will likely be extremes of language in the text. This might be an issue when applying the model on more moderate reviews.

#### Word Count Per Review
![](https://github.com/ddrevicky/amazon-fine-foods/blob/master/images/word_count.png)

#### Reviews Per Product
![](https://github.com/ddrevicky/amazon-fine-foods/blob/master/images/product_review_count.png)
#### Reviews Per User
![](https://github.com/ddrevicky/amazon-fine-foods/blob/master/images/user_review_count.png)

#### Reviews Years
![](https://github.com/ddrevicky/amazon-fine-foods/blob/master/images/review_years.png)

Most reviews come from years 2008-2011. This might be relevant when using and evaluating our model. 
Although it is sensible to assume that the English vocabulary has not changed much for example between 1999 and 2011 
and between 2011 and 2019 (if we were to deploy it on reviews written today), some new words might have entered the 
language that were not in usage then. But in our case this is just something to be aware of, the time span is not
so large to make an important difference.

## Language Model
The language model will form a key part of the sentiment classfier we will train. Its task is more or less
to learn to predict the next word of a sentence based on the words it has seen previously (the context of the word).
It is based on the AWD-LSTM architecture and is pretrained on the subset of English Wikipedia 
(the WikiText-103 corpus). We then fine-tune it on our corpus of ~500k reviews. The language model is able to predict
the next word of the sentence 37% of the time (as evaluated on a subset of the data reserved for testing its accuracy).

Here is an example review that the model generates:
````
I liked this  pasta The box was not as pictured , but it was not packaged in a box . i did
not want to hear that item had the packing problem . But , i have no clue what happened . 
The cookies are a good quality product and would definitely recommend them . xxbos Great
product , lousy service i love the flavor of this tea . i use it in my chai tea and it 
gives it a hint of sweetness . i love the fact that it 's organic and makes it
easy to drink.
````
It's not exactly sane but not completely out of place either. Note that there are some special tokens (`xxbos`) generated 
by the model, these are more of a technical detail and not a part of the review.

## Sentiment Classifier

This is the model we are really interested in. It receives a text review on input and predicts whether it 
is positive or negative. It will use the language model as its body and we will put a classifier head on top. As the
review passes through the language model it will produce a matrix of features. Based on these features, the head
will learn to classify the sentiment of the review.

Training is done using progressive unfreezing where we first freeze all the layers except the head and train just that.
We then progressively unfreeze earlier layers. This is a more robust approach than unfreezing all of the layers at the 
beginning since the random parameters of the newly attached head require different learning rates compared with the 
already finely tuned parameters of the body (language model).

## Conclusion:
The model achieves an accuracy of ~95% on the test set. Some suggestions for further improvement:

- Retrain on all of the data (including test and validation) before final deployment
- Train the language model and classifier for longer since they were not overfitting yet
- Tune hyperparameters
- Fine-tune the language model on all Amazon reviews irrespective of the product category (not just for fine foods). 
It is very likely that a positive/negative review for an electronic appliance for example will contain very
similar words as a review for food [(available here)](http://jmcauley.ucsd.edu/data/amazon/)
- Train a regression model classifying the sentiment in the range 1-5. I made a decision to label the ratings 3,4,5 as
positive and 1,2 as negative but some users might consider a rating of 3 already to be a negative one
- Apply some forms of data augmentation to the text
