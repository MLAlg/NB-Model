#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 06:29:32 2019

@author: samaneh
"""
#import dataset
# %pylab inline
import numpy as np
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')

print(news.keys())
print(type(news.data), type(news.target), type(news.target_names))
print(news.target_names)
print(len(news.data))
print(len(news.target))
print(news.data[0],"\n news_target ",  news.target[0], "\n news_target_name ", news.target_names[news.target[0]])

# split dataset
SPLIT_PERC = 0.75
split_size = int(len(news.data) * SPLIT_PERC)
x_train = news.data[:split_size]
x_test = news.data[split_size:]
y_train = news.target[:split_size]
y_test = news.target[split_size:]

# create classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline # to construct compound classifiers
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer,TfidfVectorizer

clf_1 = Pipeline([('vect', CountVectorizer()),('clf', MultinomialNB())])
clf_2 = Pipeline([
        ('vect', HashingVectorizer(norm=None)), # non_negative=True ?
        ('clf', MultinomialNB())
        ])
clf_3 = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB())])

#clf_4 = Pipeline([('vect', TfidfVectorizer(token_pattern=u"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b")), ('clf', MultinomialNB())]) # improve accuracy with this Vectorizer

# K-fold cross-validation and scores
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem # compute standard error

def evaluate_cross_validation(clf, x, y, k): # define a function to evaluate several classifiers
    cv = KFold(5, shuffle=True, random_state=0) # create a k-fold croos validation iterator of k=5 folds
    scores = cross_val_score(clf, x, y, cv=cv)
    print(scores)
    print("Mean score: {0:.3f} (+/- {1:.3f})".format(np.mean(scores), sem(scores)))

clfs = [clf_1, clf_3]
for clf in clfs:
    evaluate_cross_validation(clf, news.data, news.target, 5)
'''
# improve acuracy by removing useless words
# getting the file of stop_words
def get_stop_words():
    result = set()
    for line in open('stopwords_en.txt','r').readlines():
        result.add(line.strip())
    return result

clf_5 = Pipeline([('vect', TfidfVectorizer(stop_words=get_stop_words())), ('clf'), MultinomialNB])
evaluate_cross_validation(clf_5, news.data, news.target, 5)
'''
# improve accuracy by lower alpha parameter
clf_6 = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB(alpha=0.01))])
evaluate_cross_validation(clf_6, news.data, news.target, 5)

# evaluate the performance on the best classifier
from sklearn import metrics
def train_and_evaluate(clf, x_train, x_test, y_train, y_test):
    clf.fit(x_train, y_train) # train the classifier
    print("Accuracy on training set: ")
    print(clf.score(x_train, y_train))
    print("Accuracy on testing set: ")
    print(clf.score(x_test, y_test))
    y_pred = clf.predict(x_test)
    print("Classification report: ")
    print(metrics.classification_report(y_test, y_pred))
    print("Confusion Matrix: ")
    print(metrics.confusion_matrix(y_test, y_pred))
train_and_evaluate(clf_6, x_train, x_test, y_train, y_test)

# tokens used
print(len(clf_6.named_steps['vect'].get_feature_names()))
#feature names
clf_6.named_steps['vect'].get_feature_names()





































