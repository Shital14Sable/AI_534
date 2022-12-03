# CS 534
# IA-4
# Cole Martin Jetton, Shital Dnyandeo Sable

from GloVe_Embedder import GloVe_Embedder
import os
import re
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
import gensim.downloader as api
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


ge = api.load("glove-twitter-200")
word_list = ['flight', 'good', 'terrible', 'help','late'] #also can remove for initial computation testing
colors = ['#AC92EB','#4FC1E8','#A0D567','#FFCE54','#ED5564'] #color scheme, https://www.pinterest.com/pin/594615957035870869/
#colors = ['r','g','b','m','y']
#colors = [1,2,3,4,10]


def text_cleanup(text_str):
    """
    :param text_str: Input text string
    :return: Processed test string
    """
    text_str = "".join([c for c in text_str if c not in string.punctuation])
    token_txt = re.split('\W+', text_str)
    return token_txt


def create_BoW(input_data, test_data):
    for i in range(input_data.shape[0]):
        tweet_i = input_data.text[i]
        tweet_i = text_cleanup(tweet_i)
        words_of_tweet = " "
        tweet_i = words_of_tweet.join(tweet_i)
        input_data.loc[i, 'text'] = tweet_i

    for j in range(test_data.shape[0]):
        tweet_j = test_data.text[j]
        tweet_j = text_cleanup(tweet_j)
        words_of_tweet_j = " "
        tweet_j = words_of_tweet_j.join(tweet_j)
        test_data.loc[j, 'text'] = tweet_j


    # CountVectorizer
    vect_elem_cv = CountVectorizer()  # create an object
    train_words_X = vect_elem_cv.fit_transform(input_data['text'])
    train_words_Y = input_data['sentiment']
    bag_of_words = pd.DataFrame(train_words_X.toarray(), columns=vect_elem_cv.get_feature_names())
    test_X = vect_elem_cv.transform(test_data['text'])
    test_Y = test_data['sentiment']

    return train_words_X, train_words_Y, test_X, test_Y, bag_of_words


def rep_new_tweets(train_words_X, train_words_Y, test_X, test_Y):
    reg_model = LogisticRegression()
    reg_model.fit(train_words_X, train_words_Y)
    train_score = reg_model.score(train_words_X, train_words_Y)
    test_score = reg_model.score(test_X, test_Y)
    return train_score, test_score


def nb_classifier(train_words_X, train_words_Y, test_X, test_Y):
    nb_ist = MultinomialNB()
    nb_ist.fit(train_words_X, train_words_Y)
    train_scor_nb = nb_ist.score(train_words_X, train_words_Y)
    test_score_nb = nb_ist.score(test_X, test_Y)
    return train_scor_nb, test_score_nb


def rf_classifier(train_words_X, train_words_Y, test_X, test_Y):
    rf_ist = RandomForestClassifier()
    rf_ist.fit(train_words_X, train_words_Y)
    train_scor_rf = rf_ist.score(train_words_X, train_words_Y)
    test_score_rf = rf_ist.score(test_X, test_Y)
    return train_scor_rf, test_score_rf


if __name__ == "__main__":
    train_data = pd.read_csv("./IA3-train.csv")
    test_data = pd.read_csv("./IA3-dev.csv")
    train_words_X, train_words_Y, test_X, test_Y, bag_of_words = create_BoW(train_data, test_data)
    print(rep_new_tweets(train_words_X, train_words_Y, test_X, test_Y))
    print(nb_classifier(train_words_X, train_words_Y, test_X, test_Y))
    print(rf_classifier(train_words_X, train_words_Y, test_X, test_Y))
