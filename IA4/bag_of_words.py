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

    # CountVectorizer
    vect_elem_cv = CountVectorizer()  # create an object
    train_words_X = vect_elem_cv.fit_transform(input_data['text'])
    train_words_Y = input_data['sentiment']
    test_X = vect_elem_cv.transform(test_data['text'])
    test_Y = test_data['sentiment']
    bag_of_words = vect_elem_cv.vocabulary_

    return train_words_X, train_words_Y, test_X, test_Y, bag_of_words


def rep_new_tweets(train_words_X, train_words_Y, test_X, test_Y):
    reg_model = LogisticRegression()
    reg_model.fit(train_words_X, train_words_Y)
    train_score = reg_model.score(train_words_X, train_words_Y)
    reg_model.fit(test_X, test_Y)
    test_score = reg_model.score(test_X, test_Y)
    return train_score, test_score


if __name__ == "__main__":
    train_data = pd.read_csv("./IA3-train.csv")
    test_data = pd.read_csv("./IA3-dev.csv")
    train_words_X, train_words_Y, test_X, test_Y, bag_of_words = create_BoW(train_data,test_data)
    print(rep_new_tweets(train_words_X, train_words_Y, test_X, test_Y,))


