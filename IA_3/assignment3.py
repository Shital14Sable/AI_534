# CS 534
# IA-3
# Cole Martin Jetton, Shital Dnyandeo Sable

import os
import re
import string
import numpy as np
import numpy.random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.feature_extraction.text import CountVectorizer
import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

class Supvecmac:
    def __init__(self, path_training, path_testing):
        """
        :param path_training: path for training file
        :param path_testing: path for testing file
        """
        self.path_training = path_training
        self.path_testing = path_testing

    def validitychecks(self):
        """
        :return: training and testing data as pandas dataframe
        """

        if not os.path.exists(self.path_training):
            print(f"Error at loading training data \n Invalid Path: {self.path_training}")
            exit(0)

        if not os.path.exists(self.path_training):
            print(f"Error at testing data \n Invalid Path: {self.path_testing}")
            exit(0)

        train_data = pd.read_csv(self.path_training)
        test_data = pd.read_csv(self.path_testing)

        return train_data, test_data

    def text_cleanup(self, text_str):
        """
        :param text_str: Input text string
        :return: Processed test string
        """
        stopwords = ['is', 'to', 'of', 'for', 'my', 'the']
        text_str = "".join([c for c in text_str if c not in string.punctuation])
        token_txt = re.split('\W+', text_str)
        ps = nltk.PorterStemmer()
        text_str = [ps.stem(word) for word in token_txt if word not in stopwords]
        return text_str


    def preprocessing_data(self):

        """
        :return: Top 10 words and the count in a dataframe by 2 functions: 'CountVectorizer' and 'TfidfVectorizer', format: dataframe
        """
        [train_data, test_data] = self.validitychecks()

        # CountVectorizer
        vect_elem_cv = CountVectorizer(analyzer=self.text_cleanup, lowercase=True)  # create an object
        feature_count_cv = vect_elem_cv.fit_transform(train_data['text'])
        # print(feature_count_cv.shape)
        feature_count_cv_df = pd.DataFrame(feature_count_cv.toarray(), columns=vect_elem_cv.get_feature_names())
        cv_sum = feature_count_cv_df.sum()
        cv_10 = cv_sum[cv_sum.sort_values(ascending=False).index[:10]]

        # TfidfVectorizer
        vect_elem_tf = TfidfVectorizer(analyzer=self.text_cleanup, lowercase=True)
        feature_count_tf = vect_elem_tf.fit_transform(train_data['text'])
        # print(feature_count_tf.shape)
        feature_count_tf_df = pd.DataFrame(feature_count_tf.toarray(), columns=vect_elem_tf.get_feature_names())
        tf_sum = feature_count_tf_df.sum()
        tf_10 = tf_sum[tf_sum.sort_values(ascending=False).index[:10]]


        return cv_10, tf_10


if __name__ == "__main__":
    out_val = Supvecmac("./IA3-train.csv", "./IA3-dev.csv")
    [x, y] = out_val.preprocessing_data()
    print(x)
    print(y)
