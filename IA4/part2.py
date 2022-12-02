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
import gensim.downloader as api
import string


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


def inexact_matches(ip_word, model):
    # Split the word
    len_word = len(ip_word)
    labels = []
    wordvecs = []

    for i in len(len_word):
        split1 = ip_word[:i]
        split2 = ip_word[i:]

        if model[split1] == 'True' and model[split2] == 'True':
            wordvecs.append(model[split1])
            labels.append(split1)
            wordvecs.append(model[split2])
            labels.append(split2)
            break
        elif model(split1) == 'False' and model(split2) == 'False':
            continue
        elif model(split1) == 'True' and model(split2) == 'False':
            wordvecs.append(model[split1])
            labels.append(split1)
            break
        elif model(split1) == 'False' and model(split2) == 'True':
            wordvecs.append(model[split2])
            labels.append(split2)
            break

    return labels, wordvecs


def tsne_plot(model,  input_data):
    labels = []
    wordvecs = []

    for i in range(input_data.shape[0]):
        tweet_i = input_data.text[i]
        tweet_i = text_cleanup(tweet_i)
        print(tweet_i)

        for word in tweet_i:
            if model[word] == 'True':
                labels, wordvecs = inexact_matches(word, model)
                print(labels)
                print(wordvecs)
            else:
                wordvecs.append(model[word])
                labels.append(word)

        tsne_model = TSNE(perplexity=3, n_components=2, init='pca', random_state=42)
        coordinates = tsne_model.fit_transform(wordvecs)

        x = []
        y = []
        for value in coordinates:
            x.append(value[0])
            y.append(value[1])

        plt.figure(figsize=(8, 8))
        for i in range(len(x)):
            plt.scatter(x[i], y[i])
            plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(2, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.show()

if __name__ == "__main__":
    train_data = pd.read_csv("./IA3-train.csv")
    test_data = pd.read_csv("./IA3-dev.csv")
    tsne_plot(ge, train_data)


