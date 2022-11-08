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
from sklearn import svm
import time

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
        
        train_response = train_data['sentiment']

        # CountVectorizer
        vect_elem_cv = CountVectorizer(analyzer=self.text_cleanup, lowercase=True,min_df = 10)  # create an object
        feature_count_cv = vect_elem_cv.fit_transform(train_data['text'])
        # print(feature_count_cv.shape)
        feature_count_cv_df = pd.DataFrame(feature_count_cv.toarray(), columns=vect_elem_cv.get_feature_names())
        
        #split cv in positive and negative then count
        feature_count_cv_df_pos = feature_count_cv_df.drop(train_response[train_response==0].index) #positive (remove all zeros)
        feature_count_cv_df_neg = feature_count_cv_df.drop(train_response[train_response==1].index)#negative (remove all ones)
        
        #count
        cv_sum_p = feature_count_cv_df_pos.sum()
        cv_sum_n = feature_count_cv_df_neg.sum()
        cv_10_p = cv_sum_p[cv_sum_p.sort_values(ascending=False).index[:10]]
        cv_10_n = cv_sum_n[cv_sum_n.sort_values(ascending=False).index[:10]]

        # TfidfVectorizer
        vect_elem_tf = TfidfVectorizer(analyzer=self.text_cleanup, lowercase=True,min_df=10)
        feature_count_tf = vect_elem_tf.fit_transform(train_data['text'])
        # print(feature_count_tf.shape)
        feature_count_tf_df = pd.DataFrame(feature_count_tf.toarray(), columns=vect_elem_tf.get_feature_names())

        #split tf into positive and negative then coutn
        feature_count_tf_df_pos = feature_count_tf_df.drop(train_response[train_response==0].index) #positive (remove all zeros)
        feature_count_tf_df_neg = feature_count_tf_df.drop(train_response[train_response==1].index)#negative (remove all ones)
       
        #count       
        tf_sum_p = feature_count_tf_df_pos.sum()
        tf_sum_n =feature_count_tf_df_neg.sum()
        tf_10_p = tf_sum_p[tf_sum_p.sort_values(ascending=False).index[:10]]
        tf_10_n = tf_sum_n[tf_sum_n.sort_values(ascending=False).index[:10]]
        
        # TfidfVectorizer for Validation data
        feature_count_tf_validation = vect_elem_tf.fit_transform(test_data['text'])        
        feature_count_tf_df_validation = pd.DataFrame(feature_count_tf_validation.toarray(), columns=vect_elem_tf.get_feature_names())
        
        #remove all feature that are in the validation data but not in the training data 
        for i in list(feature_count_tf_df_validation):
            if i not in list(feature_count_tf_df):
                feature_count_tf_df_validation.drop([i],axis=1,inplace=True)

        #add blank features to the validation data based on features that are in the training data
        for i in list(feature_count_tf_df):
            if i not in list(feature_count_tf_df_validation):
                feature_count_tf_df_validation[i]=0
                
        #ensure that columns are in the same order (SVM gives issues if they're not)
        feature_count_tf_df_validation=feature_count_tf_df_validation[feature_count_tf_df.columns.tolist()]
        
        # Exporting Analyzable Data and Along with Responses
        preprocessed_training = format_data(feature_count_tf_df,train_response.to_numpy())
        preprocessed_validation = format_data(feature_count_tf_df_validation, test_data['sentiment'].to_numpy())
        

        return cv_10_p, cv_10_n, tf_10_p,tf_10_n, preprocessed_training, preprocessed_validation


class format_data:
    def __init__(self,features,responses):
        self.X = features
        self.Y = responses




class SVM_Analysis:
    #Takes in: (from format_data type of X and Y)
    #note: I don't actually know how to do object oriented 
    def __init__(self,training,validation,c_linear,c_quad,c_rbf):
        self.training = training
        self.validation = validation
        self.c_linear = c_linear
        self.c_quad = c_quad
        self.c_rbf = c_rbf
        

    #create definitions within here, one for linear svm, one for quadratic svm, and then one for rbf
        
    def linear_SVM(self):
        #create export data for graphs
        train_acc = []
        val_acc = []
        support_vecs = []
        
        #run through each value, fit, then record the accuracy
        for i in self.c_linear:
            print('Starting training with i =',i)     
            c_t = time.time()
            clf = svm.SVC(C = 10**(i),kernel = "linear")#, tol=0.1)
            clf.fit(self.training.X,self.training.Y)
            print('training complete, calculating accuracy...\n\n')
            train_acc.append(sum(clf.predict(self.training.X)== self.training.Y)/len(self.training.Y))
            val_acc.append(sum(clf.predict(self.validation.X)==self.validation.Y)/len(self.validation.Y))
            support_vecs.append(sum(clf.n_support_))
            print('Linear kernal with i =',i,'in',time.time()-c_t,'seconds. Complete, onto the next...\n\n')
        
        print('Done! :) \n\n\n')
        return train_acc,val_acc,support_vecs

"""    
class SVM_plots:
    def __init__(self,)
"""
#Part ?: c and gamma values to be tried (keeping these all the same fore now and need to modify stuff for adding gamma)
#NOTE: AFTER RUNNING FOR TESTING WE SHOULD RERUN IT
c_linear = [-4,-3,-2,-1,0,1,2,3,4]
c_quad =  [-4,-3,-2,-1,0,1,2,3,4]
c_rbf = [-4,-3,-2,-1,0,1,2,3,4]

#Part 0: Preprocessing
if __name__ == "__main__":
    #preprocessing
    out_val = Supvecmac("./IA3-train.csv", "./IA3-dev.csv")
    [top_count_p,top_count_n,top_tfidf_p,top_tfidf_n, training, validation] = out_val.preprocessing_data()
    
    print('Count Vectorizing\nPositive:\n',top_count_p,'\n\nNegative:\n',top_count_n,'\n\n\n')
    print('Count tfidf\nPositive:\n',top_tfidf_p,'\n\nNegative:\n',top_tfidf_n)
    
    #should be able to do the results as a single line.
    results = SVM_Analysis(training,validation,c_linear,c_quad,c_rbf)
    [Lin_train_acc,Lin_val_acc,Lin_SV_count] = results.linear_SVM()

    plt.figure(1);plt.plot(c_linear,Lin_train_acc,c_linear,Lin_val_acc);plt.legend(['Training','Validation'],loc='upper left')
    plt.figure(2);plt.plot(c_linear,Lin_SV_count)