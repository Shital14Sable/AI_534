# CS 534
# IA-4
# Cole Martin Jetton, Shital Dnyandeo Sable

from GloVe_Embedder import GloVe_Embedder
import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

word_list = ['flight', 'good', 'terrible', 'help','late'] #also can remove for initial computation testing
colors = ['#AC92EB','#4FC1E8','#A0D567','#FFCE54','#ED5564'] #color scheme, https://www.pinterest.com/pin/594615957035870869/
#colors = ['r','g','b','m','y']
#colors = [1,2,3,4,10]
ge = GloVe_Embedder(os.getcwd()+'\\'+'GloVe_Embedder_data.txt')


#%%

#Build each list dataframe and save for the report and make sure to add to eachother

data_set = pd.DataFrame(columns = ['Word','Distance','Seed Color','Seed Class']) #create empty df to append to 
for i in range(0, len(word_list)):
    word_nearest = ge.find_k_nearest(word_list[i], 30) #need 30 since 29 ends up including the original
    current_df = pd.DataFrame(word_nearest, columns = ['Word','Distance'])
    current_df['Seed Color'] = [colors[i]]*30
    current_df['Seed Class'] = [word_list[i]]*30
    current_df.to_csv(os.getcwd()+'\\'+word_list[i]+'.csv')
    data_set = pd.concat([data_set,current_df])

full_list = list(data_set['Word'])    #I actually overcomplicated this and didn't need to do it this way
#%% PCA Decomposition

#Take word list the embedder to generate the X for the PCA Decomposition
X = ge.embed_list(full_list) 
pca = PCA()
Xt = pca.fit_transform(X)


#%% PCA Visualization

#2D Plot
fig1,ax1 = plt.subplots()
j = 0
for i in range(0,5): #this is how I got it to work with labeling stuff.
    j_n = j+29
    ax1.scatter(Xt[j:j_n,0],Xt[j:j_n,1],c=colors[i],label = word_list[i])
    j = j_n+1
ax1.legend()

#3D Plot because why not. Took a few seconds and may help our grade 
fig1 = plt.figure()
ax1 = fig1.add_subplot(projection ='3d')
j=0
for i in range(0,5):
    j_n = j+29
    ax1.scatter(Xt[j:j_n,0],Xt[j:j_n,1],Xt[j:j_n,2],c=colors[i],label = word_list[i])
    j = j_n+1   
ax1.legend()
