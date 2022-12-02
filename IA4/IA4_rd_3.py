# CS 534
# IA-4
# Cole Martin Jetton, Shital Dnyandeo Sable

from GloVe_Embedder import GloVe_Embedder
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import metrics
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
ax1.set_xlabel('Dimension 1')
ax1.set_ylabel('Dimension 2')

#3D Plot because why not. Took a few seconds and may help our grade 
fig1 = plt.figure()
ax1 = fig1.add_subplot(projection ='3d')
j=0
for i in range(0,5):
    j_n = j+29
    ax1.scatter(Xt[j:j_n,0],Xt[j:j_n,1],Xt[j:j_n,2],c=colors[i],label = word_list[i])
    j = j_n+1   
ax1.legend()
ax1.set_xlabel('Dimension 1')
ax1.set_ylabel('Dimension 2')
ax1.set_zlabel('Dimension 3')
#%% t-SNE Decomposition

#having this whole thing work as a loop to turn it into a single figure plot. 
#that'll make it easier to show everything at once
"""
fig1,ax1 = plt.subplots()
#Establishing the test problems
Xtsne = TSNE(n_components = 2,perplexity = 5).fit_transform(X)
ax1.scatter(Xtsne[:,0],Xtsne[:,1],c=data_set['Seed Color'])
"""
plex = np.array([[1,5],[10,25],[40,50],[75,100]])

fig, axs = plt.subplots(4, 2)

for i in range(0,4):
    for j in range(0,2):
        Xtsne = TSNE(n_components = 2,perplexity = plex[i,j]).fit_transform(X)
        axs[i,j].scatter(Xtsne[:,0],Xtsne[:,1],c=data_set['Seed Color'])
        title = 'Perplexity = '+ np.array2string(plex[i,j])
        axs[i,j].set_title(title,fontweight="bold")
        axs[i,j].set_xlabel('Dimension 1')
        axs[i,j].set_ylabel('Dimension 2')

fig.tight_layout(pad=2, w_pad=1, h_pad=0.2)


#%% k-means Clustering

k_v = range(2,20)
objective = []
purity = []
adj_rand = []
norm_mut_info = []

def purity_score(y_t,y_p):
    p = 0
    for i in range(0,4):
        #essentially match numbers and positions to see what happens
        #find position of particular category i
        #count if that position matches initial category position i
        p+=np.sum(np.multiply(1*(np.array(y_t)==i),1*(np.array(y_p)==i)))
    
    return p*1/len(y_t)

#setting the ground truth position
def assign_ground_truth(y_p,k):
    #take the position of the ground truth and use that as the category assigned by kmeans
    
    
    """
    #one way of doing it. Adds additional labels to other categories
    #requires an iff statement. It's the only way I could think to do it.
    if k == 2:
        y_t = [y_p[0]]*30 + [y_p[30]]*30 + [2]*30 + [3]*30 + [4]*30
    elif k == 3:
        y_t = [y_p[0]]*30 + [y_p[30]]*30 + [y_p[60]]*30 + [4]*30 + [5]*30
    elif k ==4:
        y_t = [y_p[0]]*30 + [y_p[30]]*30 + [y_p[60]]*30 + [y_p[90]]*30 + [6]*30
    else:
        y_t = [y_p[0]]*30 + [y_p[30]]*30 + [y_p[60]]*30 + [y_p[90]]*30 + [y_p[120]]*30
    """  
    y_t = [y_p[0]]*30 + [y_p[30]]*30 + [y_p[60]]*30 + [y_p[90]]*30 + [y_p[120]]*30
    
    return y_t

for i in range(0,len(k_v)):
    #objective creation and testing
    kmeans_i = KMeans(n_clusters = k_v[i],random_state=0).fit(X)#for reproduc
    objective.append(kmeans_i.inertia_)
    #score recording
    predict_labels = kmeans_i.labels_
    true_labels = assign_ground_truth(predict_labels,k_v[i])
    purity.append(purity_score(true_labels,predict_labels))
    adj_rand.append(adjusted_rand_score(true_labels,predict_labels))
    norm_mut_info.append(normalized_mutual_info_score(true_labels,predict_labels))
                    
    
plt.figure()
plt.plot(k_v,objective)
plt.xlabel('k value')
plt.xticks(k_v)
plt.ylabel('k means objective')

plt.figure()
plt.plot(k_v,purity)
plt.xlabel('k value')
plt.xticks(k_v)
plt.ylabel('Purity')

plt.figure()
plt.plot(k_v,adj_rand)
plt.xlabel('k value')
plt.xticks(k_v)
plt.ylabel('Adjusted Rand Index')

plt.figure()
plt.plot(k_v,norm_mut_info)
plt.xlabel('k value')
plt.xticks(k_v)
plt.ylabel('Normalized Mutual Information')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    