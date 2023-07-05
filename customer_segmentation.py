#!/usr/bin/env python
# coding: utf-8

# # CUSTOMER SEGMENTATION
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore") 


# ## Data set

# In[2]:


df = pd.read_csv("Customer Data.csv")
df


# In[3]:


df.shape


# In[4]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


#filling NAN values
df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].mean())
df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].mean())


# In[9]:


df.isnull().sum()


# In[10]:


# duplicate rows in the dataset
df.duplicated().sum()


# In[11]:


# drop CUST_ID column because it is not used
df.drop(columns=["CUST_ID"],axis=1,inplace=True)


# In[12]:


df.columns


# In[13]:


plt.figure(figsize=(30,45))
for i, col in enumerate(df.columns):
    if df[col].dtype != 'object':
        ax = plt.subplot(9, 2, i+1)
        sns.kdeplot(df[col], ax=ax)
        plt.xlabel(col)
        
plt.show()


# In[14]:


plt.figure(figsize=(10,60))
for i in range(0,17):
    plt.subplot(17,1,i+1)
    sns.distplot(df[df.columns[i]],kde_kws={'color':'b','bw': 0.1,'lw':3,'label':'KDE'},hist_kws={'color':'g'})
    plt.title(df.columns[i])
plt.tight_layout()


# In[15]:


plt.figure(figsize=(12,12))
sns.heatmap(df.corr(), annot=True)
plt.show()


# ## Scaling the DataFrame

# In[16]:


scaled_df = scalar.fit_transform(df)


# In[17]:


pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_df)
pca_df = pd.DataFrame(data=principal_components ,columns=["PCA1","PCA2"])
pca_df


# ## Hyperparameter tuning

# In[18]:


#elbow method
inertia = []
range_val = range(1,15)
for i in range_val:
    kmean = KMeans(n_clusters=i)
    kmean.fit_predict(pd.DataFrame(scaled_df))
    inertia.append(kmean.inertia_)
plt.plot(range_val,inertia,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Inertia') 
plt.title('The Elbow Method using Inertia') 
plt.show()


# ## KMeans- Model building

# In[20]:


kmeans_model=KMeans(4)
kmeans_model.fit_predict(scaled_df)
pca_df_kmeans= pd.concat([pca_df,pd.DataFrame({'cluster':kmeans_model.labels_})],axis=1)


# ## clustering

# In[21]:


plt.figure(figsize=(8,8))
ax=sns.scatterplot(x="PCA1",y="PCA2",hue="cluster",data=pca_df_kmeans,palette=['red','green','blue','black'])
plt.title("Clustering using K-Means Algorithm")
plt.show()


# In[22]:


# find all cluster centers
cluster_centers = pd.DataFrame(data=kmeans_model.cluster_centers_,columns=[df.columns])
# inverse transform the data
cluster_centers = scalar.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data=cluster_centers,columns=[df.columns])
cluster_centers


# In[23]:


# Creating a target column "Cluster" for storing the cluster segment
cluster_df = pd.concat([df,pd.DataFrame({'Cluster':kmeans_model.labels_})],axis=1)
cluster_df


# In[24]:


cluster_1_df = cluster_df[cluster_df["Cluster"]==0]
cluster_1_df


# In[25]:


cluster_2_df = cluster_df[cluster_df["Cluster"]==1]
cluster_2_df


# In[26]:


cluster_3_df = cluster_df[cluster_df["Cluster"]==2]
cluster_3_df


# In[27]:


cluster_4_df = cluster_df[cluster_df["Cluster"] == 3]
cluster_4_df


# In[28]:


#Visualization
sns.countplot(x='Cluster', data=cluster_df)


# In[29]:


for c in cluster_df.drop(['Cluster'],axis=1):
    grid= sns.FacetGrid(cluster_df, col='Cluster')
    grid= grid.map(plt.hist, c)
plt.show()


# In[ ]:





# In[30]:


#Saving Scikitlearn models
import joblib
joblib.dump(kmeans_model, "kmeans_model.pkl")


# In[31]:


cluster_df.to_csv("Clustered_Customer_Data.csv")


# ## training and testing accuracy using decision tree

# In[32]:


#Split Dataset
X = cluster_df.drop(['Cluster'],axis=1)
y= cluster_df[['Cluster']]
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3)


# In[33]:


X_train


# In[34]:


X_test


# In[35]:


#Decision_Tree
model= DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[36]:


#Confusion_Matrix
print(metrics.confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[37]:


import pickle
filename = 'customer_segmentation_final_model.sav'
pickle.dump(model, open(filename, 'wb'))
 

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result,'% Acuuracy')


# In[ ]:




