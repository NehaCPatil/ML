#!/usr/bin/env python
# coding: utf-8

# # 2. Apply K-Means clustering on below dataset

# In[13]:


# import libraries
# for basic operations
import numpy as np
import pandas as pd

# for data visualization
from matplotlib import pyplot as plt
import seaborn as sns
import plotly as py

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# for path
import os


# In[14]:


# reading the data and finding the computation time for uploading the dataset

get_ipython().run_line_magic('time', 'df = pd.read_csv(\'Dataset/USCensus1990.data.txt\',delimiter=",", sep=\'\\t\', iterator=True, chunksize=10000)')
dataset = pd.concat(df,ignore_index=True)

# checking the head of the data
dataset.head()


# In[15]:


# getting to know the shape of the data
print(dataset.shape)


# In[16]:


# describing the data

dataset.describe()


# In[17]:


# checking if the dataset contains any NULL values

dataset.isnull().sum()
# dataset.isnull().any()


# In[18]:


# check for duplicates
dataset.duplicated().sum()


# In[19]:


dataset.min()


# In[20]:


# take only 20000 rows in dataset

x = dataset.iloc[:20000].values


# # K-Means Clustering and using the elbow method
# 

# In[21]:


# The Elbow Method
# It is used to find the Optimal no. of Clusters in any Dataset

# applying kmeans clustering and using the elbow method

# the no. of clusters are where there deviations start decreasing

from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
  km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
  km.fit(x)
  wcss.append(km.inertia_)
  
plt.style.use('fivethirtyeight')
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()

