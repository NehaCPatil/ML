
#!/usr/bin/env python
# coding: utf-8

# ##  2.Apply Hierarchical clustering on below dataset
# 

# In[ ]:


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


# In[2]:


# reading the data and finding the computation time for uploading the dataset

get_ipython().run_line_magic('time', 'df = pd.read_csv(\'Dataset/USCensus1990.data.txt\',delimiter=",", sep=\'\\t\', iterator=True, chunksize=10000)')
dataset = pd.concat(df,ignore_index=True)

# checking the head of the data
dataset.head()


# In[3]:


# getting to know the shape of the data
print(dataset.shape)


# In[4]:


# describing the data

dataset.describe()


# In[5]:


# checking if the dataset contains any NULL values

dataset.isnull().sum()
# dataset.isnull().any()


# In[6]:


# check for duplicates
dataset.duplicated().sum()


# In[7]:


dataset.min()


# In[12]:


x = dataset.iloc[:10000,:].values


# # hierarchial clustering using dendrograms

# In[13]:


# hierarchial clustering using dendrograms

import scipy.cluster.hierarchy as sch
plt.style.use('fivethirtyeight')
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))

plt.title('Dendrogam', fontsize = 20)
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

