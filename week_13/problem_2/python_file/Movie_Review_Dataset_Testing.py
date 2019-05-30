#!/usr/bin/env python
# coding: utf-8

# #  Large Movie Review Dataset
# 
# We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. 
# There is additional unlabeled data for use as well. 
# Raw text and already processed bag of words formats are provided
# 

# In[1]:


import nltk
import random
import pickle
from nltk.corpus import movie_reviews
import os
import pandas as pd


# ### Load model

# In[6]:


# open a file, where you stored the pickled testing_data
file = open('testing.pickel', 'rb')

# dump information to that file
data = pickle.load(file)


# In[3]:


# open a file, where you stored the pickled naivebayes data

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()


# In[4]:


print("Classifier accuracy percent testing_set:",(nltk.classify.accuracy(classifier, data))*100)


# In[5]:


classifier.show_most_informative_features(15)

