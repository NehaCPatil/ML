#!/usr/bin/env python
# coding: utf-8

# # XGBOOST
# 
# For a given dataset predict whether customer will exit (Output variable “Exited”) the bank or not using artificial neural network

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


# In[2]:


# import dataset

file =open("test_data/test.pickle",'wb')

# split dataset in features and label

X_test=pickle.load(X_test,file)
y_test=pickle.load(y_test,file)


# In[10]:


# load model

file =open("XGBoost_Model/Model.pickle",'rb')
classifier = pickle.load(classifier,file)


# In[11]:


# Encoding categorical data

labelencoder_X_1 = LabelEncoder()
X_test[:, 1] = labelencoder_X_1.fit_transform(X_test[:, 1])
labelencoder_X_2 = LabelEncoder()
X_test[:, 2] = labelencoder_X_2.fit_transform(X_test[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_test = onehotencoder.fit_transform(X_test).toarray()
X_test = X_test[:, 1:]


# In[15]:


# Predicting the Test set results

y_pred = classifier.predict(X_test)


# In[16]:


# Making the Confusion Matrix

cm = confusion_matrix(y_test, y_pred)


# In[17]:


# Applying k-Fold Cross Validation

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy",accuracies)


# In[18]:


print("accuracies mean",accuracies.mean())
print("accuracies std",accuracies.std())

