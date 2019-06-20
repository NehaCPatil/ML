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
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


# In[2]:


# Importing the dataset

dataset = pd.read_csv('Dataset/Churn_Modelling.csv')


# In[3]:


# information of dataset

dataset.info()


# In[4]:


# description of dataset 

dataset.describe().T


# In[5]:


# check for null

dataset.isnull().sum()


# In[6]:


# check for duplicates

dataset.duplicated().sum()


# In[7]:


# check for min

dataset.min()


# In[8]:


# check for datatype

dataset.dtypes


# In[9]:


dataset.head()


# In[10]:


# split dataset in features and label
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# In[11]:


# Encoding categorical data

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


# In[12]:


# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


X_train,X_cv,y_train,y_cv=train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)


# In[13]:


# create pickle file for test data

file =open("test_data/test.pickle",'wb')
pickle.dump(X_test,file)
pickle.dump(y_test,file)
file.close()


# In[14]:


# Fitting XGBoost to the Training set

classifier = XGBClassifier()
classifier=classifier.fit(X_train, y_train)


# In[15]:


# Predicting the Test set results

y_pred = classifier.predict(X_cv)


# In[16]:


# Making the Confusion Matrix

cm = confusion_matrix(y_cv, y_pred)


# In[17]:


# Applying k-Fold Cross Validation

accuracies = cross_val_score(estimator = classifier, X = X_cv, y = y_cv, cv = 10)
print("Accuracy",accuracies)


# In[18]:


print("accuracies mean",accuracies.mean())
print("accuracies std",accuracies.std())


# In[19]:


# save model in pickle file

file =open("XGBoost_Model/Model.pickle",'wb')
pickle.dump(classifier,file)
file.close()
print("Pickle file create: Model.pickle")

