#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import pandas.api.types as ptypes
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *
import pickle
import csv
from sklearn.metrics import accuracy_score


# In[2]:


# Import dataset

dataset1 = pd.read_csv("CSV/crossV_file.csv")
x =dataset1.iloc[:,:-1].values
y =dataset1.iloc[:,1].values
dataset1.head()


# In[3]:


# check for num of rows and cols

print ('The train data has {0} rows and {1} columns'.format(dataset1.shape[0],dataset1.shape[1]))


# In[4]:


#shape of x train and y train 

print("x train data shape",x.shape)
print("y train data shape",y.shape)


# In[5]:


# load pickle file

file = open('trainmodel.pkl', 'rb')
pickle_in = pickle.load(file)


# In[6]:


# dump train model pickle file
file = open('trainmodel.pkl', 'rb')
pickle_in = pickle.load(file)


# In[7]:


# predicting the test set result and train set result

y_pred_cv = pickle_in.predict(x)
print("pred_cv",y_pred_cv)


# In[8]:


# visualising the training set results

plt.scatter(x, y , color = 'blue')
plt.plot(x,y_pred_cv, color = 'red')
plt.title("Salary Vs YearsExperience(Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


# In[9]:


# Accuracy for train and test
# Accuracy = r2_score(y,y_pred_cv)*100
error = r2_score(y, y_pred_cv)      
acc_r2 = (1-error)*100
       
# using median_ab_error
median_ab_error = median_absolute_error(y, y_pred_cv)
print("Accuracy",acc_r2, median_ab_error)

