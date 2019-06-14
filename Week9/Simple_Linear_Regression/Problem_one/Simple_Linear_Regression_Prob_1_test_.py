#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

import import_ipynb
import function

from sklearn.metrics import accuracy_score


# In[4]:


# splitting x y data

dataset1 = pd.read_csv("CSV_file/test_data.csv")
x =dataset1.iloc[:,:-1].values
y =dataset1.iloc[:,1].values
dataset1.head()


# In[6]:


# check for num of rows and cols

print ('The train data has {0} rows and {1} columns'.format(dataset1.shape[0],dataset1.shape[1]))


# In[24]:


#shape of x train and y train 

print("x train data shape",x.shape)
print("y train data shape",y.shape)


# In[25]:


# load pickle file

file = open('trainmodel.pkl', 'rb')
pickle_in = pickle.load(file)


# In[31]:


class Test_Simple_Lr:
    # predicting the test set result and train set result

    def predict(self,x):
        y_pred_cv = pickle_in.predict(x)
        print("pred_cv",y_pred_cv)
        return y_pred_cv
    
    # visualising the training set results
    
    def visualising(self,x, y):
        plt.scatter(x, y , color = 'blue')
        plt.plot(x, pickle_in.predict(x), color = 'red')
        plt.title("Salary Vs YearsExperience(Training set)")
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.show()
    
    # Accuracy for train and test
    
    def Accuracy(self,y,y_pred_cv):
        
        Accuracy = r2_score(y_pred_cv,y)*100
        print("Accuracy Train",Accuracy)
        return Accuracy
    
def main(dataset1,x,y):
    obj = Test_Simple_Lr()
    
    obj.predict(x)
    
    obj.visualising(x, y)
    
    obj.Accuracy(y,y_pred_cv)
    
main(dataset1,x,y)
        
        
        

