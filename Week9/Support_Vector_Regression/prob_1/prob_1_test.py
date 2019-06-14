#!/usr/bin/env python
# coding: utf-8

# In[37]:


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


# In[38]:


# splitting x y data

dataset1 = pd.read_csv("CSV_file/test_data.csv")
x =dataset1.iloc[:,:-1].values
y =dataset1.iloc[:,1].values
dataset1.head()


# In[39]:


# check for num of rows and cols

print ('The train data has {0} rows and {1} columns'.format(dataset1.shape[0],dataset1.shape[1]))


# In[40]:


#shape of x train and y train 

print("x train data shape",x.shape)
print("y train data shape",y.shape)


# In[41]:


# load pickle file

file = open('trainmodel.pkl', 'rb')
pickle_in = pickle.load(file)


# In[45]:


class Test_Support_Vector_Regression:
    # predicting the test set result and train set result

    def predict(self,x,pickle_in):
        y_pred = pickle_in.predict(x)
        print("pred_cv",y_pred)
        return y_pred
    
    # visualising the training set results
    
    def visualising(self,x,y,y_pred):
        print("\n visualising using SVR \n ")
        plt.scatter(x, y , color = 'pink')
        plt.plot(x, y_pred, color = 'red')
        plt.title("Truth or Bulff(SVR)")
        plt.xlabel("Position Level")
        plt.ylabel("Salary")
        plt.show()
        
        print("\n visualising Regression Model \n ")
        plt.scatter(x, y, color ='red')
        plt.plot(x, y_pred, color ='green')
        plt.title('Truth or Bluff (Reg Model)')
        plt.xlabel('Position level')
        plt.ylabel('Salary')
        plt.show()
    
    # Accuracy for train and test
    
    def Accuracy(self,y,y_pred):
        
        print("\n\n ACCURACY \n\n")
        Accuracy1 = explained_variance_score(y_pred,y) * 100# -4
        print("\n Accuracy of explained_variance_score :", Accuracy1)
       
        error = mean_absolute_error(y, y_pred) # 125
        Accuracy2 = (1 -error) * 100
        print("\n Accuracy of mean_absolute_error :", Accuracy2)
        
        return Accuracy1,Accuracy2
    
def main(dataset1,x,y):
    obj = Test_Support_Vector_Regression()
    
    y_pred = obj.predict(x,pickle_in)
    
    obj.visualising(x,y,y_pred)
    
    Accuracy1,Accuracy2 = obj.Accuracy(y,y_pred)
    
main(dataset1,x,y)
        
        
        


# In[ ]:





# In[ ]:




