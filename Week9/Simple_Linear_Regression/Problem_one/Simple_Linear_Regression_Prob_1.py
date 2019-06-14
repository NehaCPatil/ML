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

dataset = pd.read_csv("Salary_Data.csv")
x =dataset.iloc[:,:-1].values
y =dataset.iloc[:,1].values
dataset.head() 


# In[3]:


# check for describe
dataset.describe()


# In[4]:


# check for info
dataset.info()


# In[5]:


# check for num of rows and cols

print ('The train data has {0} rows and {1} columns'.format(dataset.shape[0],dataset.shape[1]))


# In[6]:


# check for null 

dataset.isnull()


# In[9]:


os.mkdir("CSV_file")
def csv_file(x_train_data,y_train_data,file_name):
    myData = x_train_data
    myFile = open('CSV_file/'+file_name, 'w')  
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(myData)
    # colomn
    colnames=['x'] 
    df = pd.read_csv('CSV_file/'+file_name, names=colnames, header=None)
    # inserting column with static value in data frame 
    df.insert(1, "y", y_train_data)
   
    df.to_csv('CSV_file/'+file_name, index =  False)
    


# In[10]:


def splitdata(x, y):
    
    x_train_data,x_test_data,y_train_data,y_test_data = train_test_split(x,y,test_size = 1/3, random_state=0)
    
    csv_file(x_test_data,y_test_data,'test_data.csv')

    x_train_data, x_cv_data, y_train_data, y_cv_data = train_test_split(x_train_data,y_train_data,test_size = 0.40,random_state=0)
    
    csv_file(x_train_data,y_train_data,'train_data.csv')
    csv_file(x_cv_data,y_cv_data,'cv_data.csv') 
#     return x_train_data,x_test_data,y_train_data,y_test_data,x_cv_data,y_cv_data
splitdata(x, y)


# In[11]:


dataset = pd.read_csv("CSV_file/train_data.csv")


# In[12]:


print("x_train shape",x.shape)
print("y_train shape",y.shape)


# In[15]:


class Simple_Linear_regression:
    
    # fitting sample linear regresion to the training set
    
    def model_train(self,x,y):  
        regressor = LinearRegression()
        regressor.fit(x,y)
        return regressor
    
    
    # predicting the test set result and train set result
    
    def prediction(self,x,regressor):
    
        y_pred_train = regressor.predict(x)
        return y_pred_train

    # Accuracy for train and test
    
    def accuracy(self,y,y_pred_train):
        
        # Accuracy = r2_score(y,y_pred_train)*100
        Accuracy = r2_score(y_pred_train,y)*100
        return Accuracy
    
    # visualising the training set results
    
    def plotting(self,x, y,y_pred_train):
        
        plt.scatter(x, y , color = 'blue')
        plt.plot(x, y_pred_train, color = 'red')
        plt.title("Salary Vs YearsExperience(Training set)")
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.show()
        
    # dump train model pickle file
    
    def pickel_file(self,regressor):
        
       
        file = open('trainmodel.pkl', 'wb')
        pickle.dump(regressor,file)
        file.close()
    
        
def main(dataset,x,y):
    
    object_SimpleLR = Simple_Linear_regression()
    
    regressor = object_SimpleLR.model_train(x,y)
#     print("Regressor : ", regressor)
        
    
    y_pred_train = object_SimpleLR.prediction(x,regressor)
#     print("prediction of y train data : ",y_pred_train)
    
    Accuracy = object_SimpleLR.accuracy(y,y_pred_train)
    print("Accuracy train data : ", Accuracy)
    
    
    object_SimpleLR.plotting(x,y,y_pred_train)
    
    object_SimpleLR.pickel_file(regressor)
    print("Pickle file create")
    
    
    
    
    
main(dataset,x,y)
    


# In[ ]:




