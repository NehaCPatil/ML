#!/usr/bin/env python
# coding: utf-8

# In[3]:


#importing the libraries

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt  
import pandas as pd

#imputer to handle missing data 

from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split

# handle categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder


#regression librarry

from sklearn.ensemble import RandomForestRegressor

#o check accuracy

from sklearn.metrics import accuracy_score

# to check accuracy

from sklearn.metrics import *

import pickle 
import os, sys
import csv

# ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[4]:


# load dataset
dataset_original = pd.read_csv ("Position_Salaries.csv")
dataset = dataset_original
dataset.head()


# In[5]:


print("Dataset has {} rows and {} Columns".format(dataset.shape[0],dataset.shape[1])) 


# In[6]:


dataset.sample()


# In[7]:


# check dataset information
dataset.info()


# In[8]:


dataset.describe().T


# In[9]:


# handling missing data if nessesary
"""
if missing values are present
imputer = Imputer(missing_values=0, axis=0)
imputer = imputer.fit(x_data[:, 3:16])
"""
dataset.isnull().sum()


# In[10]:


# check for minimum dataset
dataset.min()


# In[11]:


# # Handle Missing data
# def handle_min_values(dataset):
#     # replace min values by mean
#     dataset.replace(0, dataset.mean(), inplace=True)
#     return dataset

# dataset = handle_min_values(dataset)


# In[12]:


# #check dataset replace with mean or not
# dataset.min()


# In[16]:


# create directory to store csv files
os.mkdir("CSV_files")


# In[13]:


# # seperate fetures and label
# x_data = dataset.iloc[:, :-1].values
# y_data = dataset.iloc[:, 1].values


# In[51]:


# split dataset 

def splitdata(dataset):
    # split train and test data
    train, test = train_test_split(dataset,test_size = 0.30, random_state=0)
    print("train : ", train.shape, " test : ", test.shape)

    # saving datasets into csv files
    test.to_csv('CSV_files/test_file.csv',index=False,encoding='utf-8')

    # divide train data into train and cross validation 
    train_data, crossV_data = train_test_split(train,test_size = 0.30,random_state=0)
    
     #load data into csv for train and cross validation
    train_data.to_csv('CSV_files/train_file.csv',index=False,encoding='utf-8')
    crossV_data.to_csv('CSV_files/CValidation_file.csv',index=False,encoding='utf-8')
    
    print("train_data : ", train_data.shape, " crossV_data : ", crossV_data.shape)

splitdata(dataset)


# In[14]:


# # handle categorical data
# def handle_categorical_data(x_data):
#     #encode categorical data
#     label_encod = LabelEncoder()
#     x_data[:, 1] = label_encod.fit_transform(x_data[:, 1])
    
#     # one hot encoding
#     onehotencode = OneHotEncoder(categorical_features= [1])
#     x_data = onehotencode.fit_transform(x_data).toarray()
    
#     return x_data
    
# x_data = handle_categorical_data(x_data)


# In[52]:


# load dataset
train_dataset = pd.read_csv ("CSV_files/train_file.csv")
print("Dataset has {} rows and {} Columns".format(train_dataset.shape[0],train_dataset.shape[1])) 


# In[15]:


# #convert numpy.ndarray to DataFrame
# x_data = pd.DataFrame(x_data)
# x_data.shape


# In[53]:


train_dataset.head()


# In[62]:


class RandomForest():
    
    def create_module(self,x_train,y_train):
        
        # fitting simple LR to the training set
        
        regression = RandomForestRegressor(n_estimators= 300 ,random_state=0)
        regression.fit(x_train,y_train)
       
        return regression
          
        
    
    def y_prediction(self,x_train,regression):
        
        # predicting the test set result
        
        y_predict = regression.predict(x_train)
        print("y_predict value for 6.5 is ", regression.predict(np.array(6.5).reshape(-1,1)))
        return y_predict
    
    def accuracy(self,y_predict_train,y_train):
        # accuracy using r2 score
        acc_r2 = r2_score(y_train, y_predict_train)*100      
#         acc_r2 = (1-error)*100
  
        total_error = mean_absolute_error(y_train, y_predict_train)
        mean_ab=( 1- (total_error / len(y_train))) *100
        
        mean_sq  = mean_squared_error(y_train, y_predict_train) 

        mean_sq_log = mean_squared_log_error(y_train, y_predict_train)  
    
        median_ab_error = median_absolute_error(y_train, y_predict_train)
        
        return acc_r2,mean_ab,mean_sq,mean_sq_log, median_ab_error
    

    
    def visualization(self,x_train,y_train,regression):
        
        # Visualization the Decision Tree result (for higher resolution & smoother curve)
       
        x_grid = np.arange(min(x_train), max(x_train), 0.01)
        x_grid = x_grid.reshape((len(x_grid),1))
        
        plt.scatter(x_train,y_train, color = 'red')

        # reshape x_grid or not both will give same ploting
        
        plt.plot(x_grid,regression.predict(x_grid.reshape(-1,1)), color = 'blue')
        plt.title("Truth or bluff(Decision tree algorithm)")
        plt.xlabel("Position Level")
        plt.ylabel("Salary")
        plt.show()
        
    def create_piklefile(self,regression):
        
        # dump train model pickle file
        
        file = open('RandomForestModule.pkl', 'wb')
        pickle.dump(regression,file)
        file.close()
        

def main():
    #class obj created
    obj  = RandomForest()
    
    # seperate fetures and label
    # here we taking only 2 columns level and salary
    x_train = train_dataset.iloc[:,1:2].values
    y_train = train_dataset.iloc[:,2].values  
    

    print(x_train.shape, y_train.shape)
    regression = obj.create_module(x_train,y_train)
    
    y_train_pre = obj.y_prediction(x_train,regression)
    
    
    acc_r2,mean_ab,mean_sq,mean_sq_log, median_ab_error = obj.accuracy(y_train_pre,y_train)
    print("\n Accuracy train by acc_r2", acc_r2)
    
    obj.visualization(x_train,y_train, regression)
    
    obj.create_piklefile(regression)
    print("\nPikle file created")
    

if __name__ == '__main__':
    main()

