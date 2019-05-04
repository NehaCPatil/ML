#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures

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


# In[2]:


# load dataset
dataset_original = pd.read_csv ("Position_Salaries.csv")
dataset = dataset_original
dataset.head()


# In[3]:


print("Dataset has {} rows and {} Columns".format(dataset.shape[0],dataset.shape[1])) 


# In[4]:


dataset.sample()


# In[5]:


# check dataset information
dataset.info()


# In[6]:


dataset.describe().T


# In[7]:


# handling missing data if nessesary
dataset.isnull().sum()


# In[8]:


# check for minimum dataset
dataset.min()


# In[9]:


# checks for duplicate values
dataset.duplicated().sum()


# In[10]:


dataset = dataset[['Level','Salary']]


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


# In[13]:


# # seperate fetures and label
# x_data = dataset.iloc[:, :-1].values
# y_data = dataset.iloc[:, 1].values


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


# In[15]:


# #convert numpy.ndarray to DataFrame
# x_data = pd.DataFrame(x_data)
# x_data.shape


# In[16]:


# create directory to store csv files
os.mkdir("CSV_files")


# In[17]:


# split dataset 

def splitdata(dataset):
    # split train and test data
    train, test = train_test_split(dataset,test_size = 0.20, random_state=0)
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


# In[18]:


# load dataset
train_dataset = pd.read_csv ("CSV_files/train_file.csv")
print("Dataset has {} rows and {} Columns".format(train_dataset.shape[0],train_dataset.shape[1])) 


# In[19]:


train_dataset.head()


# In[20]:


# seperate fetures and label
# here we taking only 2 columns level and salary
x_train = train_dataset.iloc[:,:-1].values
y_train = train_dataset.iloc[:,1].values  
    


# In[21]:


# # fitting simple linear regression model to the training dataset
lin_reg = LinearRegression(normalize=True)  
lin_reg.fit( x_train, y_train)  

# fitting polynomial regression model to the training dataset
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x_train)
# fit into multiple Linear regression model
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y_train)


# In[22]:


class Polynomial_Reg():
    
    def reference_module(self,x_train):
        # fitting polynomial regression model to the training dataset
        poly_reg = PolynomialFeatures(degree=5)
        x_poly = poly_reg.fit_transform(x_train)
        return poly_reg, x_poly
      

    def create_module(self,x_train,y_train, x_poly):
        # fit into multiple Linear regression model
        lin_reg2 = LinearRegression()
        lin_reg2.fit(x_poly,y_train)
        return lin_reg2          
        
    
    def y_prediction(self,x_train,lin_reg2,poly_reg):
        # predicting the train set result
        y_pred_train = lin_reg2.predict(poly_reg.fit_transform(x_train))
        return y_pred_train
    
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
    

    
    def visualization(self,x_train,y_train,poly_reg, lin_reg2):
        # Visualization the Decision Tree result (for higher resolution & smoother curve)
        x_grid=np.arange(min(x_train),max(x_train),0.1)
        x_grid=x_grid.reshape((len(x_grid),1))
        
        plt.scatter(x_train,y_train,color='red')
        plt.plot(x_grid,lin_reg2.predict(poly_reg.fit_transform(x_grid)),color='blue')
        plt.title('predict salary  based on position (Training Set)')
        plt.xlabel('Level')
        plt.ylabel('Salary')
        plt.show()
        
    def create_piklefile(self,poly_reg, lin_reg2):
        fileObject = open("train_data.pkl",'wb')       
        # dump train model pickle file
        file = open('Polynomial_RegModule.pkl', 'wb')
        pickle.dump(poly_reg,file)
        pickle.dump(lin_reg2,file)
        # here we close the fileObject
        file.close()
        

def main():
    #class obj created
    obj  = Polynomial_Reg()
    
    
    poly_reg, x_poly = obj.reference_module(x_train)
    
    lin_reg2 = obj.create_module(x_train,y_train, x_poly)
  
    y_train_pre = obj.y_prediction(x_train,lin_reg2, poly_reg)

    acc_r2,mean_ab,mean_sq,mean_sq_log, median_ab_error = obj.accuracy(y_train_pre,y_train)
    print("\n Accuracy train by acc_r2", acc_r2)
    print("\n Accuracy train by mean_ab", mean_ab)
    print("\n Accuracy train by mean_sq", mean_sq)
    print("\n Accuracy train by mean_sq_log", mean_sq_log)
    print("\n Accuracy train by median_ab_error", median_ab_error)
    
    
    
    obj.visualization(x_train,y_train, poly_reg, lin_reg2)
    
    obj.create_piklefile(poly_reg, lin_reg2)


if __name__ == '__main__':
    main()

