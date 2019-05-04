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

dataset = pd.read_csv("bike_sharing.csv")

dataset.head() 


# In[3]:


# check for describe
dataset.describe()


# In[4]:


dataset.columns


# In[5]:


# check for info
dataset.info()


# In[6]:


# check for num of rows and cols

print ('The train data has {0} rows and {1} columns'.format(dataset.shape[0],dataset.shape[1]))


# In[7]:


dataset = dataset.loc[:,['temp','cnt']]
dataset.dtypes


# In[8]:


# check for null 

dataset.isnull().sum()


# In[9]:


dataset.info()


# In[10]:


dataset.duplicated().sum()


# In[11]:


dataset.min()


# In[12]:


# Handle Missing data
def handle_min_values(dataset):
    # replace min values by mean
    dataset.replace(0, dataset.mean(), inplace=True)
    return dataset

dataset = handle_min_values(dataset)


# In[13]:


dataset.info()


# In[14]:


os.mkdir("CSV")


# In[15]:


def splitdata(dataset):
    # split train and test data
    train, test = train_test_split(dataset,test_size = 0.35, random_state=0)
    print("train : ", train.shape, " test : ", test.shape)
    #     print("y_train : ", y_train.shape, " y_test : ", y_test.shape)
    
    # saving datasets into csv files
    test.to_csv('CSV/test_file.csv',index=False,encoding='utf-8')
    
    
    # divide train data into train and cross validation 
    train_data, crossV_data = train_test_split(train,test_size = 0.40,random_state=0)
    
    train_data.to_csv('CSV/train_file.csv',index=False,encoding='utf-8')
    crossV_data.to_csv('CSV/crossV_file.csv',index=False,encoding='utf-8')
   
    
    
    print("train_data : ", train_data.shape, " crossV_data : ", crossV_data.shape)
    
    #     return x_train_data, x_cv, y_train_data, y_cv 

splitdata(dataset)


# In[16]:


dataset1 = pd.read_csv("CSV/train_file.csv")

dataset1.shape


# In[17]:


x =dataset1.iloc[:,:-1].values
y =dataset1.iloc[:,1].values


# In[18]:


print("x_train shape",x.shape)
print("y_train shape",y.shape)


# In[24]:


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
        
#         # Accuracy = r2_score(y,y_pred_train)*100
#         error = r2_score(y,y_pred_train)*100 # 15
#         Accuracy = (1 -error) * 100
#         print("Accuracy of r2_score : ", Accuracy)

        
#         Accuracy1 = explained_variance_score(y_pred_train,y)# -4
#         print("Accuracy of explained_variance_score :", Accuracy1)

#         Accuracy2 = explained_variance_score(y,y_pred_train) # 0
#         print("Accuracy of explained_variance_score : ", Accuracy2)
        
#         error = mean_absolute_error(y, y_pred_train) # 125
#         Accuracy3 = (1 -error) * 100
#         print("Accuracy of mean_absolute_error :", Accuracy3)
        
#         Accuracy4 = mean_squared_error(y, y_pred_train) # 27709.06811979052
#         print("Accuracy of mean_squared_error :", Accuracy4)
        
#         Accuracy5 = mean_squared_log_error(y, y_pred_train) # 2
#         print("Accuracy of mean_squared_log_error :", Accuracy5)



#         Accuracy6 = median_absolute_error(y, y_pred_train)
#         print("Accuracy of median_absolute_error : ", Accuracy6)
        # acc using r2
        error = r2_score(y, y_pred_train)      
        acc_r2 = (1-error)*100
       
        # using median_ab_error
        median_ab_error = median_absolute_error(y, y_pred_train)
        return acc_r2, median_ab_error

        


    
    # visualising the training set results
    
    def plotting(self,x, y,regressor):
        
        plt.scatter(x, y , color = 'pink')
        plt.plot(x, regressor.predict(x), color = 'red')
        plt.title("Salary Vs YearsExperience(Training set)")
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.show()
        
        
    # dump train model pickle file
    
    def pickel_file(self,regressor):
        
       
        file = open('trainmodel.pkl', 'wb')
        pickle.dump(regressor,file)
        file.close()
        
def main(dataset1,x,y):
    
    object_SimpleLR = Simple_Linear_regression()
    
    regressor = object_SimpleLR.model_train(x,y)
    print("Regressor : ", regressor)
    
    y_pred_train = object_SimpleLR.prediction(x,regressor)
    print("prediction of y train data : ",y_pred_train)
    
    acc_r2, median_ab_error = object_SimpleLR.accuracy(y,y_pred_train)
    print("\n Accuracy train data acc_r2 :\n ", acc_r2)
    print("\n Accuracy train data median_ab_error : \n",  median_ab_error)
    
    print()
    print("\n ------------visualising the training set results:----------- ")
    object_SimpleLR.plotting(x,y,regressor)
    
    object_SimpleLR.pickel_file(regressor)
    print("pickel file create")
    
    
    
    
    
main(dataset1,x,y)
    


# In[ ]:




