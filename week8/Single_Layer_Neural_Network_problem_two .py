#!/usr/bin/env python
# coding: utf-8

# In[61]:


import os
import pandas as pd
import pandas.api.types as ptypes
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt


# In[62]:


# load dataset
df_original = pd.read_csv('Churn_Modelling.csv')
df = df_original


# In[63]:


# print top 5 row
df.head()


# In[64]:


# describe dataset
df.describe()


# In[65]:


print ('The train data has {0} rows and {1} columns'.format(df.shape[0],df.shape[1]))


# In[66]:


df.info()


# In[67]:


# check for col
print("\n Column name",df.columns)


# In[68]:


# check for datatype
print("\n",df.dtypes)


# In[69]:


# check for null value
print("\n\n",df.isnull().sum())


# In[70]:


# calculating  sum of null values
df.isna().sum()


# In[71]:


# check fo duplicate data
print("\n check for duplicate \n",df.duplicated().sum())
           


# In[72]:


# create dummy Variable for Purchased
df.replace(['yes','no'],[1,0],inplace=True)
df.head()


# In[73]:


# drop unwanted col
df= df.drop(['Surname'], axis=1)
df.dtypes


# In[74]:


# dummies for dataframe
df = pd.get_dummies(df)


# In[75]:


df.describe()


# In[76]:


# check for min values
df.min()


# In[77]:


# replace min values with mean
df.replace(0.0, df.mean(),inplace= True)


# In[78]:


# check for min values 
df.min()


# In[79]:


# check for correlation
df.corr()


# In[80]:


corr = df.corr()
sb.heatmap(corr)


# In[81]:


df.shape


# In[82]:


print (corr['Exited'].sort_values(ascending=False)[:15], '\n') #top 15 values
print ('----------------------')
print (corr['Exited'].sort_values(ascending=False)[-10:])


# In[83]:


# displot label
sb.distplot(df['Exited'])


# In[86]:


# drop unwanted col
df.drop(columns=["RowNumber", "CustomerId"], inplace = True)


# In[87]:


# feature scaling 
def feature_scaling(df):

    for name in df.columns:
        df[name] = (df[name] - df[name].min()) / (df[name].max()-df[name].min())
    print(df.head())
    return df

feature_scaling(df)


# In[88]:


# seperate data set
def features(df):
    output = df.Exited 
#     y.head()
    df = df.drop('Exited', axis = 1)
    print("Seperate Target column from dataset")
    return df, output
df, output = features(df)


# In[89]:


def split_data(df):
    # 70 % training datset
    train_per = int(0.70*len(df))
    print("Train dataset:", train_per)
    # 30% test dataset
    test_per = len(df)-train_per
    print("Test dataset:", test_per)
    
    print("\nConvert pandas datafrem into numpy")
    x_train_data = np.array(df[:train_per])   
    x_test_data  = np.array(df[:test_per])
    
    
    train_per_y = int(0.70*len(output))
    test_per_y = len(output)-train_per_y

    
    y_train_data = np.array(output[:train_per_y])
    y_test_data = np.array(output[:test_per_y])
    
   
    y_test_data = y_test_data.reshape(-1,1)
    y_train_data = y_train_data.reshape(-1,1)
        
    
    print("\nX train data shape:", x_train_data.shape)
    print("y train data shape:", y_train_data.shape)
    print("\nX test data shape:", x_test_data.shape)
    print("y test data shape:", y_test_data.shape)

    return x_train_data, y_train_data, x_test_data, y_test_data

x_train_data, y_train_data, x_test_data, y_test_data = split_data(df)        


# In[90]:


class Single_Layer_Neural_Network :
    def __init__(self):
    
        self.learning_rate = 0.001
        # Total iterations
        self.epoch = 600
        
    def gradient_descent(self,x_train_data, y_train_data):
        x_size = len(x_train_data)
        print("length of x_data",x_size)
        ncol = x_train_data.shape
        w = np.random.rand(x_train_data.shape[1], 1) * 0.01
        b = np.ones((1, 1), dtype='float') 
        
        for i in range(self.epoch):
            
            # calculate Z = W*X = (1 x m)
            z = np.dot(w.T, x_train_data.T )+ b
            
            # calculate sigmoid = 1/ (1 + np.exp(-z)) + b 
            sigmoid = 1/ (1 + np.exp(-z)) + b 
            
            
            dz = sigmoid - y_train_data.T
            
            # dw = 1/m np.sum(dz)
            dw =np.dot(x_train_data.T , dz.T) / x_size
            
            # db = 1/m np.sum(dz)
            db = (np.sum(dz)) / x_size

            # w = w - alpha dw
            w = w - np.dot(self.learning_rate,dw)
            # b = b - alpha db
            b = b - np.dot(self.learning_rate ,db)
        print("shape of z",z)
        print("shape of sigmoid",sigmoid.shape)
        print("shape of dz", dz.shape)
        print("shape of dw",dw.shape)
        print("shape of db ",db)
            
        return w, b
    

    
    def prediction(self,w,b,x_test_data):
       
        prediction = np.zeros((x_test_data.shape[0], 1), dtype=float)
        
        

        print("shape W ",w.shape)
        print("shape", x_test_data.shape)
        z = np.dot(w.T, x_test_data.T) + b # Z = W*X = (1 x m)
        sigmoid = 1 / 1 + np.exp(-z) # sigmoid = 1/ (1 + np.exp(-z)) + b 


        for i in (range(0, len(sigmoid))):
            if round(sigmoid[i][0], 2) <= 0.5:
                prediction[i][0] = 0
            else:
                prediction[i][0] = 1
               
        prediction = np.reshape(prediction, (len(prediction), 1))
        return prediction
    
    

    
def main(df, x_train_data, y_train_data, x_test_data,y_test_data ):
    
     # class obj created
    

    obj = Single_Layer_Neural_Network()
    
    x_train_data = np.column_stack((np.ones((x_train_data.shape[0], 1)), x_train_data))
    x_test_data = np.column_stack((np.ones((x_test_data.shape[0], 1)), x_test_data))
#     print("x train data .....", x_train_data.shape)

    
    w, b = obj.gradient_descent(x_train_data, y_train_data )
    
    prediction_test = obj.prediction(w,b,x_test_data)
    prediction_train = obj.prediction(w,b,x_train_data)
    
    
     # It calculates the accuracy

    train_accuracy = (100 - np.mean(np.abs(prediction_train - y_train_data)) * 100)
    test_accuracy = (100 - np.mean(np.abs(prediction_test - y_test_data)) * 100)

    print("train_accuracy = ", train_accuracy)
    print("test_accuracy = ", test_accuracy)
    
    
    
            
main(df, x_train_data,y_train_data,x_test_data,y_test_data)            
            
            
