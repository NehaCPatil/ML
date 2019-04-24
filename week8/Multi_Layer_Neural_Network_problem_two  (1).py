#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import pandas.api.types as ptypes
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt


# In[2]:


# load dataset
df_original = pd.read_csv('Churn_Modelling.csv')
df = df_original


# In[3]:


# print top 5 row
df.head()


# In[4]:


# describe dataset
df.describe()


# In[5]:


print ('The train data has {0} rows and {1} columns'.format(df.shape[0],df.shape[1]))


# In[6]:


df.info()


# In[7]:


# check for col
print("\n Column name",df.columns)


# In[8]:


# check for datatype
print("\n",df.dtypes)


# In[9]:


# check for null value
print("\n\n",df.isnull().sum())


# In[10]:


# calculating  sum of null values
df.isna().sum()


# In[11]:


# check fo duplicate data
print("\n check for duplicate \n",df.duplicated().sum())
           


# In[12]:


# create dummy Variable for Purchased
df.replace(['yes','no'],[1,0],inplace=True)
df.head()


# In[13]:


# drop unwanted col
df= df.drop(['Surname'], axis=1)
df.dtypes


# In[14]:


# dummies for dataframe
df = pd.get_dummies(df)


# In[15]:


df.describe()


# In[16]:


# check for min values
df.min()


# In[17]:


# replace min values with mean
df.replace(0.0, df.mean(),inplace= True)


# In[18]:


# check for min values 
df.min()


# In[19]:


# check for correlation
df.corr()


# In[20]:


corr = df.corr()
sb.heatmap(corr)


# In[21]:


df.shape


# In[22]:


print (corr['Exited'].sort_values(ascending=False)[:15], '\n') #top 15 values
print ('----------------------')
print (corr['Exited'].sort_values(ascending=False)[-10:])


# In[23]:


# displot label
sb.distplot(df['Exited'])


# In[24]:


# drop unwanted col
df.drop(columns=["RowNumber", "CustomerId"], inplace = True)


# In[25]:


# feature scaling 
def feature_scaling(df):

    for name in df.columns:
        df[name] = (df[name] - df[name].min()) / (df[name].max()-df[name].min())
    print(df.head())
    return df

feature_scaling(df)


# In[26]:


# seperate data set
def features(df):
    output = df.Exited 
#     y.head()
    df = df.drop('Exited', axis = 1)
    print("Seperate Target column from dataset")
    return df, output
df, output = features(df)


# In[27]:


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


# In[28]:


class MultiNeuralNetwork:

    def __init__(self):
        # Learning Rate
        self.l_rate = 0.001
        # Total iterations
        self.echops = 2000

    #Defing a function to train our data
    def train(self, x_train_data, y_train_data):

        layers = [x_train_data.shape[1], 15, 10, 5, 1]
        z = 0
        weight = []
        baised = []
        a = [0] * len(layers)
        z = [0] * len(layers)
        dg = [0] * len(layers)
        da = [0] * len(layers)
        dz = [0] * len(layers)
        db = [0] * len(layers)
        dw = [0] * len(layers)
        
        a[0] = x_train_data.T 
        m = len(x_train_data.shape)
        n = x_train_data.shape[1]

        for j in range(1, len(layers)):
            
            weight.append(np.random.rand(layers[j], layers[j-1]) * 0.001 ) 
            baised.append(np.zeros(((layers[j], 1)) ))
 
        for k in range(self.echops):

            for i in range(len(layers) - 1):  
        
                z[i] = np.dot(weight[i], a[i]) + baised[i] 
                a[i+1] = 1 / (1 + np.exp(-z[i]))

                # d_g = a * (1 - a)
                dg[i] = (1 / (1 + np.exp(-z[i]))) * (1 - (1 / (1 + np.exp(-z[i])))) 

                  # Loss Funtion
                # da = -(y/a) + ((1-y)/(1-a))
                da[i] = (-(y_train_data.T / a[i+1]) + ((1 - y_train_data.T) / (1 - a[i+1]))) 

                # dz = da * d_g
                dz[i] = (da[i] * dg[i]) 

                #dw = (dz * a) / m
                dw[i] = np.dot(dz[i], a[i].T)/ m 

                # db = np.sum(dz)
                db[i] = np.sum(dz[i], axis=1, keepdims=True) / m 

                # w = w - alpha * dw
                # b = b - alpha * db
                weight[i] =  weight[i] - np.dot(self.l_rate, dw[i]) 
                baised[i] = baised[i] - np.dot(self.l_rate, db[i])
                
        print("shape of Z", z[i].shape)
        print("shape of a[i+1]", a[i+1].shape)
        print("shape of dg[i]", dg[i].shape)
        print("shape of da[i]", da[i].shape)
        print("shape of dz[i]", dz[i].shape)
        print("shape of dw[i]", dw[i].shape)
        print("shape of db[i]", db[i].shape)
        print("shape of weight[i]", weight[i].shape)
        print("shape of baised[i]", baised[i].shape)

        return [weight, baised]
    
    def predict(self, x_test_data, parameters):

        a = [0] * 6
        a[0] = x_test_data.T
        z = [0] * 6
        a = [0] * 6
        for i in range(4):  
            z[i] = np.dot(parameters[0][i], a[i]) + parameters[1][i]
            a[i+1] = 1 / (1 + np.exp(-z[i])) 
        return a[-1]
        return y_predict

    def accuracy(self, y_data_test, y_pred_test):

        y_pred_test = np.nan_to_num(y_pred_test)
    
        test_accuracy = 100 - (np.mean(np.abs(y_pred_test - y_data_test)) * 100)        
        return test_accuracy


def main(x_train_data,y_train_data,x_test_data,y_test_data):

    #creating object for class
    obj = MultiNeuralNetwork()
    

    
    
    parameters = obj.train(x_train_data, y_train_data)
    
    y_prediction = obj.predict(x_test_data, parameters)
    print("shape of y_prediction",y_prediction.shape)
    
    y_prediction_train = obj.predict(x_train_data, parameters)
    print("shape of y_prediction_train",y_prediction_train.shape)
    
    accuracy = obj.accuracy(y_test_data, y_prediction)
    accuracy_train = obj.accuracy(y_train_data, y_prediction_train)

    print("Train accuracy: ", accuracy_train)
    print("Test accuracy: ", accuracy)

main(x_train_data,y_train_data,x_test_data,y_test_data)


# In[ ]:




