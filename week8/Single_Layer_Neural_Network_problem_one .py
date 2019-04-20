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
df_original = pd.read_csv('bank.csv', delimiter = ';')
df = df_original


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


print ('The train data has {0} rows and {1} columns'.format(df.shape[0],df.shape[1]))


# In[6]:


df.info()


# In[7]:


# check for null value
print("\n Column name",df.columns)


# In[8]:


#  check for data type
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


# create dummies for dataframe
df = pd.get_dummies(df)


# In[14]:


df.describe()


# In[15]:


df.min()


# In[16]:


# replace min with mean
df.replace(0.0, df.mean(),inplace= True)


# In[17]:


df.min()


# In[18]:


df.corr()


# In[19]:


corr = df.corr()
sb.heatmap(corr)


# In[20]:


df.shape


# In[21]:


print (corr['y'].sort_values(ascending=False)[:15], '\n') #top 15 values
print ('----------------------')
print (corr['y'].sort_values(ascending=False)[-10:])


# In[22]:



sb.distplot(df['y'])
# sb.distplot(df)


# In[23]:


sb.boxplot(data=df)
plt.show()


# In[24]:


def feature_scaling(df):

    for name in df.columns:
        df[name] = (df[name] - df[name].min()) / (df[name].max()-df[name].min())
    print(df.head())
    return df
feature_scaling(df)


# In[25]:


# seperate data set
def features(df):
    output = df.y
#     y.head()
    df = df.drop('y', axis = 1)
    print("Seperate Target column from dataset")
    return df, output
df, output = features(df)


# In[26]:


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


# In[27]:


class Single_Layer_Neural_Network :
    
    # constructor
    def __init__(self):
        
        # learning rate
        self.learning_rate = 0.01
        # Total iterations
        self.epoch = 60000
    
    # gradient descent function 
    def gradient_descent(self,x_train_data, y_train_data):
        
        x_size = len(x_train_data)
        print("length of x_data",x_size)
        ncol = x_train_data.shape
        dw = 0.0
        dz= 0.0
        db = 0.0
        w = np.full((ncol+1,1),0.75)
        b = np.zeros((1, 1), dtype='float') 
        
        for i in range(self.epoch):
            
            #  calculate Z = W*X = (1 x m)
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

            
        return w, b
    

    
    def prediction(self,w,b,x_test_data):
       
        prediction = np.zeros((x_test_data.shape[0], 1), dtype=float)
        
        print("shape W ",w.shape)
        print("shape", x_test_data.shape)
        z = np.dot(w.T, x_test_data.T) + b # Z = W*X = (1 x m)
        sigmoid = 1 / 1 + np.exp(-z) # # calculate sigmoid = 1/ (1 + np.exp(-z)) + b


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


    
    w, b = obj.gradient_descent(x_train_data, y_train_data )
    
    prediction_test = obj.prediction(w,b,x_test_data)
    print("prediction_test",prediction_test)
    prediction_train = obj.prediction(w,b,x_train_data)
    print("prediction_train",prediction_train)
    
    
     # It calculates the accuracy

    train_accuracy = (100 - np.mean(np.abs(prediction_train - y_train_data)) * 100)
    test_accuracy = (100 - np.mean(np.abs(prediction_test - y_test_data)) * 100)

    print("train_accuracy = ", train_accuracy)
    print("test_accuracy = ", test_accuracy)
    
    
    
            
main(df, x_train_data,y_train_data,x_test_data,y_test_data)            
            
            




