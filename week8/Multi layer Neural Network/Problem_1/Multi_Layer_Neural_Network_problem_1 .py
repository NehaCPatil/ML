
#!/usr/bin/env python
# coding: utf-8

# In[70]:


import os
import pandas as pd
import pandas.api.types as ptypes
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt


# In[71]:


# load dataset
df_original = pd.read_csv('bank.csv', delimiter = ';')
df = df_original


# In[72]:


df.head()


# In[73]:


df.describe()


# In[74]:


print ('The train data has {0} rows and {1} columns'.format(df.shape[0],df.shape[1]))


# In[75]:


df.info()


# In[76]:


# check for null value
print("\n Column name",df.columns)


# In[77]:


#  check for data type
print("\n",df.dtypes)


# In[78]:


# check for null value
print("\n\n",df.isnull().sum())


# In[79]:


# calculating  sum of null values
df.isna().sum()


# In[80]:


# check fo duplicate data
print("\n check for duplicate \n",df.duplicated().sum())
           


# In[81]:


# create dummy Variable for Purchased
df.replace(['yes','no'],[1,0],inplace=True)
df.head()


# In[82]:


# create dummies for dataframe
df = pd.get_dummies(df)


# In[83]:


df.describe()


# In[84]:


df.min()


# In[85]:


# replace min with mean
df.replace(0.0, df.mean(),inplace= True)


# In[86]:


df.min()


# In[87]:


df.corr()


# In[88]:


corr = df.corr()
sb.heatmap(corr)


# In[89]:


df.shape


# In[90]:


print (corr['y'].sort_values(ascending=False)[:15], '\n') #top 15 values
print ('----------------------')
print (corr['y'].sort_values(ascending=False)[-10:])


# In[91]:



sb.distplot(df['y'])
# sb.distplot(df)


# In[92]:


sb.boxplot(data=df)
plt.show()


# In[93]:


def feature_scaling(df):

    for name in df.columns:
        df[name] = (df[name] - df[name].min()) / (df[name].max()-df[name].min())
    print(df.head())
    return df
feature_scaling(df)


# In[94]:


# seperate data set
def features(df):
    output = df.y
#     y.head()
    df = df.drop('y', axis = 1)
    print("Seperate Target column from dataset")
    return df, output
df, output = features(df)


# In[95]:


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


# In[97]:


class MultiNeuralNetwork:

    def __init__(self):
        # Learning Rate
        self.l_rate = 0.001
        # Total iterations
        self.echops = 600

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
            #--------------- Forward Propagation ---------------------#
            for i in range(len(layers) - 1):
                print("Layer : ", i)
                print("weight ", i, "= ", weight[i].shape)
                print("baised ", i, "= ", baised[i].shape)
        
                z[i] = np.dot(weight[i], a[i]) + baised[i]
                print("shape of Z", z[i].shape)
                
                # Sigmoid/ Activation function
                a[i+1] = 1 / (1 + np.exp(-z[i]))
                print("shape of a[i+1]", a[i+1].shape)
            
            #------------- Backward Propagation ---------------------#   
            for i in reversed(range(len(layers) - 1)):
                print("Layer : ", i)

                # d_g = a * (1 - a)
                dg[i] = (1 / (1 + np.exp(-z[i]))) * (1 - (1 / (1 + np.exp(-z[i]))))
                print("shape of dg[i]", dg[i].shape)

                
                # da = -(y/a) + ((1-y)/(1-a))
                da[i] = (-(y_train_data.T / a[i+1]) + ((1 - y_train_data.T) / (1 - a[i+1])))
                print("shape of da[i]", da[i].shape)
                

                # dz = da * d_g
                dz[i] = (da[i] * dg[i])
                print("shape of dz[i]", dz[i].shape)

                #dw = (dz * a) / m
                dw[i] = np.dot(dz[i], a[i].T) / m
                print("shape of dw[i]", dw[i].shape)

                # db = np.sum(dz)
                db[i] = np.sum(dz[i], axis=1, keepdims=True) / m
                print("shape of db[i]", db[i].shape)
                
                # w = w - alpha * dw
                weight[i] =  weight[i] - np.dot(self.l_rate, dw[i])
                print("shape of weight[i]", weight[i].shape)
                # b = b - alpha * db
                baised[i] = baised[i] - np.dot(self.l_rate, db[i])
                print("shape of baised[i]", baised[i].shape)
                
        
            

        return [weight, baised]
    
    def predict(self, x_test_data, parameters):

        a = [0] * 5
        a[0] = x_test_data.T
        z = [0] * 5
        a = [0] * 5
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
    
    y_prediction_train = obj.predict(x_train_data, parameters)
    
    accuracy = obj.accuracy(y_test_data, y_prediction)
    accuracy_train = obj.accuracy(y_train_data, y_prediction_train)

    print("Train accuracy: ", accuracy_train)
    print("Test accuracy: ", accuracy)

main(x_train_data,y_train_data,x_test_data,y_test_data)


# In[ ]:





# In[ ]:




