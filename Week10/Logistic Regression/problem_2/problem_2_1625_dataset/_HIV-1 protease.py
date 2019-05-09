#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""The data contains lists of octamers (8 amino acids) and a flag (-1 or 1) 
    depending on whether HIV-1 protease will cleave in the central position (between amino acids 4 and 5)."""


# In[2]:


# import libraries
import os
import pandas as pd
import numpy as np
# ignore warnings
import warnings
warnings.filterwarnings('ignore')
# feature scaling
from sklearn.preprocessing import StandardScaler
# for handling categorical data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
# for logistic regression model
from sklearn.linear_model import LogisticRegression
# import function python file
import importlib.util
from sklearn.metrics import confusion_matrix


# In[3]:


# importing template file 
spec = importlib.util.spec_from_file_location("function", "/home/admin1/PycharmProjects/Machine-Learning-Followship-Program/week10/Utility/common_function.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
# creating object of Template class
object_common_function = foo.function()


# In[4]:


# Import dataset
# dataset = pd.read_csv("Social_Network_Ads.csv")
dataset = pd.read_csv("newHIV-1_data/1625Data.txt",delimiter=",",names=["Peptides", "Result"])
dataset.head()


# In[5]:


dataset.describe().T


# In[6]:


dataset.info()


# In[7]:


# Seperate all amino acids
peptides = np.array([[dataset["Peptides"][i][j] for i in range(dataset.shape[0])] for j in range(8)])
peptides.shape


# In[8]:


# Store the seperated amino acids into a dataframe
dataset_One = pd.DataFrame(peptides.T, columns=list('ABCDEFGH'))
dataset_One.shape
dataset_One.head()


# In[9]:


# dataset = dataset.join(dataset_One)
dataset = pd.concat([dataset, dataset_One], axis=1)
dataset.head()


# In[10]:


# drop unwanted col

dataset = dataset.drop(columns="Peptides", axis=0)
dataset.head()


# In[11]:


# rearrange col dataset
dataset = dataset[['A','B','C','D','E','F','G','H','Result']]
dataset.head()


# In[12]:


# x = dataset.iloc[:,1:].values
x = dataset.loc[:, dataset.columns != 'Result']
y = dataset.iloc[:,-1]


# In[13]:


# print("Feature after  :\n", list(dataset.columns))
# data_dummies = pd.get_dummies(dataset)
# print("Feature after one-hotencoding :\n", list(data_dummies.columns))
# dataset = data_dummies


# In[14]:


print("dataset has {} rows and {} Columns".format(dataset.shape[0],dataset.shape[1]))


# In[15]:


dataset.head()


# In[16]:


os.mkdir("CSV_files")


# In[24]:


object_common_function.splitdata(dataset,0.30,0.38,"CSV_files")


# In[25]:


# load train dataset

train_data = object_common_function.load_data("CSV_files/train_file.csv")
print("train_data has {} rows and {} Columns".format(train_data.shape[0],train_data.shape[1]))


# In[26]:


# check information 

train_data.info()


# In[27]:


train_data.head()


# In[28]:


# x = dataset.iloc[:,1:].values
x_train = dataset.loc[:, dataset.columns != 'Result']
y_train = dataset.iloc[:,dataset.columns == 'Result']


# In[29]:


x_train=object_common_function.one_hot_encoding(x_train)
x_train= pd.DataFrame(x_train)


# In[30]:


class Logistic_regression:
    
    # feature scaling
    
    def feature_scale(x_train):
        sc_x = StandardScaler()
        x_train = sc_x.fit_transform(x_train)
        return x_train
    
    # fit logistic regression model
    
    def fit_model(x_train,y_train):
        classifier = LogisticRegression(random_state=0)
        classifier.fit(x_train,y_train)
        return classifier
    
    # confusion matrix
    
    def confusion_matrix(y_train,y_predict): 
        cm = confusion_matrix(y_train,y_predict)
        return cm
    
    
    
def main(x_train,y_train):
    
    # object
    object_LR = Logistic_regression
    
    # call function
    x_train_data = object_LR.feature_scale(x_train)
        
    classifier = object_LR.fit_model(x_train_data,y_train)
    
    y_predict = object_common_function.prediction(x_train_data,classifier)
    
    confusion_mat = object_LR.confusion_matrix(y_train,y_predict)
    print("\n confusion_matrix\n",confusion_mat )
    
#     object_common_function.visualisation(x_train_data,y_train,classifier)
    
    accuracy,accuracy1,accuracy2,accuracy3,acc = object_common_function.Accuracy(y_train,y_predict)
#     print("Accuracy : ",accuracy,accuracy1,accuracy2,accuracy3,acc)
    
    object_common_function.pickel_file(classifier,'trainmodel.pkl')
    print("pickel file create")

if __name__ == '__main__':
    main(x_train,y_train)


# In[ ]:





# In[ ]:





# In[ ]:




