#!/usr/bin/env python
# coding: utf-8

# In[1]:


# model to predict user will click the ad or not based on his experience and estimated salary for a given data


# In[21]:


# import libraries
import os
import pandas as pd
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


# In[22]:


# importing template file 
spec = importlib.util.spec_from_file_location("function", "/home/admin1/PycharmProjects/Machine-Learning-Followship-Program/week10/Utility/common_function.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
# creating object of Template class
object_common_function = foo.function()


# In[23]:


# Import dataset
# dataset = pd.read_csv("Social_Network_Ads.csv")
dataset = object_common_function.load_data("Social_Network_Ads.csv")
dataset.head()


# In[24]:


# drop unwanted col

dataset.drop(columns=['User ID','Gender'], inplace = True)


# In[25]:


# information of dataset

dataset.info()


# In[26]:


# description of dataset 

dataset.describe().T


# In[27]:


# check for null

dataset.isnull().sum()


# In[28]:


# check for duplicates
dataset.duplicated().sum()


# In[29]:


dataset.min()


# In[30]:


dataset.dtypes


# In[31]:


# # handle categorical data
# def handle_categorical_data(x_data):
#     #encode categorical data
    
#     label_encod = LabelEncoder()
#     x_data[:, 1] = label_encod.fit_transform(x_data[:, 1])
    
# #     # one hot encoding
# #     onehotencode = OneHotEncoder(categorical_features= [1])
# #     x_data = onehotencode.fit_transform(x_data).toarray()
    
#     return x_data
    
# x_data = handle_categorical_data(x_data)


# In[32]:


# os.mkdir("CSV_files")


# In[33]:


# call function split data set

object_common_function.splitdata(dataset,0.30,0.20,"CSV_files")


# In[34]:


# load train dataset

train_data = object_common_function.load_data("CSV_files/train_file.csv")
print("train_data has {} rows and {} Columns".format(train_data.shape[0],train_data.shape[1]))


# In[35]:


# check information 

train_data.info()


# In[36]:


train_data.head()


# In[37]:


# split dataset in feature and label

x_train = train_data.iloc[:,[0,1]].values
y_train = train_data.iloc[:,2].values


# In[38]:


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
    
    object_common_function.visualisation(x_train_data,y_train,classifier)
    
    accuracy,accuracy1,accuracy2,accuracy3,acc = object_common_function.Accuracy(y_train,y_predict)
    print("Accuracy : ",accuracy,accuracy1,accuracy2,accuracy3,acc)
    
    object_common_function.pickel_file(classifier,'trainmodel.pkl')
    print("pickel file create")

if __name__ == '__main__':
    main(x_train,y_train)

