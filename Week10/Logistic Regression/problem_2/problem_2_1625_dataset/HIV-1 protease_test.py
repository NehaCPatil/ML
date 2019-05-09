#!/usr/bin/env python
# coding: utf-8

# In[1]:


# model to predict user will click the ad or not based on his experience and estimated salary for a given data


# In[23]:


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

import pickle


# In[24]:


# importing template file 
spec = importlib.util.spec_from_file_location("function", "/home/admin1/PycharmProjects/Machine-Learning-Followship-Program/week10/Utility/common_function.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
# creating object of Template class
object_common_function = foo.function()


# In[25]:


# load train dataset

test_data = object_common_function.load_data("CSV_files/test_file.csv")
print("test_data has {} rows and {} Columns".format(test_data.shape[0],test_data.shape[1]))


# In[26]:


# check information 

test_data.info()


# In[27]:


test_data.head()


# In[28]:


# x = dataset.iloc[:,1:].values
x_test = test_data.loc[:, test_data.columns != 'Result']
y_test = test_data.iloc[:,test_data.columns == 'Result']


# In[29]:


x_test=object_common_function.one_hot_encoding(x_test)
x_test= pd.DataFrame(x_test)


# In[30]:


class Logistic_regression:
    
    def feature_scale(x_test):
        sc_x = StandardScaler()
        x_test = sc_x.fit_transform(x_test)
        return x_test
        
    # load logistic regression model
    
    def load():
        # dump train model pickle file
        file = open('trainmodel.pkl', 'rb')
        pickle_in = pickle.load(file)
        return pickle_in
    
    # confusion matrix
    
    def confusion_matrix(y_test,y_predict): 
        cm = confusion_matrix(y_test,y_predict)
        return cm
    
    
    
def main(x_test,y_test):
    
    # object
    object_LR = Logistic_regression
    
    
    x_test = object_LR.feature_scale(x_test)  
    print(x_test.shape)
    pickle_in = object_LR.load()
    
    y_predict = object_common_function.prediction(x_test,pickle_in)
    
    confusion_mat = object_LR.confusion_matrix(y_test,y_predict)
    print("\n confusion_matrix\n",confusion_mat )
    
#     object_common_function.visualisation(x_test,y_test,pickle_in)
    
    accuracy,accuracy1,accuracy2,accuracy3,acc = object_common_function.Accuracy(y_test,y_predict)
#     print("Accuracy : ",accuracy,accuracy1,accuracy2,accuracy3,acc)
    
    
if __name__ == '__main__':
    main(x_test,y_test)


# In[ ]:




