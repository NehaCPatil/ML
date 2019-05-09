#!/usr/bin/env python
# coding: utf-8

# In[1]:


# model to predict user will click the ad or not based on his experience and estimated salary for a given data


# In[2]:


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
from sklearn.tree import DecisionTreeClassifier
# import function python file
import importlib.util
from sklearn.metrics import confusion_matrix

import pickle


# In[3]:


# importing template file 
spec = importlib.util.spec_from_file_location("function", "/home/admin1/PycharmProjects/Machine-Learning-Followship-Program/week10/Utility/common_function.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
# creating object of Template class
object_common_function = foo.function()


# In[4]:


# load train dataset

cross_data = object_common_function.load_data("CSV_files/crossV_file.csv")
print("cross_data has {} rows and {} Columns".format(cross_data.shape[0],cross_data.shape[1]))


# In[5]:


# check information 

cross_data.info()


# In[6]:


cross_data.head()


# In[7]:


cross_data.shape


# In[8]:


# x = dataset.iloc[:,1:].values
x_cv = cross_data.loc[:, cross_data.columns != 'Result']
y_cv = cross_data.iloc[:,cross_data.columns == 'Result']


# In[9]:


x_cv.shape


# In[10]:


# x_cv =pd.get_dummies(x_cv) 
x_cv=object_common_function.one_hot_encoding(x_cv)
x_cv= pd.DataFrame(x_cv)


# In[11]:


x_cv.shape


# In[12]:


class Decision_Tree_Classifier:
    
    def feature_scale(self,x_cv):
        sc_x = StandardScaler()
        x_cv = sc_x.fit_transform(x_cv)
        return x_cv
        
    # load logistic regression model
    
    def load(self):
        # dump train model pickle file
        file = open('trainmodel.pkl', 'rb')
        pickle_in = pickle.load(file)
        return pickle_in
    
    # confusion matrix
    
    def confusion_matrix(self,y_cv,y_predict): 
        cm = confusion_matrix(y_cv,y_predict)
        return cm
    
    
    
def main(x_cv,y_cv):
    
    # object
    object_LR = Decision_Tree_Classifier()
    
    
    x_cv = object_LR.feature_scale(x_cv)    
    pickle_in = object_LR.load()
    
    y_predict = object_common_function.prediction(x_cv,pickle_in)
    
    confusion_mat = object_LR.confusion_matrix(y_cv,y_predict)
    print("\n confusion_matrix\n",confusion_mat )
    
#     object_common_function.visualisation(x_cv,y_cv,pickle_in)
    
    accuracy,accuracy1,accuracy2,accuracy3,acc = object_common_function.Accuracy(y_cv,y_predict)
#     print("Accuracy : ",accuracy,accuracy1,accuracy2,accuracy3,acc)
    
    
if __name__ == '__main__':
    main(x_cv,y_cv)


# In[ ]:




