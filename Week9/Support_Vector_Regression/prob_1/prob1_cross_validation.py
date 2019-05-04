#!/usr/bin/env python
# coding: utf-8

# In[40]:


import os
import pandas as pd
import pandas.api.types as ptypes
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import *
import pickle
import csv
from sklearn.metrics import accuracy_score


# In[41]:


dataset1 = pd.read_csv("CSV_file/cv_data.csv")
x =dataset1.iloc[:,:-1].values
y =dataset1.iloc[:,1].values
dataset1.head()


# In[42]:


# check for num of rows and cols

print ('The train data has {0} rows and {1} columns'.format(dataset1.shape[0],dataset1.shape[1]))


# In[43]:


class cross_validation_svm():

    def load(self):
        # dump train model pickle file
        file = open('trainmodel.pkl', 'rb')
        pickle_in = pickle.load(file)
    
    def predict(self,x):
        # predicting the test set result and train set result

        y_pred = pickle_in.predict(x)
        print("pred_cv",y_pred)
        
    def accuracy(self,y,y_pred):
        Accuracy = r2_score(y_pred_cv,y)*100
        print("Accuracy Train",Accuracy)
    
        
def main(x,y):
    
    object_SVM =  cross_validation_svm()
    
    object_SVM.load()    
      
    y_pred = object_SVM.predict(x)
        
    Accuracy = object_SVM.accuracy(y,y_pred)
    
        
    
main(x,y)
    

        


# In[44]:


# visualising the training set results

print("\n visualising using SVR \n ")
plt.scatter(x, y , color = 'pink')
plt.plot(x, y_pred_cv, color = 'red')
plt.title("Truth or Bulff(SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
        
print("\n visualising Regression Model \n ")
plt.scatter(x, y, color ='red')
plt.plot(x, y_pred_cv, color ='green')
plt.title('Truth or Bluff (Reg Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[ ]:





# In[ ]:




