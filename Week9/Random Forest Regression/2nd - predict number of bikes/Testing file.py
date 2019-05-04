#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
# import matplotlib.plotly as plt
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt  
import pandas as pd

#o check accuracy
from sklearn.metrics import accuracy_score
# to check accuracy
from sklearn.metrics import *
import pickle 
#visualization in 3D
from mpl_toolkits.mplot3d import Axes3D
# ignore warnings
import warnings
warnings.filterwarnings('ignore')

import csv


# In[2]:


# load dataset
test_dataset = pd.read_csv ("CSV_files/test_file.csv")
print("Dataset has {} rows and {} Columns".format(test_dataset.shape[0],test_dataset.shape[1])) 


# In[5]:


class Testing():
           
    def y_prediction(self,regression, x_train):
        
        # predicting the test set result
        
        return regression.predict(x_train)
    
    def accuracy(self,y_predict_train,y_train):
        
        # acc using r2
        
        error = r2_score(y_train, y_predict_train)      
        acc_r2 = (1-error)*100
        
        # using median_ab_error
       
        median_ab_error = median_absolute_error(y_train, y_predict_train)
        return acc_r2, median_ab_error
    
    def visualization(self,x_train,y_train, regression):
        
        # Visualization the Decision Tree result (for higher resolution & smoother curve)
        
        x_grid = np.arange(min(x_train), max(x_train), 0.1)
        x_grid = x_grid.reshape((len(x_grid),1))
        
        plt.scatter(x_train,y_train, color = 'pink')

        # reshape x_grid or not both will give same ploting
        
        plt.plot(x_grid,regression.predict(x_grid.reshape(-1,1)), color = 'red')
        plt.title("Random Forest algorithm")
        plt.xlabel("temperature")
        plt.ylabel
        

def main():
    #class obj created
    obj  = Testing()
    
    # seperate fetures and label
    x_test = test_dataset.iloc[:,:-1].values
    y_test = test_dataset.iloc[:,1].values
 
   
    #cross validation
    file1 = open('RandomForestModule.pkl', 'rb')
    reg1 = pickle.load(file1)
    
    # y_prediction ( cross validation)   
    y_test_pre = obj.y_prediction(reg1, x_test)
    
    
    acc_r2, median_ab_error = obj.accuracy(y_test_pre,y_test)
    print("\nAccuracy train by acc_r2", acc_r2)
    print("\nAccuracy train by mean_sq_log", median_ab_error)

    obj.visualization(x_test,y_test,reg1)

if __name__ == '__main__':
    main()


# In[ ]:




