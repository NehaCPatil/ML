#!/usr/bin/env python
# coding: utf-8

# In[1]:


# model to predict user will click the ad or not based on his experience and estimated salary for a given data


# In[2]:


# import libraries
import os
# ignore warnings
import warnings
warnings.filterwarnings('ignore')
# feature scaling
from sklearn.preprocessing import StandardScaler
# for handling categorical data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
# for logistic regression model
from sklearn.svm import SVC

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

dataset = object_common_function.load_data("Social_Network_Ads.csv")
dataset.head()


# In[5]:


# drop unwanted col

dataset.drop(columns=['User ID','Gender'], inplace = True)


# In[6]:


# information of dataset

dataset.info()


# In[7]:


# description of dataset 

dataset.describe().T


# In[8]:


# check for null

dataset.isnull().sum()


# In[9]:


# check for duplicates
dataset.duplicated().sum()


# In[10]:


dataset.min()


# In[11]:


dataset.dtypes


# In[13]:


# os.mkdir("CSV_files")


# In[14]:


# call function split data set

object_common_function.splitdata(dataset,0.30,0.20,"CSV_files")


# In[15]:


# load train dataset

train_data = object_common_function.load_data("CSV_files/train_file.csv")
print("train_data has {} rows and {} Columns".format(train_data.shape[0],train_data.shape[1]))


# In[16]:


# check information 

train_data.info()


# In[17]:


train_data.head()


# In[18]:


# split dataset in feature and label

x_train = train_data.iloc[:,[0,1]].values
y_train = train_data.iloc[:,2].values


# In[20]:


class Support_Vector_Machine:
    
    # feature scaling
    
    def feature_scale(self,x_train):
        sc_x = StandardScaler()
        x_train = sc_x.fit_transform(x_train)
        return x_train
    
    # fit logistic regression model
    
    def fit_model(self,x_train,y_train):
         
        classifier = SVC(kernel='linear', random_state= 0)
        classifier.fit(x_train,y_train)
        return classifier
    
    # confusion matrix
    
    def confusion_matrix(self,y_train,y_predict): 
        cm = confusion_matrix(y_train,y_predict)
        return cm
    
    
    
def main(x_train,y_train):
    
    # object
    object_LR = Support_Vector_Machine()
    
    # call function
    x_train_data = object_LR.feature_scale(x_train)
        
    classifier = object_LR.fit_model(x_train_data,y_train)
    
    y_predict = object_common_function.prediction(x_train_data,classifier)
    
    confusion_mat = object_LR.confusion_matrix(y_train,y_predict)
    print("\n confusion_matrix\n",confusion_mat )
    
    object_common_function.visualisation(x_train_data,y_train,classifier)
    
    accuracy,accuracy1,accuracy2,accuracy3,acc = object_common_function.Accuracy(y_train,y_predict)
    
    
    object_common_function.pickel_file(classifier,'trainmodel.pkl')
    print("pickel file create")

if __name__ == '__main__':
    main(x_train,y_train)


# In[ ]:




