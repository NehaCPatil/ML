#!/usr/bin/env python
# coding: utf-8

# In[1]:


# model to predict user will click the ad or not based on his experience and estimated salary for a given data


# In[3]:


# import libraries
import os
# ignore warnings
import warnings
import numpy as np
warnings.filterwarnings('ignore')
# feature scaling
from sklearn.preprocessing import StandardScaler
# for handling categorical data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
# for logistic regression model
from sklearn.ensemble import RandomForestClassifier

# import function python file
import importlib.util

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve, GridSearchCV

from sklearn.metrics import explained_variance_score, make_scorer
from matplotlib import pyplot as plt


# In[5]:


# importing template file 
spec = importlib.util.spec_from_file_location("function", "/home/admin1/PycharmProjects/Machine-Learning-Followship-Program/week_11/Utility/common_function.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
# creating object of Template class
object_common_function = foo.function()


# In[6]:


# Import dataset

dataset = object_common_function.load_data("Data/Social_Network_Ads .csv")
dataset.head()


# In[7]:


# drop unwanted col

dataset.drop(columns=['User ID','Gender'], inplace = True)


# In[8]:


# information of dataset

dataset.info()


# In[9]:


# description of dataset 

dataset.describe().T


# In[10]:


# check for null

dataset.isnull().sum()


# In[11]:


# check for duplicates
dataset.duplicated().sum()


# In[12]:


dataset.min()


# In[13]:


dataset.dtypes


# In[14]:


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


# In[15]:


# os.mkdir("CSV_files")


# In[16]:


# call function split data set

object_common_function.splitdata(dataset,0.30,0.20,"CSV_files")


# In[17]:


# load train dataset

train_data = object_common_function.load_data("CSV_files/train_file.csv")
print("train_data has {} rows and {} Columns".format(train_data.shape[0],train_data.shape[1]))


# In[18]:


# check information 

train_data.info()


# In[19]:


train_data.head()


# In[20]:


size = len(train_data)
cv = KFold(size, shuffle=True)


# In[21]:


# split dataset in feature and label

x_train = train_data.iloc[:,[0,1]].values
y_train = train_data.iloc[:,2].values


# In[22]:


class Random_Forest_Classification:
    
    # feature scaling
    
    def feature_scale(self,x_train):
        sc_x = StandardScaler()
        x_train = sc_x.fit_transform(x_train)
        return x_train
    
    # fit logistic regression model
    
    def fit_model(self,x_train,y_train):
        
        classifier = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
        
        
        classifier.fit(x_train,y_train)
        # Then we create the CAP Curve with the following code :

#         y_pred_proba = classifier.predict_proba(X=X_test)
#         capcurve(y_values=y_test, y_preds_proba=y_pred_proba[:,1])
        
        return classifier
    
    # confusion matrix
    
    def confusion_matrix(self,y_train,y_predict): 
        cm = confusion_matrix(y_train,y_predict)
        return cm
    
    def learning_curve(self,classifier, x_train,y_train, y_predict):
     
    
        # Learning curve
#         print(x.shape)
#         print(y.shape)
        # Visualization the training set result
        train_sizes, train_scores, test_scores = learning_curve(classifier, x_train, y_train, n_jobs=-1, cv=cv, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.title("Random_forest Classification")
        plt.legend(loc="best")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.gca().invert_yaxis()

        # box-like grid
        plt.grid()

        # plot the std deviation as a transparent range at each training set size
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

        # plot the average training and test score lines at each training set size
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

        # sizes the window for readability and displays the plot
        # shows error from 0 to 1.1
        plt.ylim(-.1,1.1)
        plt.show()



    
    
    
def main(x_train,y_train):
    
    # object
    object_LR = Random_Forest_Classification()
    
    # call function
    x_train_data = object_LR.feature_scale(x_train)
        
    classifier = object_LR.fit_model(x_train_data,y_train)
    
    y_predict = object_common_function.prediction(x_train_data,classifier)
    
    
    confusion_mat = object_LR.confusion_matrix(y_train,y_predict)
    print("\n confusion_matrix\n",confusion_mat )
    
    object_common_function.visualisation(x_train_data,y_train,classifier)
    object_LR.learning_curve(classifier,x_train,y_train,y_predict)
    
    accuracy,accuracy1,accuracy2,accuracy3,acc = object_common_function.Accuracy(y_train,y_predict)
    
    
    object_common_function.pickel_file(classifier,'trainmodel.pkl')
    print("pickel file create")

if __name__ == '__main__':
    main(x_train,y_train)

