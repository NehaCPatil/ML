from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import pickle
import csv
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sb

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import fbeta_score

import matplotlib.pyplot as plt
from matplotlib import pyplot as plt 
from matplotlib.colors import ListedColormap

# for encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing 
from collections import defaultdict

class function:


    
    def load_data(self,filename):
        dataset = pd.read_csv(filename)
        return dataset
    
    """ split train and test data """
    
    def splitdata(self,dataset,size_test,size_cv,dir_name):
#         os.mkdir("CSV")
        # split train and test data
        train, test = train_test_split(dataset,test_size = size_test, random_state=0)
        print("train : ", train.shape, " test : ", test.shape)
        #     print("y_train : ", y_train.shape, " y_test : ", y_test.shape)

        # saving datasets into csv files
        test.to_csv(dir_name+'/test_file.csv',index=False,encoding='utf-8')


        # divide train data into train and cross validation 
        train_data, crossV_data = train_test_split(train,test_size = size_cv ,random_state=0)

        train_data.to_csv(dir_name+'/train_file.csv',index=False,encoding='utf-8')
        crossV_data.to_csv(dir_name+'/crossV_file.csv',index=False,encoding='utf-8')



        print("train_data : ", train_data.shape, " crossV_data : ", crossV_data.shape)
        
        
    """One hot Encoding"""
    def one_hot_encoding(self,x_train):
        # from collections import defaultdict
        d = defaultdict(LabelEncoder)
        # Encoding the variable
        fit = x_train.apply(lambda x: d[x.name].fit_transform(x))
        # Inverse the encoded
        fit.apply(lambda x: d[x.name].inverse_transform(x))
        # Using the dictionary to label future data
        x_train.apply(lambda x: d[x.name].transform(x))
        obj_oneh = OneHotEncoder()
        obj_oneh.fit(x_train)
        x_train=obj_oneh.transform(x_train).toarray()
        return x_train

        
        
    """Feature Scaling"""
    
    def feature_scale(self,x_train):
        sc_x = StandardScaler()
        x_train = sc_x.fit_transform(x_train)
        x_test = sc_x.transform(x_test)
        return x_train, sc_x
    


    """ predicting the test set result and train set result """
    
    def prediction(self,x,obj_model):
    
        y_pred_train = obj_model.predict(x)
        return y_pred_train
    
    
    """ Accuracy for train and test """
    
    def Accuracy(self,y_train,y_predict):
        
        Accuracy = accuracy_score(y_train, y_predict)*100
        print("Accuracy: accuracy_score :", Accuracy)
        
        accuracy = average_precision_score(y_train, y_predict)* 100
        print("Accuracy: average_precision_score :", accuracy)
        
        accuracy1 = balanced_accuracy_score(y_train, y_predict)*100
        print("Accuracy:  balanced_accuracy_score :", accuracy1)
        
        
        accuracy2 = hamming_loss(y_train, y_predict)*100
        print("Accuracy:  hamming_loss :", accuracy2)
        
        
        acc = fbeta_score(y_train, y_predict, average='micro', beta=0.5)*100
        print("Accuracy:  fbeta_score :", acc)
        
        
        return Accuracy,accuracy,accuracy1,accuracy2,acc
    
    
    """visualising the training set results"""
    
    def plotting(self,x, y,y_pred_train,Title,x_label,y_label):
        
        plt.scatter(x, y , color = 'pink')
        plt.plot(x, y_pred_train, color = 'red')
        plt.title(Title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
    
    # visualisation for logistic regression
    
    def visualisation(self,x_train_data,y_train,classifier):
        
        x_set,y_set = x_train_data,y_train
        x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop = x_set[:,0].max()+1 , step = 0.01),
                            np.arange(start = x_set[:,1].min()-1,stop = x_set[:,1].max()+1 , step = 0.01))
        plt.contourf(x1,x2, classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha = 0.75, cmap=ListedColormap(('pink','blue')))
        plt.xlim(x1.min(),x1.max())
        plt.ylim(x2.min(),x2.max())
        for i,j in enumerate (np.unique(y_set)):
            plt.scatter(x_set[y_set == j,0],x_set[y_set == j ,1],c = ListedColormap(('pink','blue')) (i),label = j)
        plt.title("Logistic Regression(Training set)")
        plt.xlabel("Age")
        plt.xlabel("Estimated Salary")
        plt.legend()
        plt.show()
    
    """ dump train model pickle file"""
    
    def pickel_file(self,obj_model,filename):
        
       
        file = open(filename, 'wb')
        pickle.dump(obj_model,file)
        file.close()
        
      
    
    
            
