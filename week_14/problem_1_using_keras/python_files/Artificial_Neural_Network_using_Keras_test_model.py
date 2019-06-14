
# coding: utf-8

# # Bank Customer Classification
#  
#  
#  For a given dataset predict whether customer will exit (Output variable “Exited”) the bank or not using artificial neural network
# 

# In[1]:


import theano
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import keras # Test out Theano when time permits as well


# In[2]:


# Splitting the dataset into the Training and Testing set.

file = open('Keras_testdata_Pickle/testing.pickle', 'rb')

# dump information to that file
X_test = pickle.load(file)
y_test = pickle.load(file)


# In[3]:



# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model

# load model
classifier = load_model('Pickel_classifier/model.h5')
# summarize model.
classifier.summary()


# In[4]:


y_pred = classifier.predict(X_test)
print(y_pred)


# To use the confusion Matrix, we need to convert the probabilities that a customer will leave the bank into the form true or false. So we will use the cutoff value 0.5 to indicate whether they are likely to exit or not.

# In[5]:


y_pred = (y_pred > 0.5)
print(y_pred)


# ### Making the Confusion Matrix

# In[6]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# ### Significance of the confusion matrix value:
# 
# The output should be close to the table below:
# 
# ||Predicted: No |Predicted: Yes|
#  |------| |------|   |------|  
#  |Actual: No|1504|91|
# |Actual: Yes|184|221
# 

# This means that we should have about $(1504 + 221) = 1726$ correct classifications out of our total testing data size of $2000$.
# This means that our accuracy for this trial was $1726 \div 2000 = 86.3\%$, which matches the classifier's prediction

# In[7]:


print (((cm[0][0]+cm[1][1])*100)/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]), '% of testing data was classified correctly')


# In[8]:


from sklearn.metrics import classification_report
print("\n",classification_report(y_test, y_pred))

