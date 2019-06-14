
# coding: utf-8

# In[25]:


#Importing all packages
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import keras
import pickle
from keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
from tensorflow.keras import backend


# In[28]:


# load test data
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[29]:


X_train.shape,y_train.shape,X_test.shape,y_test.shape


# In[30]:


for  i in range(5,10):
    plt.imshow(X_test[i],cmap='gray')
    plt.show()


# In[31]:


# Reshaping to format (batch, height, width, channels)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')


# In[32]:


# loaded classifier model
new_model = tf.keras.models.load_model('mnistCNN.h5')


# In[33]:


# predicction on new model
prediction = new_model.predict([X_test])


# In[34]:


for i in range(5,10):
    print(np.argmax(prediction[i]))

