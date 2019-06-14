
# coding: utf-8

# ### Convolutional Neural Network
#     Apply CNN to predict cat and dog from given image dataset
#     
#     
# Layers needed by CNN
# 
# Conv2D :- Basic Convolutional layer . Here we will be using a 64 neuron layer
# 
# Dense :- Dense layer is needed by every neural network to finally output the result however every once in while using a Dense layer helps in making model learn.
# 
# MaxPooling :- CNN has a concept of max pooling. After every convoulution we get some values in a kernel. However in max pooling we select max kernel value.
# 
# Flatten:- Conv2D layer returns doesn't return a flatten data hence we need Flatten layer before feeding it into final Dense layer
# 

# In[ ]:


import theano
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve


# Enough of data processing I wanna train :) Here are the steps to do define our CNN model
# 
# Define a Sequential model
# Start adding layers to it.
# First we will add a Conv2D layer with 64 nodes and kernel size of (3,3). 
# You can also experiment with different values here like 32, 128 etc. 
# Also we have to specify input shape which is your X shape. 
# Activation we will take 'relu' for now however there are many others to experiment with.
# 
# Now after every Conv layer we always do max pooling so we will add max pooling layer 
# with a size of (2,2)
# 
# We will repeat this combination again because come on 2 is better than one. Haha.
# We you can also add 3 or more convolution layers but keep in mind the more layers you add more time it will take to train.
# But we don't have much time so we will add a flatten layer now. As we have to feed our data to Dense layer later.
# 
# We will now add a Dense layer of 64 nodes. Note for all these layers we are using activation as 'relu' because I found results better with this.
# You can skip specifying activation but this might make a model a conveniently linear which might not work for us.
# In the end for getting our result we will add final Dense layer . 
# Activation can be sigmoid or softmax (if you need probability use sigmoid else use softmax). 
# Here I have used sigmoid.
# 
# Finally we will compile the model . 
# There are 3 things to mention here . Loss, Optimizer, Metrics
# 
# Loss :- To make our model better we either minimize loss or maximize accuracy. 
# NN always minimize loss. To measure it we can use different formulas like 'categorical_crossentropy' or 'binary_crossentropy'. Here I have used binary_crossentropy
# 
# Optimizer :- If you know a lil bit about mathematics of machine learning you might 
# be familier with local minima or global minima or cost function. 
# To minimize cost function we use different methods For ex :- like gradient descent, stochastic gradient descent. So these are call optimizers. We are using a default one here which is adam
# 
# Metrics :- This is to denote the measure of your model. 
#     Can be accuracy or some other metric.

# In[14]:


import keras # Test out Theano when time permits as well
from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Convolution2D,Flatten


# In[3]:


# Initializing the ANN

classifier = Sequential()


# In[4]:


# step1 - Convolution layer

classifier.add(Convolution2D(32,3,3, input_shape =(64,64,3),activation='relu'))


# In[5]:


# step 2 - Pooling layer

classifier.add(MaxPooling2D(pool_size=(2,2)))


# In[6]:


classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


# It’s time for us to now convert all the pooled images into a continuous vector through Flattening.
# Flattening is a very important step to understand. 
# What we are basically doing here is taking the 2-D array, 
# i.e pooled image pixels and converting them to a one dimensional single vector.

# In[7]:


# step 3 - Flattening

classifier.add(Flatten())


# In this step we need to create a fully connected layer, 
# and to this layer we are going to connect the set of nodes we got after the flattening step,
# these nodes will act as an input layer to these fully-connected layers.
# As this layer will be present between the input layer and output layer, 
# we can refer to it a hidden layer.

# In[8]:


# step 4 - Full connection

classifier.add(Dense(output_dim = 300, activation='relu'))
classifier.add(Dense(output_dim = 1, activation='sigmoid'))


# In[9]:


# step - 5 Compiling the CNN

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# ### ImageDataGenerator class from Keras.
# 
# CNN model is ready to rumble. Further we use ImageDataGenerator class from Keras. 
# It’s generate batches of image(matrix) data with real-time data augmentation(image processing). 
# The data will be looped over (in batches) indefinitely
# 
# 
# 
# Now we will fit our model with training data.
# 
# Epochs :- How many times our model will go through data
# 
# Batch size :- How much amount of data at once you wanna pass through the model
# 
# validation_split :- How much amount of data (in this case its 20 %) you will need to check cross validation error

# In[10]:


# fitting the CNN
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_set,
        steps_per_epoch=80,
        epochs=25,
        validation_data=test_set,
        validation_steps=20)


# In[26]:


file = open('Pickel_classifier/model.pkl', 'wb')
pickle.dump(classifier,file)
# pickle.dump(test_set,file)
file.close()

