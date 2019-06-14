
# coding: utf-8

# # Handwritten CNN 

# In[32]:


#Importing all packages
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import keras
from keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard


# In[33]:


from keras import backend as K
K.set_image_dim_ordering('th')
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[6], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

# reshape to be [batch][height][width][channels]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')


# In[34]:


# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255


# In[35]:


# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# In[36]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[37]:


# define the larger model
def larger_model():
    # create model
    model = Sequential()
    model.add(Convolution2D(30, (5, 5), input_shape=(X_train.shape[1], X_train.shape[2], 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[38]:


# build the model
classifier = larger_model()
# Fit the model
classifier.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# Final evaluation of the model
scores = classifier.evaluate(X_test, y_test, verbose=0)
# print("Large CNN Error: %.2f%%" % (100-scores[1]*100))


# In[39]:


print("Large CNN Error: %.2f%%" % (100-scores[1]*100))


# In[40]:


# Save the model
classifier.save('mnistCNN.h5')


# In[41]:


NAME = "Handwritten-image-digit-classification-CNN"

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
classifier.fit(X_train, y_train,batch_size=200, validation_data=(X_test, y_test), epochs=5, callbacks=[tensorboard])


# In[42]:


# Final evaluation of the model
print("Test loss & Test Accuracy: ")
# evaluate trained model
val_loss, val_acc = classifier.evaluate(X_test, y_test)
print("Validation/Test Loss: ",val_loss)
print("Validation/Test Accuracy:",(val_acc)*100)


# In[43]:


# loaded classifier model
new_model = tf.keras.models.load_model('mnistCNN.h5')


# In[44]:


# predict x_test data on new_model
prediction = new_model.predict([X_test])


# In[45]:


print(prediction[6])


# In[51]:


# 
print(np.argmax(prediction[0]))

