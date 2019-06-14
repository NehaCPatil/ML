
# coding: utf-8

# # Age Detection of Indian Actors
# The dataset contains thousands of images of Indian actors and your task is to identify their age. 
# All the images are manually selected and cropped from the video frames resulting in a high degree of variability interms of scale, pose, expression, illumination, age, resolution, occlusion, and makeup.
# 

# In[ ]:


import numpy as np
import pandas as pd
import os
import cv2


# In[2]:


train=pd.read_csv('Dataset/train.csv')
train.head()


# In[3]:


test=pd.read_csv('Dataset/test.csv')
test.head()


# In[4]:


train.describe()


# In[5]:


def image_read(imglist,path):
    image_array=[]
    for i in imglist:
        image=cv2.imread(path+i)
        image=cv2.resize(image,(64,64))
        image_array.append(image)
    image_array=np.array(image_array)    
    return image_array


# In[6]:


train_img_path="Train/"
imglist=train['ID']
#print(imglist)
X_train=image_read(imglist,train_img_path)


# In[77]:


X_train.shape


# In[7]:


test_img_path="Test/"
imglist1=test['ID']
X_test=image_read(imglist1,test_img_path)


# In[8]:


X_test.shape


# In[10]:


X_train=X_train/255


# In[11]:


X_test=X_test/255


# In[12]:


Y_train=train['Class']
Y_train = Y_train.map({'YOUNG': 0, 'MIDDLE': 1, 'OLD': 2})


# In[13]:


from sklearn.preprocessing import LabelBinarizer 
label_binarizer = LabelBinarizer()
label_binarizer.fit(Y_train)

Y_train=label_binarizer.transform(Y_train)


# In[33]:


from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid=train_test_split(X_train,Y_train,test_size=0.15, random_state=42)


# In[34]:


X_train.shape


# In[32]:


from keras.preprocessing.image import ImageDataGenerator


# In[40]:


datagen_train = ImageDataGenerator(
    width_shift_range=0.2,  # randomly shift images horizontally 
    height_shift_range=0.2,# randomly shift images vertically 
    
    horizontal_flip=True) # randomly flip images horizontally

# fit augmented image generator on data
datagen_train.fit(X_train)


# In[41]:


from keras.layers import Input,InputLayer, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Sequential,Model
from keras.callbacks import ModelCheckpoint


# In[42]:


inputShape=(64,64,3)


# In[43]:


input = Input(inputShape)
#x = Conv2D(32,(3,3),strides = (1,1),name='conv_layer1')(xInput)

x = Conv2D(64,(3,3),strides = (1,1),name='layer_conv1',padding='same')(input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2),name='maxPool1')(x)
#x = Dropout(0.5)(x)

x = Conv2D(128,(3,3),strides = (1,1),name='layer_conv2')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2),name='maxPool2')(x)
#x = Dropout(0.5)(x)

x = Conv2D(128,(3,3),strides = (1,1),name='layer_conv3')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2),name='maxPool3')(x)
#x = Dropout(0.5)(x)

x = Conv2D(64,(3,3),strides = (1,1),name='conv4')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2),name='maxPool4')(x)


x = Flatten()(x)
x = Dense(64,activation = 'relu',name='fc0')(x)
x = Dropout(0.25)(x)
x = Dense(32,activation = 'relu',name='fc1')(x)
x = Dropout(0.25)(x)
x = Dense(3,activation = 'softmax',name='fc2')(x)

model = Model(inputs = input,outputs = x,name='Predict')

   


# In[44]:


model.summary()


# In[28]:


#model.load_weights('final_weight.h5')


# In[45]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[46]:


checkpointer = ModelCheckpoint(filepath='cnnweights.best.eda.hdf5', verbose=1, save_best_only=True)


# In[48]:


model.fit_generator(datagen_train.flow(X_train, Y_train, batch_size=16), validation_data=(X_valid, Y_valid),
                          epochs=10,steps_per_epoch=X_train.shape[0],callbacks=[checkpointer], verbose=1)


# In[49]:


X_test.shape


# In[50]:


pred=model.evaluate(X_train,Y_train)
print("Accuracy : " +str(pred[1]*100))
print("Total Loss  " +str(pred[0]*100))


# In[53]:


print("on valid data")
pred1=model.evaluate(X_valid,Y_valid)
print("accuaracy", str(pred1[1]*100))
print("Total loss",str(pred1[0]*100))


# In[54]:


model.save_weights('final_weight_cnn.h5')


# In[56]:


predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis= 1)


# In[57]:


classes={0: 'YOUNG', 1: 'MIDDLE', 2: 'OLD'}


# In[58]:


predicted_class=[classes[x] for x in predictions]


# In[59]:


predicted_class[0:3]


# In[60]:


sub = pd.DataFrame({
    "Class": predicted_class,
    "ID": test['ID']
})


# In[61]:


sub.to_csv("cnn_predicted_class.csv", index=False)

