
# coding: utf-8

# #### Bank Customer Classification
# Artificial Neural Network
#  
#  For a given dataset predict whether customer will exit (Output variable “Exited”) the bank or not using artificial neural network
# 

# In[1]:


import os
import pandas as pd
import pandas.api.types as ptypes
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle


# In[2]:


# load dataset
df_original = pd.read_csv('Data/Churn_Modelling.csv')
df = df_original


# In[3]:


# print top 5 row
df.head()


# In[4]:


# describe dataset
df.describe()


# In[5]:


print ('The train data has {0} rows and {1} columns'.format(df.shape[0],df.shape[1]))


# In[6]:


df.info()


# In[7]:


# check for col
print("\n Column name",df.columns)


# In[8]:


# check for datatype
print("\n",df.dtypes)


# In[9]:


# check for null value
print("\n\n",df.isnull().sum())


# In[10]:


# calculating  sum of null values
df.isna().sum()


# In[11]:


# check fo duplicate data
print("\n check for duplicate \n",df.duplicated().sum())
           


# In[12]:


# drop unwanted col
df.drop(columns=["RowNumber", "CustomerId","Surname"], inplace = True)
df.dtypes


# In[13]:


df.describe().T


# In[14]:


# check for min values
df.min()


# In[15]:


# replace min values with mean
df.replace(0.0, df.mean(),inplace= True)


# In[16]:


# check for min values 
df.min()


# In[17]:


# check for correlation
df.corr()


# In[18]:


corr = df.corr()
sb.heatmap(corr)


# In[19]:


df.shape


# In[20]:


print (corr['Exited'].sort_values(ascending=False)[:15], '\n') #top 15 values
print ('----------------------')
print (corr['Exited'].sort_values(ascending=False)[-10:])


# In[21]:


# displot label
sb.distplot(df['Exited'])


# In[22]:


df.head()


# In[23]:


x_data= df.iloc[:,0:10].values
y_data= df.iloc[:,10].values


# In[24]:


x_data


# In[25]:


x_data.shape, y_data.shape


# In[26]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder() 
x_data[:,1] = labelencoder_X_1.fit_transform(x_data[:,1])

labelencoder_X_2 = LabelEncoder() 
x_data[:, 2] = labelencoder_X_2.fit_transform(x_data[:, 2])
"""
It label France=0,  Spain=2,  Germany= 1 using LabelEncoder
"""
print(x_data.shape)
x_data


# In[27]:


onehotencoder = OneHotEncoder(categorical_features = [1])
x_data = onehotencoder.fit_transform(x_data).toarray()
x_data = x_data[:, 1:]
print(x_data.shape)
x_data


# In[39]:


from sklearn.model_selection import train_test_split
# os.mkdir("Pickle")
        # split train and test data
x_train, x_test,y_train,y_test = train_test_split(x_data,y_data,test_size = 20, random_state=0)


# saving datasets into pickel files
file = open('Tensorflow_testdata_Pickle/testing.pickle', 'wb')

# dump information to that file
pickle.dump(x_test, file)
pickle.dump(y_test,file)

# close the file
file.close()


# divide train data into train and cross validation 
x_train, x_cv,y_train,y_cv = train_test_split(x_train,y_train,test_size = 20 ,random_state=0)


# In[40]:


# # dump train model pickle file
file = open('Tensorflow_testdata_Pickle/testing.pickle', 'rb')
x_test = pickle.load(file)
y_test = pickle.load(file)


# In[31]:


x_train.shape, y_train.shape


# In[32]:


sc = StandardScaler() 
x_train = sc.fit_transform(x_train) 
x_cv = sc.transform(x_cv) 


# In[33]:


x_cv.shape, y_cv.shape


# In[34]:


x_test.shape, y_test.shape


# In[37]:


import tensorflow as tf

y_train = y_train.reshape(9960, 1)
y_test= y_test.reshape(-1,1)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 100

x = tf.placeholder('float', [None, 11])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([11, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(10):
                _, c = sess.run([optimizer, cost], feed_dict={x: x_train, y: y_train})
                epoch_loss += c
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))*100
        print('Accuracy:',accuracy.eval({x:x_train, y:y_train}))
        print('Accuracy:',accuracy.eval({x:x_test, y:y_test}))
        
train_neural_network(x)


# In[38]:


import tensorflow as tf

y_train = y_train.reshape(9960,1)
y_test= y_test.reshape(-1,1)
y_cv = y_cv.reshape(-1,1)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 5500
# model_path = "/tmp/model.ckpt"

x = tf.placeholder('float', [None, 11])
y = tf.placeholder('float')


print(y.shape, type(y))
def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([11, n_nodes_hl1])),
                     'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                     'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                     'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                   'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels = y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
   
    hm_epochs = 10
   
    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(10):
                _, c = sess.run([optimizer, cost], feed_dict={x: x_train, y: y_train})
                epoch_loss += c
                # Create a checkpoint in every iteration
                #saver.save(sess, 'model_iter', global_step=_)

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
                       

        # Save the final model
        saver.save(sess, 'Tensorflow_testdata_Pickle/model_final')
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))*100
        print('Accuracy:',accuracy.eval({x:x_cv, y:y_cv}))
        
        
    print("\n\n Second session Start Testing dataset\n\n")
    with tf.Session() as sess:

    
        sess.run(tf.initialize_all_variables())
        # Restore model weights from previously saved model
        saver.restore(sess,'model_save/model_final')

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(10):
                _, c = sess.run([optimizer, cost], feed_dict={x: x_train, y: y_train})
                epoch_loss += c


            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))*100
        print('Accuracy:',accuracy.eval({x:x_test, y:y_test}))

        
        

train_neural_network(x)

