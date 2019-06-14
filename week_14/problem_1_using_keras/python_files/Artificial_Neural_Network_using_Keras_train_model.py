
# coding: utf-8

# # Bank Customer Classification
#  
#  
#  For a given dataset predict whether customer will exit (Output variable “Exited”) the bank or not using artificial neural network
# 

# In[1]:


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


# ### Data Preprocessing 

# 
# For as much as we care, neither the customer ID, nor the surname should matter in classification. 
# Therefore, we will use columns 3 (CreditScore) inclusive through the 13th column (exclusive).
# 

# In[2]:


#importing the dataset
dataset = pd.read_csv('Data/Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values # Credit Score through Estimated Salary
y = dataset.iloc[:, 13].values # Exited


# In[3]:


dataset.head()


# In[4]:


# Encoding categorical (string based) data. Country: there are 3 options: France, Spain and Germany
# This will convert those strings into scalar values for analysis
print(X[:8,1], '... will now become:')
print(X[:6,2], '... will now become: ')

label_X_country_encoder = LabelEncoder()
label_X_gender_encoder = LabelEncoder()
X[:,1] = label_X_country_encoder.fit_transform(X[:,1])
X[:,2] = label_X_gender_encoder.fit_transform(X[:,2])
print(X[:8,1])
print(X[:6,2])


# The Problem here is that we are treating the countries as one variable with ordinal values (0 < 1 <  2). Therefore, one way to get rid of that problem is to split the countries into respective dimensions. that is,
# 
# | Country |  -> | Country|-> |Spain|France|Germany|
# |------|      |------|  |------|    |------|    |------|
# |   Spain |   -> |0| -> |1|0|0|
# |   France | -> |1| -> |0|1|0|
# |   France | ->  |1| -> |0|1|0|
# |   Germany | -> |2| -> |0|0|1|

# 
# Gender doesn't need to go through a similar process becasue it is binary

# In[5]:


# Converting the string features into their own dimensions. Gender doesn't matter here because its binary
countryhotencoder = OneHotEncoder(categorical_features = [1]) # 1 is the country column
X = countryhotencoder.fit_transform(X).toarray()


# You can now see that the first three columns represent the three countries that constituted the "country" category. We can now observe that  we essentially only need two columns: a 0 on two countries means that the country has to be the one variable which wasn't included. This will save us from the problem of using too many dimensions
# 
# |Spain|France|Germany|-> |France|Germany|
#  |------|    |------|    |------|     |------|     |------|
#  |1|0|0|-> |0|0|
# |0|1|0|-> |1|0|
# |0|1|0|-> |1|0|
# |0|0|1|-> |0|1|

# In[6]:


X = X[:,1:] # Got rid of Spain as a dimension. It is still there through out inferences


# In[7]:


# Splitting the dataset into the Training and Testing set.

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
X_train, X_cv, y_train, y_cv = train_test_split(X_train,y_train, test_size = 0.2, random_state = 0)

file = open('Keras_Pickle/testing.pickle', 'wb')

# dump information to that file
pickle.dump(X_test, file)
pickle.dump(y_test,file)

# close the file
file.close()


# Feature scaling is a method used to standardize the range of independent variables or features of data. It is basically scaling all the dimensions to be even so that one independent variable does not dominate another. For example, bank account balance ranges from millions to 0, whereas gender is either 0 or 1. If one of the features has a broad range of values, the distance will be governed by this particular feature. Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.

# In[8]:


# Feature Scaling

sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[9]:


X_train.shape,X_test.shape


# ##  ANN

# In[10]:


import keras # Test out Theano when time permits as well
from keras.models import Sequential
from keras.layers import Dense


# In[11]:


# Initializing the ANN
classifier = Sequential()


# A hurestic tip is that the amount of nodes (dimensions) in your hidden layer should be the average of your input and output layers, which means that since we have 11 dimensions (representing **Independent variables** Note: Countries still compose only  **one** dimension) and we are looking for a binary output, we calculate this to be $(11+1)\div 2 = 6 $.
# 
# #### The breakdown of the inputs for the first layer is as follows:
# 
# *activiation*: **relu** becasue we are in an input layer. uses the ReLu activation function for $\phi$
# 
# *input_dim*: **11** because we span 11 dimensions in our input layer. This is needed for the first added layer. The subsequent layers's input dimensions can be inferred using the previously added layer's output dimension. The next hidden layer will know what to expect.
# 
# *units*: **6** nodes (number of nodes in hidden layer). Can think of this as number of nodes are in the next layer.
# 
# *kernel_initializer*: **uniform**  the distribution with which we randomly initialize weights for the nodes in this layer. 
# 
# 

# In[12]:


# This adds the input layer (by specifying input dimension) AND the first hidden layer (units)
classifier.add(Dense(activation = 'relu', input_dim = 11, units=6, kernel_initializer='uniform'))


# ### We are going to add another layer to this model because we want to implement Deep Learning, which is an artificial Neural network with many layers.
# We will make our second hidden layer also have 6 nodes, just playing with the same arithmetic we used to determine the dimensions of the first hidden layer (average of your input and output layers) $(11+1)\div 2 = 6 $.

# In[13]:


# Adding the second hidden layer
# Notice that we do not need to specify input dim. 
classifier.add(Dense(activation = 'relu', units=6, kernel_initializer='uniform')) 


# ### Adding the output layer
# #### The breakdown of the inputs for the output layer is as follows:
# 
# *activiation*: **sigmoid** becasue we are in an output layer. uses the Sigmoid activation function for $\phi$. This is used instead of the ReLu function becasue it generates probabilities for the outcome. We want the probability that each customer leaves the bank.  
# 
# *input_dim*: **11** because we span 11 dimensions in our input layer. This is needed for the first added layer. The subsequent layers's input dimensions can be inferred using the previously added layer's output dimension. The next hidden layer will know what to expect.
# 
# *units*: **6** nodes (number of nodes in hidden layer). Can think of this as number of nodes are in the next layer.
# 
# *kernel_initializer*: **uniform**  the distribution with which we randomly initialize weights for the nodes in this layer. 

# In[14]:


# Adding the output layer
# Notice that we do not need to specify input dim. 
# we have an output of 1 node, which is the the desired dimensions of our output (stay with the bank or not)
# We use the sigmoid because we want probability outcomes
classifier.add(Dense(activation = 'sigmoid', units=1, kernel_initializer='uniform')) 


# 
# ### If we want more than two categories, then we will need to change 
# 
#  1) the *units* parameter to match the desired category count
#  
#  2) the *activation* field to **softmax**.  Basically a sigmoid function but applied to a dependent variable that has more than 2 categories.

# ## Compiling the Neural Network
# Basically applying Stochastic Gradient descent on the whole Neural Network. We are Tuning the individual weights on each neuron.
# 
# #### The breakdown of the inputs for compiling is as follows:
# 
# *optimizer*: ** adam** The algorithm we want to use to find the optimal set of weights in the neural networks.  Adam is a very efficeint variation of Stochastic Gradient Descent.
# 
# *loss*: **binary_crossentropy** This is the loss function used within adam. This should be the logarthmic loss. If our dependent (output variable) is *Binary*, it is **binary_crossentropy**. If *Categorical*, then it is called **categorical_crossentropy**
# 
# *metrics*: **[accuracy]** The accuracy metrics which will be evaluated(minimized) by the model. Used as accuracy criteria to imporve model performance. 
# 
# *kernel_initializer*: **uniform**  the distribution with which we randomly initialize weights for the nodes in this layer. 

# In[15]:


classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])


# ## Fitting the Neural Network
# This is where we will be fitting the ANN to our training set.
# 
# #### The breakdown of the inputs for compiling is as follows:
# 
# ** X_train** The independent variable portion of the data which needs to be fitted with the model.
# 
# ** Y_train** The output portion of the data which the model needs to produce after fitting.
# 
# *batch_size*:  How often we want to back-propogate the error values so that individual node weights can be adjusted. 
# 
# *nb_epochs*: The number of times we want to run the entire test data over again to tune the weights. This is like the fuel of the algorithm. 

# In[16]:


classifier.fit(X_train, y_train, batch_size=10, epochs=10)


# In[17]:


classifier.fit(X_cv, y_cv, batch_size=10, epochs=10)


# In[18]:


classifier.save("Pickel_classifier/model.h5")


# 
# ## Testing the ANN
# ### Predicting the coss_validation results
# 
# This shows the probability of a customer leaving given the testing data. Each row in X_cv corresponds to a row in Y_cv

# In[19]:


y_pred = classifier.predict(X_cv)
print(y_pred)


# To use the confusion Matrix, we need to convert the probabilities that a customer will leave the bank into the form true or false. So we will use the cutoff value 0.5 to indicate whether they are likely to exit or not.

# In[20]:


y_pred = (y_pred > 0.5)
print(y_pred)


# ### Making the Confusion Matrix

# In[21]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_cv, y_pred)
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

# In[22]:


print (((cm[0][0]+cm[1][1])*100)/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]), '% of testing data was classified correctly')


# In[23]:


from sklearn.metrics import classification_report
print("\n",classification_report(y_cv, y_pred))

