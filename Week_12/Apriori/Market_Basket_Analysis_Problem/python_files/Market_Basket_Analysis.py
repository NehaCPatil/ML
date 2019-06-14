
#!/usr/bin/env python
# coding: utf-8

# # Association Rule Learning(Apriori)
# Generate association rules for dataset given in the url
# 

# In[1]:


# import libraries
# for basic operations
import numpy as np
import pandas as pd

# for data visualization
from matplotlib import pyplot as plt
import seaborn as sns
import plotly as py

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# import python file
from apyori import apriori


# In[2]:


# reading the data and finding the computation time for uploading the dataset

get_ipython().run_line_magic('time', "data =pd.read_csv('Dataset/Market_Basket_Optimisation.csv',header=None)")

# getting to know the shape of the data
print(data.shape)


# In[3]:


# checking the head of the data

data.head()


# In[4]:


# describing the data

data.describe()


# In[5]:


# checking if the dataset contains any NULL values

data.isnull().any()


# In[6]:


# check for shape

print("data has {} rows and {} Columns".format(data.shape[0],data.shape[1]))


# ###### For instance, if item A and B are bought together more frequently then several steps can be taken to increase the profit.
# For example:
# 
# 1.A and B can be placed together so that when a customer buys one of the product he 
# doesn't have to go far away to buy the other product.
# 
# 2.People who buy one of the products can be targeted through an advertisement campaign to buy the other.
# 
# 3.Collective discounts can be offered on these products if the customer buys both of them.
# 
# 4.Both A and B can be packaged together.
# 
# The process of identifying an associations between products is called association rule mining.

# #### Theory of Apriori Algorithm
# There are three major components of Apriori algorithm:
# 
# 1.Support
# 
# 2.Confidence
# 
# 3.Lift

# In[7]:


transaction_list = []
for rows in range(0,data.shape[0]):
    transaction_list.append([str(data.values[rows,col]) for col in range(0,data.shape[1])])  


# In[8]:


# for i in range(0,len(transaction_list)):
#     print(transaction_list[0])


# In[9]:


result = apriori(transaction_list,min_support = 0.003, min_confidence = 0, min_lift = 3, min_length = 2)


# In[10]:


# Association rules 

rules = list(result)
for i in range(0,len(rules)):
    rules[i]


# In[11]:


rules[1]


# In[12]:


rules[2]


# In[13]:


print("There are three major components of Apriori algorithm:")


# 1. Support(B) = (Transactions containing (B))/(Total Transactions) 
# 
#     Support(Ketchup) = (Transactions containingKetchup)/(Total Transactions)
# 
#     Support(Ketchup) = 100/1000  
#                      = 10%
# 
# 2.Confidence(A→B) = (Transactions containing both (A and B))/(Transactions containing A)
# 
#     Confidence(Burger→Ketchup) = (Transactions containing both (Burger and Ketchup))/(Transactions containing A)
# 
#     Confidence(Burger→Ketchup) = 50/150  
#                                = 33.3%
#         
# 3.Lift(A→B) = (Confidence (A→B))/(Support (B))  
#     
#     Lift(Burger→Ketchup) = (Confidence (Burger→Ketchup))/(Support (Ketchup))
# 
#     Lift(Burger→Ketchup) = 33.3/10  
#                          = 3.33

# In[14]:


for item in rules:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("_________________________________________________________________________________________________")

