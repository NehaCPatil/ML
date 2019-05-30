#!/usr/bin/env python
# coding: utf-8

# #  Large Movie Review Dataset
# 
# We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. 
# There is additional unlabeled data for use as well. 
# Raw text and already processed bag of words formats are provided
# 

# In[16]:


import nltk
import random
import pickle
from nltk.corpus import movie_reviews


# #### Converting words to Features with NLTK

# In[17]:


documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(documents):
    words = tuple(documents)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

features = find_features(documents)


# In[18]:


# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))


# In[19]:


featuresets = [(find_features(rev), category) for (rev, category) in documents]


# ### Naive Bayes Classifier with NLTK

# In[22]:


# set that we'll train our classifier with
training_set = featuresets[:1900]

# set that we'll cross validation our classifier with
cross_validation=training_set[:400]

# set that we'll test against.
testing_set = featuresets[1900:]


# In[23]:


classifier = nltk.NaiveBayesClassifier.train(training_set)


# In[24]:


print("Classifier accuracy percent training_set:",(nltk.classify.accuracy(classifier, cross_validation))*100)


# In[25]:


classifier.show_most_informative_features(15)


# ### Saving Classifiers with NLTK

# In[26]:


save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


# ### Load model

# In[27]:


classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()


# In[28]:


print("Classifier accuracy percent testing_set:",(nltk.classify.accuracy(classifier, testing_set))*100)


# In[15]:


classifier.show_most_informative_features(15)

