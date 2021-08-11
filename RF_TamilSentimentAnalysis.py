#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 

import gc
import chardet
import re

import seaborn as sns
import matplotlib.pyplot as plt



train_file = pd.read_csv("tamil_sentiment_full_train.tsv",sep="\t")
test_file = pd.read_csv("tamil_sentiment_full_dev.tsv",sep="\t")


# In[ ]:





# In[2]:


from sklearn.metrics import confusion_matrix
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[ ]:





# In[3]:


train_file["category"][32852] = "Positive"


# In[ ]:





# In[4]:


df_train_reviews = pd.DataFrame(train_file.text)
df_test_reviews = pd.DataFrame(test_file.text)
df_train_score = pd.DataFrame(train_file.category)
df_test_score = pd.DataFrame(test_file.category)


# In[ ]:





# In[5]:


import re, string
regex = re.compile('[%s]' % re.escape(string.punctuation))
import demoji



def preprocessing(document):
        document = str(document)
        document = demoji.replace_with_desc(document).replace(":"," ").replace("-"," ")
        document = regex.sub('', document)
        document = re.sub(r'[0-9]', '', document)

        # remove all single characters
#         document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
#         document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)


        # Converting to Lowercase
        document = document.lower()
        
#         document = re.sub(r'(.+?)\1+', r'\1', document)

        tokens = document.split()

        preprocessed_text = ' '.join(tokens)

        return preprocessed_text
    


# In[ ]:





# In[6]:


corpus_train = df_train_reviews.text.apply(preprocessing)
corpus_test = df_test_reviews.text.apply(preprocessing)


# In[ ]:





# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer
#cv = TfidfVectorizer(ngram_range=(1,2))
cv = TfidfVectorizer(max_features=1500)
X_train = cv.fit_transform(corpus_train).toarray()
X_test = cv.transform(corpus_test).toarray()
X_train.shape


# In[ ]:





# In[8]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler,MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # Random Forest

# In[9]:


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'gini', random_state = 0 , max_depth=100)
classifier.fit(X_train, df_train_score)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(df_test_score, y_pred)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(df_test_score , y_pred)
accuracy


# In[ ]:





# In[10]:


print(classification_report(df_test_score,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




