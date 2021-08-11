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


# In[2]:


import itertools
def plot_confusion_matrix(cm, classes,normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="blue" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[3]:


def perf_measure(y_actual, y_hat):
    y_actual=np.array(y_actual)
    y_hat=np.array(y_hat)
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i] and y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)


# In[4]:


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


# In[5]:


print(train_file[(train_file.category == 'Positive ')].index)


# In[6]:


train_file["category"][32852] = "Positive"


# In[7]:


df_train_reviews = pd.DataFrame(train_file.text)
df_test_reviews = pd.DataFrame(test_file.text)
df_train_score = pd.DataFrame(train_file.category)
df_test_score = pd.DataFrame(test_file.category)


# In[8]:


print(df_train_reviews.shape,df_test_reviews.shape)


# In[ ]:





# In[ ]:





# In[9]:


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
    


# In[10]:


corpus_train = df_train_reviews.text.apply(preprocessing)
corpus_test = df_test_reviews.text.apply(preprocessing)


# In[11]:


corpus_test 


# In[ ]:





# In[12]:


from sklearn.feature_extraction.text import TfidfVectorizer
#cv = TfidfVectorizer(ngram_range=(1,2))
cv = TfidfVectorizer(max_features=1500)
X_train = cv.fit_transform(corpus_train).toarray()
X_test = cv.transform(corpus_test).toarray()
X_train.shape


# In[13]:


type(df_test_score)


# In[14]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler,MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:





# In[ ]:





# # Naive Bayes

# In[22]:


#Naive bayes classification
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB(alpha = 1 , fit_prior=True, class_prior=None)
classifier.fit(X_train, df_train_score)
y_pred = classifier.predict(X_test)


# In[23]:


print(classification_report(df_test_score,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




