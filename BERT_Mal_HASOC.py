#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install ktrain')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import tensorflow as tf
import pandas as pd
import numpy as np
import ktrain
from ktrain import text
import tensorflow as tf


# In[ ]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())

print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")


# In[ ]:


data_test = pd.read_csv('/content/drive/MyDrive/Datasets/malayalam_hasoc_dev.tsv',sep="\t",names=["ID","Tweets","Label"])
data_train = pd.read_excel('/content/drive/MyDrive/Datasets/Malayalam__hasoc_train.xlsx',names=["ID","Tweets","Label"])


# In[ ]:


print("Size of train dataset: ",data_train.shape)
print("Size of test dataset: ",data_test.shape)


# In[ ]:


data_train = data_train.dropna()
data_train = data_train.reset_index(drop=True)

data_test = data_test.dropna()
data_test = data_test.reset_index(drop=True)


# In[ ]:


data_test.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


data_train.Label.value_counts()


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt
classes = data_train.Label.unique()
counts = []

for i in classes:
  count = len(data_train[data_train.Label==i])
  counts.append(count)

plt.bar(classes, counts)
plt.show()


# In[ ]:


data_train.isnull().sum()


# In[ ]:


data_train.Label.unique()


# In[ ]:


data_train.Label.value_counts()/data_train.shape[0]*100


# In[ ]:


(X_train, y_train), (X_test, y_test), preproc = text.texts_from_df(train_df=data_train,
                                                                   text_column = 'Tweets',
                                                                   label_columns = 'Label',
                                                                   val_df = data_test,
                                                                   maxlen = 150,
                                                                   preprocess_mode = 'bert')


# In[ ]:





# In[ ]:


model = text.text_classifier(name = 'bert',
                             train_data = (X_train, y_train),
                             preproc = preproc)


# In[ ]:


learner = ktrain.get_learner(model=model, train_data=(X_train, y_train),
                   val_data = (X_test, y_test),
                   batch_size = 30)


# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


learner.fit_onecycle(lr = 2e-5, epochs = 1)
predictor = ktrain.get_predictor(learner.model, preproc)
pred = predictor.predict(data_test.Tweets.tolist())
print(classification_report(data_test.Label.tolist(), pred))

#Corrected the error of wrong label in next cell...model training is done


# In[ ]:


pred = predictor.predict(data_test.Tweets.tolist())
print(classification_report(data_test.Label.tolist(), pred))


# In[ ]:





# In[ ]:


learner.fit_onecycle(lr = 1e-5, epochs = 1)
predictor = ktrain.get_predictor(learner.model, preproc)
pred = predictor.predict(data_test.Tweets.tolist())
print(classification_report(data_test.Label.tolist(), pred))


# In[ ]:





# In[ ]:


learner.fit_onecycle(lr = 1e-5, epochs = 1)
predictor = ktrain.get_predictor(learner.model, preproc)
pred = predictor.predict(data_test.Tweets.tolist())
print(classification_report(data_test.Label.tolist(), pred))


# In[ ]:





# In[ ]:


learner.fit_onecycle(lr = 1e-7, epochs = 1)
predictor = ktrain.get_predictor(learner.model, preproc)
pred = predictor.predict(data_test.Tweets.tolist())
print(classification_report(data_test.Label.tolist(), pred))


# In[ ]:





# In[ ]:


learner.fit_onecycle(lr = 1e-5, epochs = 2)
predictor = ktrain.get_predictor(learner.model, preproc)
pred = predictor.predict(data_test.Tweets.tolist())
print(classification_report(data_test.Label.tolist(), pred))


# In[ ]:





# In[ ]:


learner.fit_onecycle(lr = 1e-5, epochs = 2)
predictor = ktrain.get_predictor(learner.model, preproc)
pred = predictor.predict(data_test.Tweets.tolist())
print(classification_report(data_test.Label.tolist(), pred))


# In[ ]:




