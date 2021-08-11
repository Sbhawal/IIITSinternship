#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

VOC_SIZE = 7881
max_length_sequence = 100

# tf.random.set_seed(1)
np.random.seed(1)


# In[2]:


#importing previously preprocessed and vectorized dataset from preprocessing notebook as pickle file


x = np.load('data.npy')
y = np.load("y.npy")


# In[3]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(VOC_SIZE, 32, input_length=max_length_sequence))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(10)))
model.add(tf.keras.layers.Dense(1, activation = "sigmoid"))


model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.summary()


# In[4]:


# checkpoint_cb = keras.callbacks.ModelCheckpoint("model-{epoch:02d}.h5", save_best_only=True)


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[6]:


history = model.fit(X_train, y_train, epochs = 3, batch_size=5, validation_split=0.25)


# In[7]:


import pandas as pd
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


# In[8]:


model.evaluate(X_test, y_test)


# In[9]:


pred = model.predict(X_test)>0.5


# In[11]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred))

