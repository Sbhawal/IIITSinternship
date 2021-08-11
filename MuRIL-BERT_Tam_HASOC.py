#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install bert-for-tf2')
get_ipython().system('pip install sentencepiece')


# In[ ]:


get_ipython().system('nvidia-smi')


# ##Imports

# In[ ]:


import tensorflow as tf
import tensorflow_hub as hub
import bert
from tensorflow.keras.models import  Model
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn import preprocessing
from bert import bert_tokenization
print("TensorFlow Version:",tf.__version__)
print("Hub version: ",hub.__version__)


# ##Reading Datasets and droping nan values

# In[ ]:


df_train = pd.read_excel('/content/drive/MyDrive/Datasets/Tamil__hasoc_train.xlsx',names=["ID","Tweets","Labels"])
df_train.dropna(inplace=True)
df_train.reset_index(drop=True, inplace=True)
df_val = pd.read_csv("/content/drive/MyDrive/Datasets/Tamil_hasoc_dev.tsv", sep="\t",names=["ID","Tweets","Labels"])
df_val.dropna(inplace=True)
df_val.reset_index(drop=True, inplace=True)


# In[ ]:





# In[ ]:


df_test = pd.read_csv("/content/drive/MyDrive/Datasets/tamil__hasoc_test.tsv", sep="\t",names=["ID","Tweets"])
# df_test.dropna(inplace=True)
# df_test.reset_index(drop=True, inplace=True)
df_test.shape


# In[ ]:


df_val.head()


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


df_train.head(10)


# In[ ]:


df_test.head(10)


# ##Preprocessing
# 
# 
# 
# 

# In[ ]:


import re, string
regex = re.compile('[%s]' % re.escape(string.punctuation))
    
def preprocessing(document):
  document = str(document)
  # document = re.sub(r'[0-9]', ' ', document)
  document = document.replace("@"," ")
  # document = regex.sub(' ', document)
  # document = document.lower()
  # document = re.sub(' +', ' ', document)
  tokens = document.split()
  preprocessed_text = ' '.join(tokens)
  return preprocessed_text
  


# In[ ]:


df_train.Tweets =  df_train.Tweets.apply(preprocessing)
df_val.Tweets =  df_val.Tweets.apply(preprocessing)
df_test.Tweets =  df_test.Tweets.apply(preprocessing)


# In[ ]:


df_train.head(10)


# In[ ]:





# ##Mapping the labels correctly 

# In[ ]:


df_train.Labels = df_train.Labels.map({'not': 'NOT', 'OFf': 'OFF','NOT': 'NOT', 'OFF': 'OFF'})


# In[ ]:


df_train.head(10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


df_val.Labels.unique()


# ##Label Encoding to 0 and 1

# In[ ]:


unique_labels = list(np.unique(df_train["Labels"]))

train_x = df_train["Tweets"].values
train_y = df_train["Labels"].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
# le = preprocessing.LabelEncoder()

train_y = le.fit_transform(train_y)
train_y = tf.keras.utils.to_categorical(train_y, num_classes=len(unique_labels), dtype='float32')

val_x = df_val["Tweets"].values
val_y = df_val["Labels"].values

val_y = le.fit_transform(val_y)
val_y = tf.keras.utils.to_categorical(val_y, num_classes=len(unique_labels), dtype='float32')


print("number of unique labels", len(unique_labels))

test_x = df_test["Tweets"].values


# In[ ]:





# In[ ]:





# In[ ]:





# ##Helper Functions

# In[ ]:


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens,)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

# Function to create attention masks
def get_masks(tokens, max_seq_length):
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

# Function to create segment ids
def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

# Function to create input_ids, attention_masks, segment_ids for sample
def create_single_input(sentence,MAX_LEN, MAX_SEQ_LEN):
  
  stokens = tokenizer.tokenize(sentence)
  
  stokens = stokens[:MAX_LEN]
  
  stokens = ["[CLS]"] + stokens + ["[SEP]"]
 
  ids = get_ids(stokens, tokenizer, MAX_SEQ_LEN)
  masks = get_masks(stokens, MAX_SEQ_LEN)
  segments = get_segments(stokens, MAX_SEQ_LEN)

  return ids,masks,segments

def create_input_array(sentences, MAX_SEQ_LEN):

  input_ids, input_masks, input_segments = [], [], []

  for sentence in tqdm(sentences,position=0, leave=True):
  
    ids,masks,segments=create_single_input(sentence,MAX_SEQ_LEN-2, MAX_SEQ_LEN)

    input_ids.append(ids)
    input_masks.append(masks)
    input_segments.append(segments)

  return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]


# In[ ]:





# ##Downloading the MuRIL model from TFHub

# In[ ]:


muril_layer = hub.KerasLayer("https://tfhub.dev/google/MuRIL/1", trainable=True)

# Create tokenizer
vocab_file = muril_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = muril_layer.resolved_object.do_lower_case.numpy()
tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case)


# In[ ]:





# In[ ]:


max_seq_len = 100
train_x = create_input_array(train_x, max_seq_len)
val_x = create_input_array(val_x, max_seq_len)
test_x = create_input_array(test_x, max_seq_len)


# ##Defining the F1 metric

# In[ ]:


from keras import backend as K
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[ ]:





# In[ ]:


df_train.shape


# ##Establishing class weights

# In[ ]:


from sklearn.utils import class_weight
class_weights = list(class_weight.compute_class_weight('balanced',
                                             np.unique(df_train['Labels']),
                                             df_train['Labels']))

weights={}
for index, weight in enumerate(class_weights) :
  weights[index]=weight

print(weights)


# In[ ]:


#downloading model


# In[ ]:


pip install tf-models-official


# In[ ]:





# In[ ]:





# In[ ]:





# ##Defining the model

# In[ ]:


input_word_ids = tf.keras.layers.Input(shape=(100,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(100,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(100,), dtype=tf.int32,
                                    name="segment_ids")
  
outputs = muril_layer(dict(input_word_ids = input_word_ids, input_mask = input_mask, input_type_ids = segment_ids))
x = tf.keras.layers.Dropout(0.2)(outputs["pooled_output"]) # take pooled output layer
final_output = tf.keras.layers.Dense(2, activation="sigmoid", name="dense_output")(x)

model = tf.keras.models.Model(
      inputs=[input_word_ids, input_mask, segment_ids], outputs=final_output)

    
#   optimizer = 
model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  metrics=['accuracy',f1_m])



# In[ ]:


model.summary()


# ##Making checkpoint

# In[ ]:


metric = 'val_f1_m'
model_checkpoint_callback  = tf.keras.callbacks.ModelCheckpoint(filepath="/content/drive/MyDrive/Datasets/tam3", monitor=metric,
                    verbose=2, save_best_only=True, mode='max')


# In[ ]:





# ###Skipping the training as loading pretrained

# In[ ]:


# num_epochs = 15

# # Get the model object
# history = model.fit(train_x, train_y, epochs = num_epochs, batch_size = 50, validation_data = (val_x, val_y),class_weight=weights,callbacks=[model_checkpoint_callback])


# In[ ]:





# In[ ]:


# from sklearn.metrics import classification_report
# preds = model.predict(val_x)>0.5
# print(classification_report(val_y, preds))


# In[ ]:


#Reloading the model with best validation F1 score

model2 = tf.keras.models.load_model('/content/drive/MyDrive/Datasets/tam3', custom_objects={'f1_m':f1_m})


# In[ ]:


from sklearn.metrics import classification_report
preds = model2.predict(val_x)>0.5
print(classification_report(val_y, preds))


# In[ ]:


# model2.save('/content/drive/MyDrive/Datasets/tahsoc91')


# In[ ]:





# In[ ]:





# In[ ]:


# preds_test = model2.predict(test_x)>0.5


# In[ ]:


# labels = []

# for i in preds_test:
#   if i[0] == True:
#     labels.append("NOT")
#   elif i[1] == True:
#     labels.append("OFF")


# In[ ]:


# df_test["Label"] = labels


# In[ ]:


# df_test.Label.value_counts()/df_test.shape[0]*100


# In[ ]:


# df_test.head(10)


# In[ ]:


# !nvidia-smi


# In[ ]:


# df_test.to_csv("submission5.tsv", index=False,sep="\t")


# In[ ]:





# In[ ]:


# metric = 'val_f1_m'
# model_checkpoint_callback2  = tf.keras.callbacks.ModelCheckpoint(filepath="/content/drive/MyDrive/Datasets/tam2", monitor=metric,
#                     verbose=2, save_best_only=True, mode='max')


# In[ ]:


# model2.compile(loss='categorical_crossentropy',
#                   optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7),
#                   metrics=['accuracy',f1_m])

# num_epochs = 15

# # Get the model object
# history = model2.fit(train_x, train_y, epochs = num_epochs, batch_size = 50, validation_data = (val_x, val_y),class_weight=weights,callbacks=[model_checkpoint_callback2])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




