#!/usr/bin/env python
# coding: utf-8

# In[38]:


import os
import sys
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
import copy

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


# In[39]:

#LOAD WORD2VECTOR
def load_embeddings(path):
    with open(path,'rb') as f:
        emb_arr = pickle.load(f)
    return emb_arr


# In[40]:

#BASIC CONFIGURATION FOR PROCESS TEXT
BASE_DIR = ''
FASTTEXT_EMBD = os.path.join(BASE_DIR, '/home/ec2-user/SageMaker/CX/word2vec/fasttext_300d.pkl')
TEXT_DATA = os.path.join(BASE_DIR, '/home/ec2-user/SageMaker/CX/text_classfication_claims/processed_data/processed_claim.pkl')
MAX_SEQUENCE_LENGTH = 50
MAX_NUM_WORDS = 3000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.3


# In[41]:

#LOAD FASTTEXT WORD2VECTOR WHICH IS 300D
def load_fasttext(word_index,wv):
    
    x = list(tokenizer.word_counts.items())
    s = sorted(x,key=lambda p:p[1],reverse=True)
    word_index = tokenizer.word_index # get real size of vocab
    small_word_index = copy.deepcopy(word_index) # avoid size is going to change
    print("Removing less freq words from word-index dict...")
    for item in s[num_words-1:]:
        small_word_index.pop(item[0])
    print("Finished!")
    print(len(small_word_index))
    print(len(word_index))

    embedding_matrix = np.random.uniform(size=(num_words,EMBEDDING_DIM))
    for word, i in small_word_index.items():
        try:
            embedding_vector = wv[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            #print("Word: [",word,"] not in wvmodel! Use random embedding instead.")
            continue 
    return embedding_matrix


# In[42]:

#SHOW HOW MANY VECTORS THIS PRE-TRAINED WORD EMBEDDING HAS
embeddings_index = {}
embeddings_index=load_embeddings(FASTTEXT_EMBD)
print('Found %s word vectors.' % len(embeddings_index))


# In[43]:

#LOAD PRE-PROCESSED TEXT FROM PICKLE, USE JAPANAESE TOKENIZER TO SPLIT WORD AS WELL AS NORMALIZE THEM
processed_df=pd.read_pickle(TEXT_DATA)
print('Shape of data tensor:', processed_df.shape)


# In[44]:


texts = []  # list of text samples
labels_nums=0 #numbers of lables
labels = []  # list of label ids

#INPUT AND TARGET FOR TRAINING
texts=processed_df["comment"].values
labels=processed_df[processed_df.columns[:-1]].values


# In[45]:

#CREATE TOkENIZER
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#ADD PAD FOR EACH ENTRY
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels_nums=labels.shape[1]

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])

#FIX RANDOM
np.random.seed=2019
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]


# In[46]:

#LOADING PRETRAINED VERCTOR
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = load_fasttext(word_index,embeddings_index)
print('Found %s word vectors.' % len(embedding_matrix))


# In[47]:


train_dir = os.path.join(os.getcwd(), 'data/train')
os.makedirs(train_dir, exist_ok=True)

val_dir = os.path.join(os.getcwd(), 'data/val')
os.makedirs(val_dir, exist_ok=True)

embedding_dir = os.path.join(os.getcwd(), 'data/embedding')
os.makedirs(embedding_dir, exist_ok=True)

#STORE TRAIN AND VAL DATA AS NP FOTMAT
np.save(os.path.join(train_dir, 'x_train.npy'), x_train)
np.save(os.path.join(train_dir, 'y_train.npy'), y_train)
np.save(os.path.join(val_dir, 'x_val.npy'), x_val)
np.save(os.path.join(val_dir, 'y_val.npy'), y_val)
np.save(os.path.join(embedding_dir, 'embedding.npy'), embedding_matrix)


# In[48]:


#check if current server has docker-compose installed
get_ipython().system('/bin/bash /home/ec2-user/SageMaker/setup.sh')


# In[16]:


import sagemaker
from sagemaker.tensorflow import TensorFlow

model_dir = '/opt/ml/model'
train_instance_type = 'local'
hyperparameters = {'epochs': 1, 
                   'batch_size': 128, 
                   'num_words': num_words,
                   'word_index_len': len(word_index),
                   'labels_index_len': labels_nums,
                   'embedding_dim': EMBEDDING_DIM,
                   'max_sequence_len': MAX_SEQUENCE_LENGTH
                  }

local_estimator = TensorFlow(entry_point='train.py',
                       source_dir='code',
                       model_dir=model_dir,
                       train_instance_type=train_instance_type,
                       train_instance_count=1,
                       hyperparameters=hyperparameters,
                       role=sagemaker.get_execution_role(),
                       base_job_name='tf-txt-classfication-claims',
                       framework_version='1.13',
                       py_version='py3',
                       script_mode=True)


# In[17]:


inputs = {'train': f'file://{train_dir}',
          'val': f'file://{val_dir}',
          'embedding': f'file://{embedding_dir}'}

local_estimator.fit(inputs)


# In[18]:


local_predictor = local_estimator.deploy(initial_instance_count=1,instance_type='local')


# In[19]:


local_results = local_predictor.predict(x_val[:10])['predictions']


# In[20]:


print('predictions: \t{}'.format(np.argmax(local_results, axis=1)))
print('target values: \t{}'.format(np.argmax(y_val[:10], axis=1)))


# In[23]:


local_predictor.delete_endpoint()


# In[24]:


s3_prefix = 'tf-txt-classfication-claims'

traindata_s3_prefix = '{}/data/train'.format(s3_prefix)
valdata_s3_prefix = '{}/data/val'.format(s3_prefix)
embeddingdata_s3_prefix = '{}/data/embedding'.format(s3_prefix)

train_s3 = sagemaker.Session().upload_data(path='./data/train/', key_prefix=traindata_s3_prefix)
val_s3 = sagemaker.Session().upload_data(path='./data/val/', key_prefix=valdata_s3_prefix)
embedding_s3 = sagemaker.Session().upload_data(path='./data/embedding/', key_prefix=embeddingdata_s3_prefix)

inputs = {'train':train_s3, 'val': val_s3, 'embedding': embedding_s3}
print(inputs)


# In[27]:


train_instance_type = 'ml.m4.xlarge'
hyperparameters = {'epochs': 1, 
                   'batch_size': 128, 
                   'num_words': num_words,
                   'word_index_len': len(word_index),
                   'labels_index_len': labels_nums,
                   'embedding_dim': EMBEDDING_DIM,
                   'max_sequence_len': MAX_SEQUENCE_LENGTH
                  }

estimator = TensorFlow(entry_point='train.py',
                       source_dir='code',
                       model_dir=model_dir,
                       train_instance_type=train_instance_type,
                       train_instance_count=1,
                       hyperparameters=hyperparameters,
                       role=sagemaker.get_execution_role(),
                       base_job_name='tf-txt-classfication-claims',
                       framework_version='1.13',
                       py_version='py3',
                       script_mode=True)


# In[28]:


estimator.fit(inputs)


# In[ ]:




