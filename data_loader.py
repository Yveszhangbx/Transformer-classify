#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
import tensorflow as tf
import glob
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib 

#Tokenize and pad the data
def to_token(path,size,maxlen,tokenized,tok_name='xxx'):
    data = []
    for file in path:
        with open(file,'r') as f:
            l=f.readlines()
            new = [line.replace('\n','').split(',') for line in l]
            data+=new
    
    #Build a tokenizer from scratch
    if not tokenized:       
        tok = Tokenizer(num_words=size+1,oov_token='UNK')
        tok.fit_on_texts(data)
        joblib.dump(tok,tok_name+'.pkl')
    
    #Use the built tokenizer
    else:
        tok = joblib.load(tok_name+'.pkl') 
        
    dt = tok.texts_to_sequences(data)
    padded_dt = pad_sequences(dt, maxlen=maxlen)
    return padded_dt

#Load the data and split the dataset
def data_loader(data_path,batch_size,maxlen,
                size_t=0,size_c=0,size_i=0,
                tokenized = False,
                split=True):
    
    time_path = glob.glob(data_path+"/time*.txt")
    cat_path = glob.glob(data_path+"/cat*.txt")
    idx_path = glob.glob(data_path+"/idx*.txt")
    real_path = glob.glob(data_path+"/y*.txt")
    
    real = []
    
    for file in real_path:
        with open(file,'r') as f:
            l=f.readlines()
            for line in l:
                real+= [[int(item)] for item in line.replace('\n','').split(',')]
    
    padded_ts = to_token(time_path,size_t,maxlen,tokenized,tok_name=data_path+'/ts')
    padded_cat = to_token(cat_path,size_c,maxlen,tokenized,tok_name=data_path+'/cat')
    padded_idx = to_token(idx_path,size_i,maxlen,tokenized,tok_name=data_path+'/idx')
    
    dataset = tf.data.Dataset.from_tensor_slices((padded_ts, padded_cat,padded_idx,real))
    dataset = dataset.batch(batch_size)
    dataset.shuffle(10000)
    
    if split:
        length_train = int(0.9*len(dataset))
        trainset = dataset.take(length_train)
        validset = dataset.skip(length_train)

        return trainset,validset
    
    else:
        return testdataset

