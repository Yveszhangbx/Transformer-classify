#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras


# In[6]:


class LSTM_Model(keras.Model):
    def __init__(self, d_model,size_t, size_c,size_i, seq_len, rate=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.embedding_t = keras.layers.Embedding(size_t, 10)
        self.embedding_c = keras.layers.Embedding(size_c, 10)
        self.embedding_i = keras.layers.Embedding(size_i, 100)
        self.convt = keras.layers.Dense(self.d_model)
        
        self.lstm1 = tf.keras.layers.LSTM(seq_len,return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(seq_len,return_sequences=True)
        self.lstm3 = tf.keras.layers.LSTM(seq_len,return_sequences=False)

        self.dropout = keras.layers.Dropout(rate)
        self.dense = keras.layers.Dense(2, activation="softmax")
        
    def call(self, t,c,i, training, mask):
        t = self.embedding_t(t)
        c = self.embedding_c(c)
        i = self.embedding_i(i)
        tc = tf.concat([t,c], 2)
        tci = tf.concat([tc,i], 2)
        x = self.convt(tci)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        
        x = self.dropout(x, training=training)
        
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.lstm3(x)
        x = self.dense(x)
        
        return x