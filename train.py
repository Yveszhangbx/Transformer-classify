#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import sys
import time
import datetime
import tensorflow as tf
from tensorflow import keras
from Transformer import *
from data_loader import *
from config import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# In[35]:


DATA_PATH = sys.argv[1]


# In[47]:


BASE_DIR = os.getcwd()
def get_path(relative_path):
    path = os.path.join(BASE_DIR, relative_path)
    return path


# In[26]:


#Mask the padded part when calculate attention
def create_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)# [batch_size, 1, 1, seq_len]
    return seq[:, tf.newaxis, tf.newaxis, :]


# In[37]:


#Learning rate scheduler
class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

#Define the calcuation of loss
def loss_function(real, pred, loss_object):
    loss_ = loss_object(real, pred)
    return tf.reduce_mean(loss_)

#Define the validate step
def validate(transformer, dataset):
    print("*** Running Validation step ***")
    accuracy = keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
    
    for (batch, (t,c,i,real)) in enumerate(dataset):
        mask = create_mask(t)
        predictions = transformer(t,c,i,
                                     training=False,
                                     mask=mask)
        
        accuracy(real, predictions)
        
        if batch % 100 == 0:
            print('Batch {} Accuracy {:.4f}'.format(batch, accuracy.result()))
            
    print('\nVal Accuracy {}\n'.format(accuracy.result()))


# In[50]:


def train():
    # prepare dataset
    train_dataset, val_dataset = data_loader(DATA_PATH,BATCH_SIZE,MAXLEN,size_t,size_c,size_i)

    #Define the model
    d_model = D_MODEL
    transformer = Transformer(
    num_layers = LAYER_NUM,
    d_model = D_MODEL,
    num_heads = HEAD_NUM,
    dff= 4*D_MODEL,
    size_t=size_t,
    size_c=size_c,
    size_i=size_i,
    seq_len=MAXLEN
    )
    
    # Define the optimizer
    learning_rate = CustomSchedule(d_model)
    optimizer = keras.optimizers.Adam(learning_rate, beta_1=beta_1,
                                      beta_2=beta_2, epsilon=epsilon)
    
    #Define the type of loss and accuracy
    loss_object = keras.losses.SparseCategoricalCrossentropy(reduction='none')   
    train_loss = keras.metrics.Mean(name='train_loss')
    train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
       
    # Define checkpoints manager
    checkpoint_path = get_path("checkpoints")
    ckpt = tf.train.Checkpoint(model=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    
    # Visualize with Tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = get_path('logs/gradient_tape/' + current_time + '/train')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    @tf.function
    def train_step(t,c,i,real):
        mask = create_mask(t)
    
        with tf.GradientTape() as tape:
            predictions = transformer(t,c,i,
                                         training=True,
                                         mask=mask)
            loss = loss_function(real, predictions, loss_object)
            
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        
        train_loss(loss)
        train_accuracy(real, predictions)
        
    #------------------------
    #strat training
    #------------------------
    for epoch in range(EPOCH):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        
        for (batch, (t,c,i,real)) in enumerate(train_dataset):
            train_step(t,c,i,real)
            
            # record loss and accuracy
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
            
            if batch % 100 == 0:
                print('Epochs {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch+1, batch, train_loss.result(), train_accuracy.result()))
        
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
            
        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
        
        # Perform validation
        validate(transformer, val_dataset)
    
    
if __name__ == '__main__':

    train()

