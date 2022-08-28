#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import pandas as pd 
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from google.protobuf import text_format
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('caravan-challenge.csv')


# In[3]:


df.head()


# In[4]:


from sklearn.model_selection import train_test_split
X = df.drop('CARAVAN', axis=1)
y = df['CARAVAN']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, random_state=42
)


# In[5]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[43]:


# Use a multi-GPU strategy to reduce training time (if there are GPUs available)

strategy = tf.distribute.MirroredStrategy()
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA


# In[45]:


# Converting pandas datasets to tensorflow datasets

batch_size=25
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.with_options(options)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.with_options(options)
train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)


# In[54]:


import tensorflow as tf
tf.random.set_seed(42)
epochs=150
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
        
    ]
)
history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs)


# In[56]:


# Saving the model

model.save('insurance_model.h5')


# In[57]:


# Loading the mode

model = keras.models.load_model('cardio_model.h5')


# In[58]:


# Plotting the learning process

def plotLearningCurve(history,epochs):
  epochRange = range(1,epochs+1)
  plt.figure(figsize = (12,6))
  plt.plot(epochRange,history.history['accuracy'])
  plt.plot(epochRange,history.history['val_accuracy'])
  plt.title('Model Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(['Train','Validation'],loc='upper left')
  plt.show()

  plt.figure(figsize = (12,6))
  plt.plot(epochRange,history.history['loss'])
  plt.plot(epochRange,history.history['val_loss'])
  plt.title('Model Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(['Train','Validation'],loc='upper left')
  plt.show()


# In[59]:


plotLearningCurve(history,150)


# In[ ]:





# In[ ]:




