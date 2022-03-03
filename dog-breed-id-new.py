#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

train_directory = "Downloads/dog-breed-identification/train"
test_directory = "Downloads/dog-breed-identification/test"

train_labels = pd.read_csv("Downloads/dog-breed-identification/labels.csv", dtype=str)
test_labels = pd.read_csv("Downloads/dog-breed-identification/sample_submission.csv", dtype=str)

train_labels['id'] = train_labels['id'].apply(lambda x : x + '.jpg')
test_labels['id'] = test_labels['id'].apply(lambda x : x + '.jpg')


# In[2]:


train_labels.head(5) # testing to see if .jpg extension has been applied. 


# In[3]:


# mayhaps some dog pics?

import matplotlib.pyplot as plt
import os
import sys
get_ipython().run_line_magic('matplotlib', 'inline')

subclass = os.listdir(train_directory)

img = plt.figure(figsize=(10,5))
for e in range(len(subclass[:8])):
    plt.subplot(2, 4, e+1)
    dog = plt.imread(os.path.join(train_directory, subclass[e]))
    plt.imshow(dog, cmap=plt.get_cmap('gray'))


# In[4]:


EPOCHS = 10
BATCH_SIZE = 32
TARGET_SIZE = (256, 256)

train_generator = ImageDataGenerator(rescale = 1./255., 
                                    validation_split = 0.05)

dogs_train = train_generator.flow_from_dataframe(dataframe = train_labels, 
                                                directory = train_directory, 
                                                x_col = "id", 
                                                y_col = "breed", 
                                                subset = "training", 
                                                batch_size = BATCH_SIZE, 
                                                seed = 42, 
                                                shuffle = True, 
                                                class_mode = "categorical", 
                                                target_size = TARGET_SIZE, 
                                                interpolation = "nearest",
                                                save_to_dir = "Downloads/dog-breed-identification/generated",
                                                save_prefix = "", 
                                                save_format = "jpg",
                                                color_mode = "rgb")


# In[5]:


x, y = next(dogs_train)
print(x.shape)
print(y.shape)


# In[6]:


validation_generator = train_generator.flow_from_dataframe(dataframe = train_labels, 
                                                          directory = train_directory, 
                                                          x_col = "id", 
                                                          y_col = "breed", 
                                                          subset = "validation",
                                                          batch_size = BATCH_SIZE, 
                                                          seed = 42, 
                                                          shuffle = True, 
                                                          class_mode = "categorical", 
                                                          target_size = TARGET_SIZE, 
                                                          color_mode = "rgb")


# In[7]:


test_generator = train_generator.flow_from_dataframe(dataframe = test_labels,
                                                     directory = test_directory,
                                                     x_col = "id",
                                                     y_col = None,
                                                     batch_size = BATCH_SIZE,
                                                     seed = 42,
                                                     shuffle = False,
                                                     class_mode = None,
                                                     target_size = TARGET_SIZE,
                                                     color_mode = "rgb")


# In[8]:


from keras.applications.inception_resnet_v2 import InceptionResNetV2

baseline = InceptionResNetV2(include_top = False,
                             weights ='imagenet',
                             input_shape = (256, 256, 3))
baseline.trainable = False


# In[9]:


from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Dropout,Conv2D,GlobalAveragePooling2D,Flatten

model = Sequential()
model.add(baseline)
model.add(GlobalAveragePooling2D())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(120,activation = 'softmax'))

model.summary()


# In[10]:


callback = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=2),
           tf.keras.callbacks.ModelCheckpoint("MyModel.h5",save_best_only =True,verbose =2)]


# In[16]:


STEPS_PER_EPOCH = 9711 / BATCH_SIZE # = 303.468
#print(STEPS_PER_EPOCH)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics = ['accuracy'])

model.fit(dogs_train, epochs = 15, steps_per_epoch = 300)


# In[ ]:


score = model.evaluate(validation_generator, verbose = 0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


# In[ ]:


print(score)


# In[19]:


model.save("Downloads/model")


# In[20]:


model.predict(test_generator)


# In[ ]:




