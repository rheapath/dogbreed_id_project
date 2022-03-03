import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dogs_train = pd.read_csv('labels.csv', names=['id', 'breed'])
dogs_train['id'] = dogs_train['id'].apply(lambda x : x + '.jpg')

EPOCHS = 1
BATCH_SIZE = 2
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
NUM_CLASSES = 120

generator = ImageDataGenerator(rescale=1./255.)

train_generator = generator.flow_from_dataframe(
    dataframe=dogs_train,
    directory='./train/',
    x_col='id',
    y_col='breed',
    batch_size=BATCH_SIZE,
    subset='training',
    color_mode='grayscale',
    target=(IMAGE_WIDTH, IMAGE_HEIGHT),
    interpolation='nearest',
    save_to_dir='./generated/',
    save_prefix='',
    save_format='jpg')

print(train_generator.class_indices)
exit(0)

model = tf.keras.Sequential([
#    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
#    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
#    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

model.fit(train_generator, epochs=EPOCHS)
model.save('saved_model')

