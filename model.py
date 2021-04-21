import tensorflow as tf
import numpy as np
from keras.utils import Sequence
from keras.applications import InceptionV3
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
import csv
import os
from sklearn.model_selection import train_test_split
from utils.data_generator import generator as image_generator
from math import ceil

DATA_DIRECTORY = '../opt/carnd_p3/data/'
BATCH_SIZE = 128
FREEZE = True
EPOCHS = 5
samples = []

with open(DATA_DIRECTORY + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = image_generator(train_samples, DATA_DIRECTORY, batch_size=BATCH_SIZE)
validation_generator = image_generator(validation_samples, DATA_DIRECTORY, batch_size=BATCH_SIZE)

# x, y =  next(train_generator)
# print(x[0].shape)
# image shape (160, 320, 3)

weights_flag = 'imagenet'
inception = InceptionV3(weights=weights_flag, include_top=False,
                        input_shape=(139, 139, 3))

if FREEZE:
    for l in inception.layers:
        l.trainable = False

model_input = Input((160, 320, 3))
resized_input = Lambda(lambda image: tf.image.resize_images(image, (139, 139)))(model_input)
normalized_input = Lambda(lambda x: x / 127.5 - 1.)(resized_input)
inp = inception(normalized_input)
flat = Flatten()(inp)
dens1 = Dense(512, activation='relu')(flat)
predictions = Dense(1)(dens1)
model = Model(inputs=model_input, outputs=predictions)
model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

model.fit_generator(steps_per_epoch=ceil(len(train_samples) / BATCH_SIZE),
                    validation_data=validation_generator,
                    validation_steps=ceil(len(validation_samples) / BATCH_SIZE),
                    epochs=EPOCHS, verbose=1)

model.save('model.h5')
model.save('model.inception.frozen.h5')
# Check the summary of this new model to confirm the architecture
# print(model.summary())
# x, y =  next(train_generator)
# # print(x[0].shape)
# print(model.predict(np.expand_dims(x[0], axis=0)))
