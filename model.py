import tensorflow as tf
import numpy as np
from keras.utils import Sequence
from keras.applications import InceptionV3
from keras.layers import Input, Lambda, Dense, Flatten, Dropout
from keras.models import Model
from keras import backend
import csv
import os
from sklearn.model_selection import train_test_split
from utils.data_generator import generator as image_generator
from math import ceil
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print(len(backend.tensorflow_backend._get_available_gpus()) > 0)


MODEL_NAME = 'inception.frozen'
SAVE_MODEL = True
DATA_DIRECTORY = '../opt/carnd_p3/data/'
# DATA_DIRECTORY = 'data/'
BATCH_SIZE = 256
FREEZE = True
EPOCHS = 1
samples = []

with open(DATA_DIRECTORY + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
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
cropped_input = Lambda(lambda x: tf.image.crop_to_bounding_box(x,60,0,80,320))(model_input)
resized_input = Lambda(lambda image: tf.image.resize_images(image, (139, 139)))(cropped_input)
normalized_input = Lambda(lambda x: x / 127.5 - 1.)(resized_input)
inp = inception(normalized_input)
flat = Flatten()(inp)
drop = Dropout(0.5)(flat)
dens1 = Dense(512, activation='relu')(drop)
dens2 = Dense(512, activation='relu')(dens1)
predictions = Dense(1, activation='linear')(dens2)
model = Model(inputs=model_input, outputs=predictions)
model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

history_object = model.fit_generator(train_generator,
                    steps_per_epoch=ceil(len(train_samples) / BATCH_SIZE),
                    validation_data=validation_generator,
                    validation_steps=ceil(len(validation_samples) / BATCH_SIZE),
                    epochs=EPOCHS, verbose=1)

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
plt.savefig(f'{MODEL_NAME}.png')

if SAVE_MODEL:
    model.save('model.h5')
    model.save(f'model.{MODEL_NAME}.h5')


# Check the summary of this new model to confirm the architecture
# print(model.summary())
# x, y =  next(train_generator)
# # print(x[0].shape)
# print(model.predict(np.expand_dims(x[0], axis=0)))



# from cv2 import imshow,waitKey
# x, y =  next(train_generator)
# # a = tf.make_tensor_proto(tf.image.crop_to_bounding_box(x[0],60,0,70,320))
#
# a = tf.image.crop_to_bounding_box(x[0],60,0,70,320).numpy()
# b = tf.image.crop_to_bounding_box(x[0],60,0,80,320).numpy()
# imshow('image',a)
# imshow('image orig', x[0])
# imshow('image crop2', b)
# waitKey(0)
