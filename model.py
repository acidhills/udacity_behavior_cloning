import cv2
import tensorflow as tf
import numpy as np
from keras.utils import Sequence
from keras.applications import InceptionV3
from keras.layers import Input, Lambda, Dense, Flatten, Dropout
from keras.models import Model
from keras import backend
from keras.optimizers import RMSprop
import csv
import os
from sklearn.model_selection import train_test_split
from utils.data_generator import generator as image_generator
from math import ceil
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
from keras.metrics import MAPE

from utils.clr import OneCycleLR, LRFinder
from utils.inception import get_inception
print(device_lib.list_local_devices())
print(len(backend.tensorflow_backend._get_available_gpus()) > 0)


LR_SEARCH = False
MODEL_NAME = 'inception.frozen'
SAVE_MODEL = True
DATA_DIRECTORY = '../opt/carnd_p3/data/'
# DATA_DIRECTORY = 'data/'
BATCH_SIZE = 64
FREEZE = True
EPOCHS = 3
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
# xcpy = x[...,::-1,:]
#
# plt.figure()
# plt.imshow(cv2.cvtColor(x[0], cv2.COLOR_BGR2RGB))
#
# plt.figure()
# plt.imshow(cv2.cvtColor(xcpy[0], cv2.COLOR_BGR2RGB))

# print(x[0].shape)
# image shape (160, 320, 3)

model = get_inception(FREEZE)

if LR_SEARCH:
    model.compile(optimizer='RMSProp', loss='mse', metrics=[MAPE()])
    minimum_lr = 0.0001
    maximum_lr = 0.1
    lr_callback = LRFinder(len(train_samples), BATCH_SIZE,
                           minimum_lr, maximum_lr,
                           lr_scale='exp', save_dir='lr_finder')
    model.fit_generator(train_generator,
                        steps_per_epoch=ceil(len(train_samples) / BATCH_SIZE),
                        validation_data=validation_generator,
                        validation_steps=ceil(len(validation_samples) / BATCH_SIZE),
                        epochs=1, verbose=1,callbacks=[lr_callback])
    LRFinder.plot_schedule_from_file('lr_finder')


else:

    # lr = 0.001
    # wd = 0.001
    # optimizer = RMSprop(lr=lr, decay=wd)
    model.compile(optimizer='RMSProp', loss='mse', metrics=[MAPE])
    lr_manager = OneCycleLR(len(train_samples),
                            batch_size=BATCH_SIZE,
                            max_lr=0.001,
                            end_percentage=0.1, scale_percentage=None,
                            maximum_momentum=None, minimum_momentum=None)
    history_object = model.fit_generator(train_generator,
                                        steps_per_epoch=ceil(len(train_samples) / BATCH_SIZE),
                                        validation_data=validation_generator,
                                        validation_steps=ceil(len(validation_samples) / BATCH_SIZE),
                                        epochs=EPOCHS,
                                        verbose=1,
                                        callbacks=[lr_manager])

    if SAVE_MODEL:
        model.save('model.h5')
        model.save(f'model.{MODEL_NAME}.h5')


    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig(f'{MODEL_NAME}.png')



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
