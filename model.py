from keras import backend
from utils.data_generator import get_generator
from math import ceil
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
from keras.metrics import MAPE

from utils.clr import OneCycleLR, LRFinder
from utils.inception import get_inception
from utils.mobilenet import get_mobilenet

from keras.models import load_model
import tensorflow as tf


print(device_lib.list_local_devices())
print(len(backend.tensorflow_backend._get_available_gpus()) > 0)

LR_SEARCH = False
MODEL_NAME = 'mobilenet.frozen'
SAVE_MODEL = True
DATA_DIRECTORIES = ['../data2/','../data3/','../opt/carnd_p3/data/', '../data/']
# DATA_DIRECTORY = 'data/'
BATCH_SIZE = 32
FREEZE = True
EPOCHS = 3
MODEL = 'inception'
# MODEL = 'mobilenet'
# MODEL = 'load'

train_generator, validation_generator, train_len, valid_len = get_generator(DATA_DIRECTORIES, BATCH_SIZE)

model = None
if MODEL == 'inception':
    model = get_inception(FREEZE)

if MODEL == 'mobilenet':
    model = get_mobilenet(FREEZE)

if MODEL == 'load':
    model = load_model('model.h5', custom_objects={'tf': tf})
    print(model.summary())
if model is None:
    raise Exception('unknown model')

if LR_SEARCH:
    model.compile(optimizer='RMSProp', loss='mse', metrics=[MAPE])
    minimum_lr = 0.0001
    maximum_lr = 0.1
    lr_callback = LRFinder(train_len, BATCH_SIZE,
                           minimum_lr, maximum_lr,
                           lr_scale='exp', save_dir='lr_finder')
    model.fit_generator(train_generator,
                        steps_per_epoch=ceil(train_len / BATCH_SIZE),
                        validation_data=validation_generator,
                        validation_steps=ceil(valid_len / BATCH_SIZE),
                        epochs=1, verbose=1, callbacks=[lr_callback])
    lr_callback.plot_schedule()
    lr_callback.plot_schedule_from_file('lr_finder')

else:

    # lr = 0.001
    # wd = 0.001

    # inception lr 0.001
    # mobile lr 0.0001
    model.compile(optimizer='RMSProp', loss='mse', metrics=[MAPE])
    lr_manager = OneCycleLR(train_len,
                            batch_size=BATCH_SIZE,
                            max_lr=0.001,
                            end_percentage=0.1, scale_percentage=None,
                            maximum_momentum=None, minimum_momentum=None)
    history_object = model.fit_generator(train_generator,
                                         steps_per_epoch=ceil(train_len / BATCH_SIZE),
                                         validation_data=validation_generator,
                                         validation_steps=ceil(valid_len / BATCH_SIZE),
                                         epochs=EPOCHS,
                                         verbose=1,
                                         callbacks=[lr_manager])

    if SAVE_MODEL:
        model.save('model.h5')
        model.save('model.' + MODEL_NAME + '.h5')

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig(MODEL_NAME + '.png')

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
