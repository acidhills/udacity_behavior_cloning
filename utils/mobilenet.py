from keras.applications import MobileNet
from keras.layers import Input, Lambda, Dense, Flatten, Dropout
from keras.models import Model, Sequential
import tensorflow as tf


def get_mobilenet(Freez=True):
    weights_flag = 'imagenet'
    mobilenet = MobileNet(weights=weights_flag, include_top=False,
                          input_shape=(224, 224, 3))

    if Freez:
        for l in mobilenet.layers:
            l.trainable = False

    return Sequential([
        Lambda(lambda x: tf.image.crop_to_bounding_box(x, 60, 0, 80, 320), input_shape=(160, 320, 3)),
        Lambda(lambda image: tf.image.resize_images(image, (224, 224))),
        Lambda(lambda x: x / 127.5 - 1.),
        mobilenet,
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),
        Dense(1, activation='linear')
    ])
