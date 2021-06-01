from keras.applications import InceptionV3
from keras.layers import Input, Lambda, Dense, Flatten, Dropout
from keras.models import Model, Sequential
import tensorflow as tf
from utils.perspective_layer import PerspectiveLayer, getPerspectiveTransformMatrix


def get_inception(Freez=True):
    weights_flag = 'imagenet'
    inception = InceptionV3(weights=weights_flag, include_top=False,
                            input_shape=(139, 139, 3))

    M, M_inv = getPerspectiveTransformMatrix()

    if Freez:
        for l in inception.layers:
            l.trainable = False

    return Sequential([
            PerspectiveLayer(M, input_shape=(160, 320, 3)),
            Lambda(lambda x: tf.image.crop_to_bounding_box(x, 0, 0, 132, 320)),
            Lambda(lambda image: tf.image.resize_images(image, (139, 139))),
            Lambda(lambda x: x / 127.5 - 1.),
            inception,
            Flatten(),
            Dropout(0.5),
            Dense(100, activation='relu'),
            Dense(50, activation='relu'),
            Dense(10, activation='relu'),
            Dense(1, activation='linear')
        ])


