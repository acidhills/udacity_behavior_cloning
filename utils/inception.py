from keras.applications import InceptionV3
from keras.layers import Input, Lambda, Dense, Flatten, Dropout
from keras.models import Model
import tensorflow as tf


def get_inception(Freez=True):
    weights_flag = 'imagenet'
    inception = InceptionV3(weights=weights_flag, include_top=False,
                            input_shape=(139, 139, 3))

    if Freez:
        for l in inception.layers:
            l.trainable = False

    model_input = Input((160, 320, 3))
    cropped_input = Lambda(lambda x: tf.image.crop_to_bounding_box(x, 60, 0, 80, 320))(model_input)
    resized_input = Lambda(lambda image: tf.image.resize_images(image, (139, 139)))(cropped_input)
    normalized_input = Lambda(lambda x: x / 127.5 - 1.)(resized_input)
    inp = inception(normalized_input)
    flat = Flatten()(inp)
    drop = Dropout(0.5)(flat)
    dens1 = Dense(512, activation='relu')(drop)
    dens2 = Dense(512, activation='relu')(dens1)
    predictions = Dense(1, activation='linear')(dens2)
    return Model(inputs=model_input, outputs=predictions)