from keras.layers import Layer
from keras import backend as K
import cv2
import numpy as np
import tensorflow as tf


class PerspectiveLayer(Layer):
    def __init__(self, perspective_matrix, **kwargs):
        self.perspective_matrix = perspective_matrix
        super(PerspectiveLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        out = tf.py_func(lambda x: self.func_(x, self.perspective_matrix),
                         [inputs],
                         'float32',
                         stateful=False)
        K.stop_gradient(out)
        out.set_shape(inputs.shape)
        return out

    def func_(self, imgs, M):
        res = [cv2.warpPerspective(x, M, None) for x in imgs]
        return np.float32(res)

    def compute_output_shape(self, input_shape):
        return (input_shape)


def getPerspectiveTransformMatrix(xlen=320, ylen=160, x_top_shift=50, y_top_shift=90):
    src = np.int32([[0, ylen], [xlen, ylen], [x_top_shift, y_top_shift], [xlen - x_top_shift, y_top_shift]])

    dst = np.int32([[xlen / 4, ylen], [xlen * 3 / 4, ylen], [0, 0], [xlen, 0]])

    src = np.float32(src)
    dst = np.float32(dst)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv
