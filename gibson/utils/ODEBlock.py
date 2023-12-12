# import sys, os
# currentdir = '/home/berk/VS_Projects/Gibson_Exercise/gibson/utils/tensorflowlib'
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# os.sys.path.insert(0,parentdir)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras.backend as K
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Layer
import tf_slim as slim
from .odes import *

class ODEBlock(Layer):

    def __init__(self, filters, kernel_size, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        super(ODEBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv2d_w1 = self.add_weight("conv2d_w1", self.kernel_size + (self.filters + 1, self.filters), initializer='glorot_uniform')
        self.conv2d_w2 = self.add_weight("conv2d_w2", self.kernel_size + (self.filters + 1, self.filters), initializer='glorot_uniform')
        self.conv2d_b1 = self.add_weight("conv2d_b1", (self.filters,), initializer='zero')
        self.conv2d_b2 = self.add_weight("conv2d_b2", (self.filters,), initializer='zero')
        super(ODEBlock, self).build(input_shape)

    def call(self, x, **kwargs):
        t = K.constant([0, 1], dtype="float32")
        return odeint(self.ode_func, x, t, rtol=1e-3, atol=1e-3)[1]

    def compute_output_shape(self, input_shape):
        return input_shape

    def ode_func(self, x, t):
        y = self.concat_t(x, t)
        y = K.conv2d(y, self.conv2d_w1, padding="same")
        y = K.bias_add(y, self.conv2d_b1)
        y = K.relu(y)

        y = self.concat_t(y, t)
        y = K.conv2d(y, self.conv2d_w2, padding="same")
        y = K.bias_add(y, self.conv2d_b2)
        y = K.relu(y)

        return y

    def concat_t(self, x, t):
        new_shape = tf.concat(
            [
                tf.shape(x)[:-1],
                tf.constant([1],dtype="int32",shape=(1,))
            ], axis=0)

        t = tf.ones(shape=new_shape) * tf.reshape(t, (1, 1, 1, 1))
        return tf.concat([x, t], axis=-1)