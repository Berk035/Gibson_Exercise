import baselines.common.tf_util as U
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Conv2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
from keras.layers import add
from keras.activations import relu, softmax
from keras.models import Model
from keras import regularizers
import gym
from baselines.common.distributions import make_pdtype
from baselines.common.mpi_running_mean_std import RunningMeanStd
#from gibson.core.render.profiler import Profiler
import cv2,os
import numpy as np
import matplotlib.pyplot as plt

def block(n_output, upscale=False):
    # n_output: number of feature maps in the block
    # upscale: should we use the 1x1 conv2d mapping for shortcut or not
    
    # keras functional api: return the function of type
    # Tensor -> Tensor
    def f(x):
        
        # H_l(x):
        # first pre-activation
        #h = BatchNormalization()(x)
        h = Activation(relu)(x)
        # first convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(h)
        
        # second pre-activation
        #h = BatchNormalization()(x)
        h = Activation(relu)(h)
        # second convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(h)
        
        # f(x):
        if upscale:
            # 1x1 conv2d
            f = Conv2D(kernel_size=1, filters=n_output, strides=1, padding='same')(x)
        else:
            # identity
            f = x
        
        # F_l(x) = f(x) + H_l(x):
        return add([f, h])
    
    return f

class ResPolicy(object):
    recurrent = False
    def __init__(self, name, ob_space, sensor_space, ac_space, session, hid_size, num_hid_layers, save_per_acts=None,
                 kind='large', elm_mode=False):
        self.total_count = 0
        self.curr_count = 0
        self.save_per_acts = save_per_acts
        self.session = session
        with tf.variable_scope(name):
            self._init(ob_space, sensor_space, ac_space, hid_size, num_hid_layers, kind, elm_mode)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, sensor_space, ac_space, hid_size, num_hid_layers, kind, elm_mode):
        assert isinstance(ob_space, gym.spaces.Box)
        assert isinstance(sensor_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        ob_sensor = U.get_placeholder(name="ob_sensor", dtype=tf.float32,
                                      shape=[sequence_length] + list(sensor_space.shape))

        x = ob / 255.0
        # x = tf.nn.relu(U.conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID"))
        # x = tf.nn.relu(U.conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID"))

        # num_res_net_blocks = 3
        # for i in range(num_res_net_blocks):
        #     input_data = x
        #     for j in range(2):
        #         x = tf.nn.relu(U.conv2d(x, 32, "l%i"%(2*i+3+j), filter_size=[8, 8], pad="SAME"))
        #     x = tf.nn.relu(tf.math.add(x,input_data))

        # x = U.flattenallbut0(x)
        # x = tf.nn.relu(tf.layers.dense(x, 256, name='lin', kernel_initializer=U.normc_initializer(1.0)))

        # input_tensor = Input(shape=(128,128,1), name="input")
        # x = Conv2D(3,(3,3),padding='same')(x)
        
        # x = tf.image.grayscale_to_rgb(x, name=None)
        # base_model = ResNet50(input_tensor=x, weights=None)
        # x = base_model.output
        # # x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # x = Dense(256, activation='relu')(x)


        # first conv2d with post-activation to transform the input data to some reasonable form
        x = Conv2D(kernel_size=3, filters=32, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
        #x = BatchNormalization()(x)
        x = Activation(relu)(x)

        # F_1
        x = block(32)(x)
        # F_2
        x = block(32)(x)

        # F_3
        # H_3 is the function from the tensor of size 28x28x16 to the the tensor of size 28x28x32
        # and we can't add together tensors of inconsistent sizes, so we use upscale=True
        # x = block(32, upscale=True)(x)       # !!! <------- Uncomment for local evaluation
        # F_4
        # x = block(32)(x)                     # !!! <------- Uncomment for local evaluation
        # F_5
        # x = block(32)(x)                     # !!! <------- Uncomment for local evaluation

        # F_6
        # x = block(48, upscale=True)(x)       # !!! <------- Uncomment for local evaluation
        # F_7
        # x = block(48)(x)                     # !!! <------- Uncomment for local evaluation

        # last activation of the entire network's output
        #x = BatchNormalization()(x)
        x = Activation(relu)(x)

        # average pooling across the channels
        # 28x28x48 -> 1x48
        x = GlobalAveragePooling2D()(x)

        # dropout for more robust learning
        #x = Dropout(0.2)(x)

        # last softmax layer
        x = Dense(units=256, kernel_regularizer=regularizers.l2(0.01))(x)
        x = Activation(relu)(x)

        ## Obfilter on sensor output
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=sensor_space.shape)
        obz_sensor = tf.clip_by_value((ob_sensor - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

        last_out = obz_sensor
        if not elm_mode:
            ## Adapted from mlp_policy
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="vffc%i" % (i + 1),
                                                      kernel_initializer=U.normc_initializer(1.0)))
            y = tf.layers.dense(last_out, 64, name="vffinal", kernel_initializer=U.normc_initializer(1.0))
        else:
            last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="vffc1",
                                                  kernel_initializer=U.normc_initializer(1.0), trainable=False))
            y = tf.layers.dense(last_out, 64, name="vffinal", kernel_initializer=U.normc_initializer(1.0))


        x = tf.concat([x, y], 1)
        logits = tf.layers.dense(x, pdtype.param_shape()[0], name="logits",
                                 kernel_initializer=U.normc_initializer(0.01))
        self.pd = pdtype.pdfromflat(logits)
        self.vpred = tf.layers.dense(x, 1, name="value", kernel_initializer=U.normc_initializer(1.0))[:, 0]

        # self.session.run(logits.kernel)
        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample()  # XXX
        self._act = U.function([stochastic, ob, ob_sensor], [ac, self.vpred, logits])

    def act(self, stochastic, ob, ob_sensor):
        ac1, vpred1, _ = self._act(stochastic, ob[None], ob_sensor[None])
        self.total_count = self.total_count + 1
        self.curr_count = self.curr_count + 1

        return ac1[0], vpred1[0]

    def get_logits(self, stochastic, ob, ob_sensor):
        ac1, vpred1, logits = self._act(stochastic, ob[None], ob_sensor[None])
        self.total_count = self.total_count + 1
        self.curr_count = self.curr_count + 1

        return ac1[0], vpred1[0], logits[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []