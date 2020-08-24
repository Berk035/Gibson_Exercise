import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
#from gibson.core.render.profiler import Profiler
import cv2,os
import numpy as np
import matplotlib.pyplot as plt

class CnnPolicy(object):
    recurrent = False
    def __init__(self, name, ob_space, ac_space, session, save_per_acts=1e4, kind='large'):
        self.total_count = 0
        self.curr_count = 0
        self.save_per_acts = save_per_acts
        self.session = session
        with tf.variable_scope(name):
            self._init(ob_space, ac_space, kind)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, kind):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        x = ob / 255.0
        #print(x)

        print(x.shape)
        #ob = np.reshape(ob,(128,128))
        #plt.imshow(x, cmap='Greys')
        #plt.show()

        '''
        record_depth = 1
        if record_depth:
            path_1 = "/home/berk/PycharmProjects/Gibson_Exercise/gibson/utils/models/depth_images_iteration"
            try:
                os.mkdir(path_1)
            except OSError:
                pass
            x = x * 35 + 20  # DEPTH SCALE FACTOR and DEPTH SCALE OFFSET
            overflow = x > 255.
            x[overflow] = 255.
            cv2.imwrite(os.path.join(path_1, 'Frame_{:d}.jpg').format(self.total_count), x)
        self.total_count += 1'''

        '''x = tf.nn.relu(U.conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID"))
        x = tf.nn.relu(U.conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID"))
        x = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

        num_res_net_blocks = 10
        for i in range(num_res_net_blocks):
            input_data = x
            for j in range(2):
                x = tf.nn.relu(U.conv2d(x, 32, "l%i"%(2*i+3+j), [2, 2], pad="SAME"))
            #x = tf.nn.batch_normalization(x)
            #x = tf.nn.batch_normalization(x)
            x = tf.nn.relu(tf.math.add(x,input_data))

        x = tf.nn.relu(U.conv2d(x, 32, "out", [2, 2], pad="VALID"))
        '''#x = tf.layers.average_pooling2d(x,pool_size=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        x = U.flattenallbut0(x)
        #x = tf.nn.relu(tf.layers.dense(x, 256, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        #x = tf.nn.dropout(x,0.5)

        logits = tf.layers.dense(x, pdtype.param_shape()[0], name="logits", kernel_initializer=U.normc_initializer(0.01))
        self.pd = pdtype.pdfromflat(logits)
        self.vpred = tf.layers.dense(x, 1, name="value", kernel_initializer=U.normc_initializer(1.0))[:,0]

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample() # XXX
        self._act = U.function([stochastic, ob], [ac, self.vpred])


    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        self.total_count = self.total_count + 1
        self.curr_count = self.curr_count + 1

        '''x = np.reshape(ob, (128, 128))
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(x, cmap='Greys')
        fig.add_subplot(1, 2, 2)
        plt.imshow(x/255, cmap='Greys')
        plt.show()'''

        '''if self.curr_count > self.save_per_acts:
            self.saver = tf.train.Saver()
            self.curr_count = self.curr_count - self.save_per_acts
            self.saver.save(self.session, '/home/berk/PycharmProjects/Gibson_Exercise/gibson/utils/cnn_policy',  
                            global_step=self.total_count)'''

        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []
