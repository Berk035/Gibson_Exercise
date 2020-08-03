import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time


# CHECK : Constants
omega = 1.

class ELM(object):
    def __init__(self, sess, batch_size, input_len, hidden_num, output_len):
        '''
        Args:
          sess : TensorFlow session.
          batch_size : The batch size (N)
          input_len : The length of input. (L)
          hidden_num : The number of hidden node. (K)
          output_len : The length of output. (O)
        '''

        self._sess = sess
        self._batch_size = batch_size
        self._input_len = input_len
        self._hidden_num = hidden_num
        self._output_len = output_len

        # for train
        self._x0 = tf.placeholder(tf.float32, [self._batch_size, self._input_len])
        self._t0 = tf.placeholder(tf.float32, [self._batch_size, self._output_len])

        # for test
        self._x1 = tf.placeholder(tf.float32, [None, self._input_len])
        self._t1 = tf.placeholder(tf.float32, [None, self._output_len])

        self._W = tf.Variable(
            tf.random_normal([self._input_len, self._hidden_num]),
            trainable=False, dtype=tf.float32)
        self._b = tf.Variable(
            tf.random_normal([self._hidden_num]),
            trainable=False, dtype=tf.float32)
        self._beta = tf.Variable(
            tf.zeros([self._hidden_num, self._output_len]),
            trainable=False, dtype=tf.float32)
        self._var_list = [self._W, self._b, self._beta]

        self.H0 = tf.matmul(self._x0, self._W) + self._b  # N x L
        self.H0_T = tf.transpose(self.H0)

        self.H1 = tf.matmul(self._x1, self._W) + self._b  # N x L
        self.H1_T = tf.transpose(self.H1)

        # beta analytic solution : self._beta_s (K x O)
        if self._input_len < self._hidden_num:  # L < K
            identity = tf.constant(np.identity(self._hidden_num), dtype=tf.float32)
            self._beta_s = tf.matmul(tf.matmul(tf.matrix_inverse(
                tf.matmul(self.H0_T, self.H0) + identity / omega),
                self.H0_T), self._t0)
            # _beta_s = (H_T*H + I/om)^(-1)*H_T*T
        else:
            identity = tf.constant(np.identity(self._batch_size), dtype=tf.float32)
            self._beta_s = tf.matmul(tf.matmul(self.H0_T, tf.matrix_inverse(
                tf.matmul(self.H0, self.H0_T) + identity / omega)), self._t0)
            # _beta_s = H_T*(H*H_T + I/om)^(-1)*T

        self._assign_beta = self._beta.assign(self._beta_s)
        self._fx0 = tf.matmul(self.H0, self._beta)
        self._fx1 = tf.matmul(self.H1, self._beta)

        self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._fx0, labels=self._t0))

        self._init = False
        self._feed = False

        # for the mnist test
        self._correct_prediction = tf.equal(tf.argmax(self._fx1, 1), tf.argmax(self._t1, 1))
        self._accuracy = tf.reduce_mean(tf.cast(self._correct_prediction, tf.float32))

    def feed(self, x, t):
        '''
        Args :
          x : input array (N x L)
          t : label array (N x O)
        '''

        if not self._init: self.init()
        self._sess.run(self._assign_beta, {self._x0: x, self._t0: t})
        self._feed = True

    def init(self):
        self._sess.run(tf.initialize_variables(self._var_list))
        self._init = True

    def test(self, x, t=None):
        if not self._feed: exit("Not feed-forward trained")
        if t is not None:
            print("Accuracy: {:.9f}".format(self._sess.run(self._accuracy, {self._x1: x, self._t1: t})))
        else:
            return self._sess.run(self._fx1, {self._x1: x})

# Basic tf setting
tf.set_random_seed(1000)
sess = tf.Session()

# Get data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Construct ELM
batch_size = 400
hidden_num = 10
print("batch_size : {}".format(batch_size))
print("hidden_num : {}".format(hidden_num))

tic = time.time()

elm = ELM(sess, batch_size, 784, hidden_num, 10)
# one-step feed-forward training
train_x, train_y = mnist.train.next_batch(batch_size)
elm.feed(train_x, train_y)

toc = time.time()
sec = toc - tic; min, sec = divmod(sec, 60); hour, min = divmod(min, 60)
print("Process Time: {:.4g} hour {:.4g} min {:.4g} sec".format(hour, min, sec))

'''
        Args:
          sess : TensorFlow session.
          batch_size : The batch size (N)
          input_len : The length of input. (L)
          hidden_num : The number of hidden node. (K)
          output_len : The length of output. (O)
'''

# testing
elm.test(mnist.test.images, mnist.test.labels)

