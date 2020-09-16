import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype

class ODEBlock(object):
    def __init__(self, name, ob_space, ac_space, session, save_per_acts=1e4, kind='large'):
        self.total_count = 0
        self.curr_count = 0
        self.save_per_acts = save_per_acts
        self.session = session
        self.filters = 32
        self.kernel_size = 4
        with tf.variable_scope(name):
            self._init(ob_space, ac_space, kind)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, kind):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        t = tf.constant([0, 1], dtype="float32")
        tf.contrib.integrate.odeint(self.ode_func, x, t, rtol=1e-3, atol=1e-3)[1]
        #-------------------------------------BURADA KALDIK

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        x = ob / 255.0
        x = tf.nn.relu(U.conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID"))
        x = tf.nn.relu(U.conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID"))
        x = U.flattenallbut0(x)
        x = tf.nn.softmax(tf.layers.dense(x, 256, name='lin', kernel_initializer=U.normc_initializer(1.0)))

        logits = tf.layers.dense(x, pdtype.param_shape()[0], name="logits", kernel_initializer=U.normc_initializer(0.01))
        self.pd = pdtype.pdfromflat(logits)
        self.vpred = tf.layers.dense(x, 1, name="value", kernel_initializer=U.normc_initializer(1.0))[:, 0]

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample()  # XXX
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        self.total_count = self.total_count + 1
        self.curr_count = self.curr_count + 1

        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []