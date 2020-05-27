import tensorflow as tf


def pinv(A):
    # Moore-Penrose pseudo-inverse
    with tf.name_scope("pinv"):
        s, u, v = tf.svd(A, compute_uv=True)
        s_inv = tf.reciprocal(s)
        s_inv = tf.diag(s_inv)
        left_mul = tf.matmul(v, s_inv)
        u_t = tf.transpose(u)
        return tf.matmul(left_mul, u_t)


def train(x, y, weights, bias, activation=tf.nn.relu, name="elm_train"):
    with tf.name_scope("{}_{}".format(name, 'hidden')):
        with tf.name_scope("H"):
            h_matrix = tf.matmul(x, weights) + bias
            h_act = activation(h_matrix)

        h_pinv = pinv(h_act)

        with tf.name_scope("Beta"):
            beta = tf.matmul(h_pinv, y)
        return beta


def inference(x, weights, bias, beta, activation=tf.nn.relu, name="elm_inference"):
    with tf.name_scope("{}_{}".format(name, 'out')):
        with tf.name_scope("H"):
            h_matrix = tf.matmul(x, weights) + bias
            h_act = activation(h_matrix)

        out = tf.matmul(h_act, beta)
        return out

