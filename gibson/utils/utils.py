import numpy as np
import tensorflow as tf  # pylint: ignore-module
import builtins
import functools
import copy
import os
import collections


def make_gpu_session(num_gpu=1):
    if num_gpu == 1:
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    else:
        sess = tf.compat.v1.Session()
    return sess

