import numpy as np
import tensorflow as tf  # pylint: ignore-module
import builtins
import functools
import copy
import os
import collections


def make_gpu_session(num_gpu=1):
    if num_gpu == 1:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
    else:
        sess = tf.Session()
    return sess

