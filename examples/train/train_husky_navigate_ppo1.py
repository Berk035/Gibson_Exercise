#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import gym, logging
from mpi4py import MPI
from gibson.envs.husky_env import HuskyNavigateEnv
from baselines.common import set_global_seeds
from gibson.utils import pposgd_simple, pposgd_fuse
import baselines.common.tf_util as U
from gibson.utils import cnn_policy, mlp_policy, fuse_policy
from gibson.utils import utils
from baselines import logger
from gibson.utils.monitor import Monitor
import os.path as osp
import tensorflow as tf
import random
import sys
import time
import datetime

#Training code adapted from: https://github.com/openai/baselines/blob/master/baselines/ppo1/run_atari.py
#Shows computation device ----> sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
def train(seed):
    rank = MPI.COMM_WORLD.Get_rank()
    sess = utils.make_gpu_session(args.num_gpu)
    sess.__enter__()

    if args.meta != "":
        saver = tf.train.import_meta_graph(args.meta)
        saver.restore(sess, tf.train.latest_checkpoint('./'))

    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    use_filler = not args.disable_filler

    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'config_husky.yaml')
    print(config_file)

    raw_env = HuskyNavigateEnv(gpu_idx=args.gpu_idx, config=config_file)
    step = raw_env.config['n_step']; episode = raw_env.config['n_episode']; iteration = raw_env.config['n_iter']
    elm_policy = raw_env.config['elm_active']
    num_timesteps = step*episode*iteration
    tpa = step*episode

    if args.mode == "SENSOR":
        def policy_fn(name, ob_space, ac_space):
            return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=128, num_hid_layers=4,
                                        elm_mode=elm_policy)
    else:
        def policy_fn(name, ob_space, sensor_space, ac_space):
            return fuse_policy.FusePolicy(name=name, ob_space=ob_space, sensor_space = sensor_space, ac_space=ac_space,
                                          save_per_acts=10000, hid_size=64, num_hid_layers=3, session=sess, elm_mode=elm_policy)

    env = Monitor(raw_env, logger.get_dir() and
                  osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    #args.reload_name = '/home/berk/PycharmProjects/Gibson_Exercise/gibson/utils/models/PPO_DEPTH_2020-08-04_500_50_91_10.model'
    print(args.reload_name)

    if args.mode == "SENSOR":
        pposgd_simple.learn(env, policy_fn,
                            max_timesteps=int(num_timesteps * 1.1),
                            timesteps_per_actorbatch=tpa,
                            clip_param=0.2, entcoeff=0.01,
                            optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
                            gamma=0.996, lam=0.95,
                            schedule='linear',
                            save_name="PPO_{}_{}_{}_{}_{}".format(args.mode, datetime.date.today(), step, episode,
                                                                  iteration),
                            save_per_acts=10,
                            sensor=args.mode == "SENSOR",
                            reload_name=args.reload_name
                            )
    else:
        pposgd_fuse.learn(env, policy_fn,
                              max_timesteps=int(num_timesteps * 1.1),
                              timesteps_per_actorbatch=tpa,
                              clip_param=0.2, entcoeff=0.01,
                              optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
                              gamma=0.99, lam=0.95,
                              schedule='linear',
                              save_name="PPO_{}_{}_{}_{}_{}".format(args.mode, datetime.date.today(), step, episode,
                                                                    iteration),
                              save_per_acts=10,
                              reload_name=args.reload_name
                              )

    env.close()

def callback(lcl, glb):
    # stop training if reward exceeds 199
    total = sum(lcl['episode_rewards'][-101:-1]) / 100
    totalt = lcl['t']
    is_solved = totalt > 2000 and total >= -50
    return is_solved

def main():
    tic = time.time()
    train(seed=5)
    toc = time.time()
    sec = toc - tic;    min, sec = divmod(sec,60);   hour, min = divmod(min,60)
    print("Process Time: {:.4g} hour {:.4g} min {:.4g} sec".format(hour,min,sec))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default="DEPTH")
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--disable_filler', action='store_true', default=False)
    parser.add_argument('--meta', type=str, default="")
    parser.add_argument('--resolution', type=str, default="SMALL")
    parser.add_argument('--reload_name', type=str, default=None)
    parser.add_argument('--save_name', type=str, default=None)
    args = parser.parse_args()
    main()