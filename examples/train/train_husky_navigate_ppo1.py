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
# from examples.plot_result import *
import baselines.common.tf_util as U
from gibson.utils import cnn_policy, mlp_policy, fuse_policy, resnet_policy, ode_policy
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

def callback(lcl, glb):
    # stop training if reward exceeds 199
    #total = sum(lcl['rewbuffer'][-101:-1]) / 100
    total = lcl['tot']
    totalt = lcl['t']
    is_solved = totalt > 2000 and total >= 50
    return is_solved

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

    if args.mode == "SENSOR": #Blind Mode
        def policy_fn(name, ob_space, ac_space):
            return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=128, num_hid_layers=4,
                                        elm_mode=elm_policy)
    elif args.mode == "DEPTH" or args.mode == "RGB": #Fusing sensor space with image space
        def policy_fn(name, ob_space, sensor_space, ac_space):
            return fuse_policy.FusePolicy(name=name, ob_space=ob_space, sensor_space = sensor_space, ac_space=ac_space,
                                          save_per_acts=10000, hid_size=128, num_hid_layers=4, session=sess, elm_mode=elm_policy)

    elif args.mode == "RESNET":
        def policy_fn(name, ob_space, sensor_space, ac_space):
            return resnet_policy.ResPolicy(name=name, ob_space=ob_space, sensor_space = sensor_space, ac_space=ac_space,
                                          save_per_acts=10000, hid_size=128, num_hid_layers=4, session=sess, elm_mode=elm_policy)

    elif args.mode == "ODE":
        def policy_fn(name, ob_space, sensor_space, ac_space):
            return ode_policy.ODEPolicy(name=name, ob_space=ob_space, sensor_space = sensor_space, ac_space=ac_space,
                                          save_per_acts=10000, hid_size=128, num_hid_layers=4, session=sess, elm_mode=elm_policy)

    else: #Using only image space
        def policy_fn(name, ob_space, ac_space):
            return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space, session=sess, kind='small')

    env = Monitor(raw_env, logger.get_dir() and
                  osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    #args.reload_name = '/home/berk/VS_Projects/Gibson_Exercise/gibson/utils/models/PPO_ODE_2020-12-05_500_50_137_150.model'
    print(args.reload_name)

    modes_camera = ["DEPTH", "RGB", "RESNET", "ODE"]
    if args.mode in modes_camera:
        pposgd_fuse.learn(env, policy_fn,
                          max_timesteps=int(num_timesteps * 1.1),
                          timesteps_per_actorbatch=tpa,
                          clip_param=0.2, entcoeff=0.03,
                          vfcoeff=0.01,
                          optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
                          gamma=0.99, lam=0.95,
                          schedule='linear',
                          save_name="PPO_{}_{}_{}_{}_{}".format(args.mode, datetime.date.today(), step, episode,
                                                                iteration),
                          save_per_acts=15,
                          reload_name=args.reload_name
                          )
    else:
        if args.mode == "SENSOR": sensor = True
        else: sensor = False
        pposgd_simple.learn(env, policy_fn,
                            max_timesteps=int(num_timesteps * 1.1),
                            timesteps_per_actorbatch=tpa,
                            clip_param=0.2, entcoeff=0.03,
                            vfcoeff=0.01,
                            optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
                            gamma=0.996, lam=0.95,
                            schedule='linear',
                            save_name="PPO_{}_{}_{}_{}_{}".format(args.mode, datetime.date.today(), step, episode,
                                                                  iteration),
                            save_per_acts=15,
                            sensor=sensor,
                            reload_name=args.reload_name
                            )
    env.close()

def main():
    tic = time.time(); start = time.ctime()
    #args.eps=7500 ;mesh_2D_v2.main(raw_args=args)
    train(seed=5)
    toc = time.time(); finish = time.ctime()
    sec = toc - tic;    min, sec = divmod(sec,60);   hour, min = divmod(min,60)
    #mesh_2D_v2.main(raw_args=args)
    print("Process Time: {:.4g} hour {:.4g} min {:.4g} sec".format(hour,min,sec))
    pathtxt = os.path.join(os.path.expanduser("~"),
                           "VS_Projects/Gibson_Exercise/gibson/utils/models/time_elapsed.txt")
    f = open(pathtxt, "w+"); f.write("Date: {}\n".format(datetime.date.today()))
    f.write("Start-Finish: {} *** {}\n".format(start,finish))
    f.write("Total Time: {:.4g} hour {:.4g} min {:.4g} sec\n".format(hour, min, sec))
    f.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default="ODE")
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--disable_filler', action='store_true', default=False)
    parser.add_argument('--meta', type=str, default="")
    parser.add_argument('--resolution', type=str, default="SMALL")
    parser.add_argument('--reload_name', type=str, default=None)
    parser.add_argument('--save_name', type=str, default=None)
    #---------Show Result------------
    parser.add_argument('--eps', type=int, default=5000)  # Number of episode
    parser.add_argument('--map', type=int, default=5)  # Number of shown map
    parser.add_argument('--model', type=str, default="Euharlee")  # Map ID
    args = parser.parse_args()
    main()