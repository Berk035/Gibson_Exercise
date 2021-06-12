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
from gibson.utils.actor_critic_simple import learn
from examples.plot_result import mesh_2D_v2
import baselines.common.tf_util as U
from gibson.utils import ac2_policy
from gibson.utils import utils_2
from baselines import logger
from gibson.utils.monitor import Monitor
import os.path as osp
import tensorflow as tf
import random
import sys
import time
import datetime
import examples.plot_result

#Training code adapted from: https://github.com/openai/baselines/blob/master/baselines/ppo1/run_atari.py
#Shows computation device ----> sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def train(seed, policy, lrschedule):
    rank = MPI.COMM_WORLD.Get_rank()
    sess = utils_2.make_gpu_session(args.num_gpu)
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
    num_timesteps = step*episode*iteration
    tpa = step*episode

    env = Monitor(raw_env, logger.get_dir() and
                  osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    #def policy_fn(name, ob_space, sensor_space, ac_space, reuse):
    #    return ac2_policy.CnnPolicy(name=name, ob_space=ob_space, sensor_space = sensor_space, ac_space=ac_space,
    #                                      save_per_acts=10000, hid_size=128, num_hid_layers=4, sess=sess,
    #                                nbatch=tpa, nsteps=num_timesteps, reuse=reuse)

    policy_fn = ac2_policy.CnnPolicy
    #def policy_fn():
    #    return ac2_policy.FusePolicy(name=name, ob_space=ob_space, sensor_space = sensor_space, ac_space=ac_space,
    #                                      save_per_acts=10000, hid_size=128, num_hid_layers=4, session=sess)

    learn(policy_fn, env, seed, nsteps=step, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule)
    env.close()

def main():
    tic = time.time(); start = time.ctime()
    logger.configure()
    train(seed=5, policy=args.policy, lrschedule=args.lrschedule)
    toc = time.time(); finish = time.ctime()
    sec = toc - tic;    min, sec = divmod(sec,60);   hour, min = divmod(min,60)
    mesh_2D_v2.main(raw_args=args)
    print("Process Time: {:.4g} hour {:.4g} min {:.4g} sec".format(hour,min,sec))
    pathtxt = os.path.join(os.path.expanduser("~"),
                           "PycharmProjects/Gibson_Exercise/gibson/utils/models/time_elapsed.txt")
    f = open(pathtxt, "w+"); f.write("Date: {}\n".format(datetime.date.today()))
    f.write("Start-Finish: {} *** {}\n".format(start,finish))
    f.write("Total Time: {:.4g} hour {:.4g} min {:.4g} sec\n".format(hour, min, sec))
    f.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default="A2C")
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
    #--------------------------------
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'],
                        default='constant')
    args = parser.parse_args()
    main()