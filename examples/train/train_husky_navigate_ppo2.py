#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import gym, logging
from mpi4py import MPI
from gibson.envs.husky_env import HuskyNavigateEnv
from baselines.common import set_global_seeds
import baselines.common.tf_util as U
from gibson.utils.fuse_policy2 import MlpPolicy, MlpPolicy2, CnnPolicy2
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from gibson.utils import utils
import datetime
from baselines import logger
#from baselines.ppo2 import ppo2
from gibson.utils import ppo2
from gibson.utils import ppo2_imgs
from gibson.utils.monitor import Monitor
import os.path as osp
import tensorflow as tf
import random
import sys

## Training code adapted from: https://github.com/openai/baselines/blob/master/baselines/ppo1/run_atari.py

def train(num_timesteps, seed):
    rank = MPI.COMM_WORLD.Get_rank()
    #sess = U.single_threaded_session()
    sess = utils.make_gpu_session(args.num_gpu)
    sess.__enter__()
    if args.meta != "":
        saver = tf.train.import_meta_graph(args.meta)
        saver.restore(sess,tf.train.latest_checkpoint('./'))

    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    use_filler = not args.disable_filler

    if args.mode == "SENSOR":
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'husky_navigate_nonviz_train.yaml')
    else:
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'husky_navigate_rgb_train.yaml')
    print(config_file)

    raw_env = HuskyNavigateEnv(gpu_idx=args.gpu_idx,
                                config=config_file)

    env = Monitor(raw_env, logger.get_dir() and
        osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)

    gym.logger.setLevel(logging.WARN)

    policy_fn = MlpPolicy if args.mode == "SENSOR" else CnnPolicy2
    args.reload_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'gibson', 'utils', 'models', '00100')

    ppo2.learn(policy=policy_fn, env=env, nsteps=500, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.1,
        lr=lambda f : f * 2e-4,
        cliprange=lambda f : f * 0.2,
        total_timesteps=int(num_timesteps * 1.1),
        save_interval=10,
        sensor= args.mode == "SENSOR",
        reload_name=args.reload_name)

    env.close()

def callback(lcl, glb):
    # stop training if reward exceeds 199
    total = sum(lcl['episode_rewards'][-101:-1]) / 100
    totalt = lcl['t']
    is_solved = totalt > 2000 and total >= -50
    return is_solved


def main():
    train(num_timesteps=90000, seed=5)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default="RGB")
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--disable_filler', action='store_true', default=False)
    parser.add_argument('--meta', type=str, default="")
    parser.add_argument('--resolution', type=str, default="SMALL")
    parser.add_argument('--reload_name', type=str, default=None)
    parser.add_argument('--save_name', type=str, default=None)
    args = parser.parse_args()
    main()
