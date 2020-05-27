# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import tensorflow as tf
import gym, logging
from mpi4py import MPI
from gibson.envs.husky_env import HuskyGibsonFlagRunEnv
from baselines.common import set_global_seeds
from gibson.utils import fuse_policy
from gibson.utils import utils
from gibson.utils import pposgd_fuse
from baselines import logger



## Training code adapted from: https://github.com/openai/baselines/blob/master/baselines/ppo1/run_atari.py
def enjoy(seed):
    rank = MPI.COMM_WORLD.Get_rank()
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

    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs',
                               'husky_gibson_flagrun_train_enjoy.yaml')
    env = HuskyGibsonFlagRunEnv(gpu_idx=0, config=config_file)
    step = env.config['n_step']; batch = env.config['n_batch']; iteration = env.config['n_iter']
    num_timesteps = step * batch * iteration
    tpa = step * batch

    def policy_fn(name, ob_space, sensor_space, ac_space):
        return fuse_policy.FusePolicy(name=name, ob_space=ob_space, sensor_space = sensor_space,  ac_space=ac_space,
                                      hid_size=128, num_hid_layers=2, save_per_acts=10, session=sess)

    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    args.reload_name = '/home/berk/PycharmProjects/Gibson_Exercise/gibson/utils/models/PPO_DEPTH_2019-11-30_500_32_100_100.model'

    pposgd_fuse.enjoy(env, policy_fn,
                        max_timesteps=int(num_timesteps * 1.1),
                        timesteps_per_actorbatch=tpa,
                        clip_param=0.2, entcoeff=0.0,
                        optim_epochs=4, optim_stepsize=3e-3, optim_batchsize=64,
                        gamma=0.996, lam=0.95,
                        schedule='linear',
                        save_per_acts=25,
                        reload_name=args.reload_name,
                      )
    env.close()

def callback(lcl, glb):
    # stop training if reward exceeds 199
    total = sum(lcl['episode_rewards'][-101:-1]) / 100
    totalt = lcl['t']
    is_solved = totalt > 2000 and total >= -50
    return is_solved


def main():
    enjoy(seed=5)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default="DEPTH")
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--disable_filler', action='store_true', default=False)
    parser.add_argument('--meta', type=str, default="")
    parser.add_argument('--reload_name', type=str, default=None)
    parser.add_argument('--save_name', type=str, default=None)
    args = parser.parse_args()

    main()