from gibson.envs.husky_env import HuskyNavigateEnv
from gibson.utils.play import play
import os

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'play', 'play_husky_camera.yaml')
#config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'config_husky.yaml')
print(config_file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default=config_file)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    env = HuskyNavigateEnv(config=args.config, gpu_idx = args.gpu)
    play(env, zoom=4)