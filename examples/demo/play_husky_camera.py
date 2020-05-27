from gibson.envs.husky_env import HuskyNavigateEnv
from gibson.utils.play import play
import os
#from examples.train.exercise_files.play2 import play
from examples.train.exercise_files.play2 import PlayPlot as plt

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'play', 'play_husky_camera.yaml')
#config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'config_deneme.yaml')
print(config_file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default=config_file)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    env = HuskyNavigateEnv(config=args.config, gpu_idx = args.gpu)
    play(env, zoom=4)

    #p = plt(callback=plt.callback_2, horizon_timesteps=10000000000, plot_names=["reward"])
    #play(env, zoom=4, callback=p.callback)

    #------------ DENEME
    '''env.reset()
    import numpy
    for i in range(10000): # number of steps
        action = numpy.random.random_integers(0,1)
        obs, rew, env_done, info = env.step(action)
        pose = [env.robot.get_position(), env.robot.get_orientation()]
        print(pose)
        print(action)
    '''