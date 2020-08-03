import os, inspect
import os.path as osp
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import matplotlib.pyplot as plt
import numpy as np
from gibson.envs.husky_env import HuskyNavigateEnv


i_x=-7.5; i_y= 5.042
t_x=-12.5; t_y= 5.042


def read_file(all=None):

    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'configs', 'config_deneme.yaml')
    print(config_file)
    raw_env = HuskyNavigateEnv(config=config_file)
    step = raw_env.config['n_step']; batch = raw_env.config['n_batch']; iteration = raw_env.config['n_iter']
    initial_pos = raw_env.config['initial_pos']; target_pos = raw_env.config['target_pos']

    for ep_n in range((iteration+1)*batch):
        count = 0
        for line in open(r"/home/berk/PycharmProjects/Gibson_Exercise/gibson/utils/models/episodes/positions" +
                         "_" + str(ep_n) + ".txt").readlines(): count += 1
        timesteps = count
        fn = np.arange(timesteps)

        ep_pos = open(r"/home/berk/PycharmProjects/Gibson_Exercise/gibson/utils/models/episodes/positions" +
                          "_" + str(ep_n) + ".txt", "r")

        x_pos = np.zeros(timesteps)
        y_pos = np.zeros(timesteps)
        x_vel = np.zeros(timesteps)
        y_vel = np.zeros(timesteps)
        tot_ret = np.zeros(timesteps)
        counter = 0; sum = 0
        for line in ep_pos:
            pos = line.split(";")
            nframe = pos[0]
            x_pos[counter] = pos[1]
            y_pos[counter] = pos[2]
            x_vel[counter] = pos[3]
            y_vel[counter] = pos[4]
            sum += float(pos[5].replace('\n', ''))
            tot_ret[counter] = sum
            counter += 1
            if nframe == str(timesteps):
                break

        if all:
            plt.figure(1, figsize=(18, 10))
            plt.subplot((iter_n+1)*batch, 2, 2*ep_n+1)
            plt.xlim(-15, 0)
            plt.ylim(-15, 15)
            plt.title('Husky Path <%i>' % ep_n)
            plt.annotate('Start Point', xy=(i_x, i_y), xytext=(i_x, i_y),
                         arrowprops=dict(facecolor='black', shrink=0.05))
            plt.annotate('Target Point', xy=(t_x, t_y), xytext=(t_x, t_y),
                         arrowprops=dict(facecolor='blue', shrink=0.05))
            plt.plot(x_pos, y_pos, 'r')
            plt.grid(True)
            plt.subplot((iter_n+1)*batch, 2, 2*(ep_n+1))
            plt.title('Episode Reward <%i>' % ep_n)
            plt.plot(fn, tot_ret, 'b')
            plt.grid(True)
            plt.tight_layout()
        else:
            plt.figure(1, figsize=(18, 10))
            plt.subplot(1, 2, 1)
            #plt.xlim(-15, 0)
            #plt.ylim(-15, 15)
            plt.title('Husky Path <%i>' % ep_n)
            plt.annotate('Start Point', xy=(i_x, i_y), xytext=(i_x, i_y),
                         arrowprops=dict(facecolor='black', shrink=0.05))
            plt.annotate('Target Point', xy=(t_x, t_y), xytext=(t_x, t_y),
                         arrowprops=dict(facecolor='blue', shrink=0.05))
            plt.plot(x_pos, y_pos, 'r')
            plt.grid(True)
            plt.subplot(1, 2, 2)
            plt.title('Episode Reward <%i>' % ep_n)
            plt.plot(fn, tot_ret, 'b')
            plt.grid(True)
            plt.tight_layout()

        ep_pos.close()

        if not all:
            C1 = '\033[93m'
            C1END = '\033[0m'
            print(C1 + "PLOTTING:" + C1END)
            plt.show()
        elif (iter_n+1)*batch-1==ep_n:
            C1 = '\033[93m'
            C1END = '\033[0m'
            print(C1 + "PLOTTING:" + C1END)
            plt.show()



read_file(all=False) #Eğer True seçerseniz hocam tüm batchleri çizdirmeye çalışacaktır.. Düşük Batch sayıları için uydundur.
