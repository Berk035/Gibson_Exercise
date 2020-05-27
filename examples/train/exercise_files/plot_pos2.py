import matplotlib.pyplot as plt
import numpy as np

iter_n = 50
epoch = 30

def read_file(ep_n=0):
    count = 0
    for line in open(r"/home/berk/PycharmProjects/Gibson_Exercise/gibson/utils/models/episodes/positions" +
                     "_" + str(ep_n) + ".txt").readlines(): count += 1
    timesteps = count
    fn = np.arange(timesteps)

    ep_pos = open(r"/home/berk/PycharmProjects/Gibson_Exercise/gibson/utils/models/episodes/positions" +
                    "_" + str(ep_n) + ".txt", "r")

    x_pos = np.zeros(timesteps)
    y_pos = np.zeros(timesteps)
    tot_ret = np.zeros(timesteps)

    counter = 0; sum = 0
    for line in ep_pos:
        pos = line.split(";")
        nframe = pos[0]
        x_pos[counter] = pos[1]
        y_pos[counter] = pos[2]
        sum += float(pos[3].replace('\n', ''))
        tot_ret[counter] = sum
        counter += 1
        if nframe == str(timesteps):
            break
    ep_pos.close()
    return tot_ret


def main():
    sum = np.zeros(iter_n)
    dev = np.zeros(iter_n)
    hold = 0
    std = 0
    counter = 0
    for i in range(71):
       rew = read_file(ep_n=i+1)
       hold +=np.mean(rew[-10:])
       std=np.std(rew[-10:])
       if (i+1) % (epoch) == 0:
           sum[counter] = hold/epoch
           dev[counter] = std
           hold=0
           counter += 1

    fn_it = np.arange(iter_n)

    plt.figure(1, figsize=(8,4))
    plt.subplot(1, 2, 1)
    plt.title('Iteration Reward Means')
    plt.plot(fn_it, sum, 'r')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.title('Standart Deviation')
    plt.plot(fn_it, dev, 'b')
    plt.grid(True)
    plt.tight_layout()

    C1 = '\033[93m'
    C1END = '\033[0m'
    print(C1 + "PLOTTING:" + C1END)
    plt.show()

#main()