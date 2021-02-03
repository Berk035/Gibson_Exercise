from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
from gibson.core.render.profiler import Profiler
import os
import tempfile
import cloudpickle
import zipfile, csv
import sys
import matplotlib.pyplot as plt, cv2


def save(self, path=None):
    """Save model to a pickle located at `path`"""
    if path is None:
        path = os.path.join(logger.get_dir(), "model.pkl")

    with tempfile.TemporaryDirectory() as td:
        U.save_state(os.path.join(td, "model"))
        arc_name = os.path.join(td, "packed.zip")
        with zipfile.ZipFile(arc_name, 'w') as zipf:
            for root, dirs, files in os.walk(td):
                for fname in files:
                    file_path = os.path.join(root, fname)
                    if file_path != arc_name:
                        zipf.write(file_path, os.path.relpath(file_path, td))
        with open(arc_name, "rb") as f:
            model_data = f.read()
    with open(path, "wb") as f:
        cloudpickle.dump((model_data), f)


def load(path):
    with open(path, "rb") as f:
        model_data = cloudpickle.load(f)
    sess = U.get_session()
    sess.__enter__()
    with tempfile.TemporaryDirectory() as td:
        arc_path = os.path.join(td, "packed.zip")
        with open(arc_path, "wb") as f:
            f.write(model_data)

        zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
        U.load_state(os.path.join(td, "model"))
    #return ActWrapper(act, act_params)


#Iteration döngüsü
def traj_segment_generator(pi, env, horizon, stochastic, sensor=False):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob_all = env.reset()
    ob_sensor = ob_all['nonviz_sensor']
    if 'rgb_filled' in ob_all:
        ob = np.concatenate([ob_all['rgb_filled'], ob_all["depth"]], axis=2)
    else:
        ob = ob_all['depth']
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    # Initialize history arrays
    if sensor:
        obs = np.array([ob_sensor for _ in range(horizon)])
    else:
        obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

#Episode döngüsü
    while True:
        prevac = ac
        #with Profiler("agent act"):
        if sensor:
            ac, vpred = pi.act(stochastic, ob_sensor)
        else:
            ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon

        if sensor:
            obs[i] = ob_sensor
        else:
            obs[i] = ob

        record_depth=0
        if record_depth:
            path_1 = os.path.join(os.path.expanduser("~"),"PycharmProjects/Gibson_Exercise/gibson/utils/depth_images_iteration")
            try:
                os.mkdir(path_1)
            except OSError:
                pass
            ob1 = ob * 35 + 20 #DEPTH SCALE FACTOR and DEPTH SCALE OFFSET
            overflow = ob1 > 255.
            ob1[overflow] = 255.
            cv2.imwrite(os.path.join(path_1, 'Frame_{:d}.jpg').format(t), ob1)

        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        #with Profiler("Environment step"):
        ob_all, rew, new, meta = env.step(ac)

        if 'rgb_filled' in ob_all:
            ob = np.concatenate([ob_all['rgb_filled'], ob_all["depth"]], axis=2)
        else:
            ob = ob_all['depth']
        ob_sensor = ob_all['nonviz_sensor']
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob_all = env.reset()
            if 'rgb_filled' in ob_all:
                ob = np.concatenate([ob_all['rgb_filled'], ob_all["depth"]], axis=2)
            else:
                ob = ob_all['depth']
            ob_sensor = ob_all['nonviz_sensor']
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_func, *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        vfcoeff,
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        sensor = False,
        save_name=None,
        save_per_acts=3,
        reload_name=None
        ):
    # Setup losses and stuff
    # ----------------------------------------
    if sensor:
        ob_space = env.sensor_space
    else:
        ob_space = env.observation_space
    ac_space = env.action_space

    pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_func("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epsilon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = vfcoeff*tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()

    if reload_name:
        saver = tf.train.Saver()
        saver.restore(tf.get_default_session(), reload_name)
        print("Loaded model successfully.")

    # from IPython import embed; embed()
    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True,
                                     sensor = sensor)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************" % (iters_so_far+1))

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult) 
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)            
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        elapse = time.time() - tstart
        logger.record_tabular("TimeElapsed", elapse)

        #Iteration Recording
        record = 1
        if record:
            file_path = os.path.join(os.path.expanduser("~"),"PycharmProjects/Gibson_Exercise/gibson/utils/models/iterations")
            try:
                os.mkdir(file_path)
            except OSError:
                pass

            if iters_so_far == 1:
                with open(os.path.join(os.path.expanduser("~"),'PycharmProjects/Gibson_Exercise/gibson/utils/models/iterations/values.csv'),
                          'w', newline='') as csvfile:
                    fieldnames = ['Iteration', 'TimeSteps','Reward','LossEnt','LossVF','PolSur']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writeheader()
                    writer.writerow({'Iteration':iters_so_far, 'TimeSteps':timesteps_so_far,
                                     'Reward':np.mean(rews),'LossEnt':meanlosses[4],
                                     'LossVF':meanlosses[2],'PolSur':meanlosses[1]})
            else:
                with open(os.path.join(os.path.expanduser("~"),'PycharmProjects/Gibson_Exercise/gibson/utils/models/iterations/values.csv'),
                          'a', newline='') as csvfile:
                    fieldnames = ['Iteration', 'TimeSteps', 'Reward', 'LossEnt', 'LossVF', 'PolSur']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writerow({'Iteration': iters_so_far, 'TimeSteps': timesteps_so_far,
                                     'Reward': np.mean(rews), 'LossEnt': meanlosses[4],
                                     'LossVF': meanlosses[2],'PolSur':meanlosses[1]})


        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.dump_tabular()

        load_number = 0
        if not reload_name == None:
            load_number = int(str(reload_name.split('_')[7]).split('.')[0])

        if save_name and (iters_so_far % save_per_acts == 0):
            base_path = os.path.dirname(os.path.abspath(__file__))
            print(base_path)
            out_name = os.path.join(base_path, 'models', save_name + '_' + str(iters_so_far+load_number) + ".model")
            U.save_state(out_name)
            print("Saved model successfully.")


def enjoy(env, policy_func, *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        save_name=None,
        save_per_acts=3,
        sensor=False,
        reload_name=None,
        target_pos=None
        ):
    if sensor:
        ob_space = env.sensor_space
    else:
        ob_space = env.observation_space
    ac_space = env.action_space

    pi = policy_func("pi", ob_space, ac_space)  # Construct network for new policy
    oldpi = policy_func("oldpi", ob_space, ac_space)  # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return
    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32,
                            shape=[])  # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult  # Annealed cliping parameter epsilon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # pnew / pold
    surr1 = ratio * atarg  # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg  #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in
                                                    zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()

    if reload_name:
        saver = tf.train.Saver()
        saver.restore(tf.get_default_session(), reload_name)
        print("Loaded model successfully.")

    # from IPython import embed; embed()
    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True,
                                     sensor=sensor)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************" % (iters_so_far + 1))

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"]  # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

        assign_old_eq_new()  # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        iters_so_far += 1



def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
