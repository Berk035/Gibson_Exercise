import argparse
import matplotlib.pyplot as plt
import meshcut
import numpy as np
import pandas
import seaborn as sns
import pandas as pd
import sys, os
#from scipy.stats import norm

SAVE_PATH = os.path.join(os.path.expanduser("~"),'VS_Projects/Gibson_Exercise/gibson/utils/models/')
WAY_PATH = os.path.join(os.path.expanduser("~"),'VS_Projects/Gibson_Exercise/gibson/core/physics/waypoints/')

def load_obj(fn):
	verts = []
	faces = []
	with open(fn) as f:
		for line in f:
			if line[:2] == 'v ':
				verts.append(list(map(float, line.strip().split()[1:4])))
			if line[:2] == 'f ':
				face = [int(item.split('/')[0]) for item in line.strip().split()[-3:]]
				faces.append(face)
	verts = np.array(verts)
	faces = np.array(faces) - 1
	return verts, faces


def mesh(model_id="", episode=0, waypoint=False):
	C1 = '\033[91m'
	C1END = '\033[0m'
	print(C1 + "PLOTTING EPISODE:" + C1END)
	plt.style.use('default') # switches back to matplotlib style

	fn = os.path.join(os.path.expanduser("~"),
					  "VS_Projects/Gibson_Exercise/gibson/assets/dataset/") + str(model_id) + "/mesh_z_up.obj"
	verts, faces = load_obj(fn)
	z = np.min(verts[:, -1]) + 0.5  # your robot height
	cross_section = meshcut.cross_section(verts, faces, plane_orig=(0, 0, z), plane_normal=(0, 0, 1))

	plt.figure(figsize=(16, 8))
	plt.subplot(1, 2, 1)
	for item in cross_section:
		for i in range(len(item) - 1):
			plt.plot(item[i:i + 2, 0], item[i:i + 2, 1], 'k')

	plt.title('Map of Navigation')
	plt.xlabel('X Position'); plt.ylabel('Y Position')
	plt.grid(True)

	if waypoint:
		#df = pandas.read_csv('euharlee_waypoints_sort.csv')
		df = pandas.read_csv(WAY_PATH + str('euharlee_waypoints_clipped_sort.csv'))
		#df = pandas.read_csv('aloha_waypoints_clipped_sort.csv')
		points = df.values
		length = len(points)
		sp = np.zeros((length, 3)); ang = np.zeros((length, 1)); gp = np.zeros((length, 3))
		complexity = np.zeros((length, 1))
		for r in range(length):
			sp[r] = np.array([points[r][2], points[r][3], points[r][4]])
			ang[r] = np.array([points[r][5]])
			gp[r] = np.array([points[r][6], points[r][7], points[r][8]])
			complexity[r] = np.array([points[r][10]/points[r][9]])

		print("ModelID:", points[0][1])
		plt.plot(sp[episode % len(sp) - 1][0], sp[episode % len(sp) - 1][1], 'r*')
		plt.plot(gp[episode % len(sp) - 1][0], gp[episode % len(sp) - 1][1], 'g*')
		print("(%i) Nav Complexity ---> %.3f" % (episode, complexity[episode % len(sp) - 1]))

	debug=0
	if debug:
		plt.show()

def read_file(ep_n=0):
	file = os.path.join(os.path.expanduser("~"),"PycharmProjects/Gibson_Exercise/gibson/utils/models/episodes/positions")
	count = 0
	for line in open(file +	 "_" + str(ep_n) + ".txt").readlines(): count += 1

	timesteps = count
	fn = np.arange(timesteps)
	ep_pos = open(file +  "_" + str(ep_n) + ".txt", "r")

	x_pos = np.zeros(timesteps)
	y_pos = np.zeros(timesteps)
	actions = np.zeros(timesteps)
	tot_ret = np.zeros(timesteps)

	counter = 0; sum = 0
	for line in ep_pos:
		pos = line.split(";")
		nframe = pos[0]
		actions[counter] = pos[1]
		x_pos[counter] = pos[2]
		y_pos[counter] = pos[3]
		sum += float(pos[4].replace('\n', ''))
		tot_ret[counter] = sum
		counter += 1
		if nframe == str(timesteps):
			break

	i_x = x_pos[0];	i_y = y_pos[0]

	print("Episode: %i" % ep_n)
	print("Mean Rew: %.2f" % np.mean(tot_ret))
	print("---------------------------")
	plt.style.use('seaborn') # switch to seaborn style

	plt.title('Husky Path <%i>' % ep_n)
	plt.xlabel('X Pos'); plt.ylabel('Y Pos')
	plt.annotate('S: ({:.3g}),({:.3g})'.format(i_x, i_y), xy=(i_x, i_y), xytext=(i_x, i_y),
				 arrowprops=dict(facecolor='black', shrink=0.05))
	# plt.annotate('T: ({:.3g}),({:.3g})'.format(t_x,t_y), xy=(t_x, t_y), xytext=(t_x, t_y), arrowprops=dict(facecolor='blue', shrink=0.05))
	plt.plot(x_pos, y_pos, 'r')
	plt.grid(True)

	plt.subplot(1, 2, 2)
	plt.title('Episode Reward <%i>' % ep_n)
	plt.xlabel('Frames')
	plt.plot(fn, tot_ret, 'b')
	plt.grid(True)
	plt.tight_layout()

	'''plt.subplot(1, 3, 3)
	sns.distplot(actions)
	plt.title('Behaviour of Actions')
	plt.xlabel('Actions')
	plt.ylabel('# of Actions')
	plt.tight_layout()

	plt.subplot(1, 3, 3)
	sns.distplot(SPL)
	plt.title('Success Weighted by Path Length')
	plt.xlabel('Amount')
	plt.ylabel('Episodes')
	plt.tight_layout()'''

	ep_pos.close()
	#plt.show()
	plt.savefig(os.path.join(SAVE_PATH + 'ep_%i.png' % ep_n))

	# the histogram of the data
	#mu = 0; sigma=1
	#num, bins, patches = plt.hist(actions, 50, density=1)
	#y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
	#plt.plot(bins, y, '--')

def plot_csv(debug=False):
	"Plotting iterations vs reward, entrophy loss,value loss graphs"
	C1 = '\033[94m'
	C1END = '\033[0m'
	print(C1 + "PLOTTING ITERATION:" + C1END)

	data = pd.read_csv(os.path.join(os.path.expanduser("~"),"PycharmProjects/Gibson_Exercise/gibson/utils/models/iterations/values.csv"))
	#print(data.head())
	sns.set(style="darkgrid", context="paper")
	fig, axes = plt.subplots(figsize=(8,8),nrows=2, ncols=2)
	plt.subplots_adjust(wspace=1, hspace=1)
	fig.add_subplot(axes[0, 0])
	sns.lineplot(x="Iteration", y="Reward", data=data, ax=axes[0,0], color='blue')
	d1 = data["Reward"]
	plt.fill_between(data["Iteration"], d1 - np.std(d1), d1 + np.std(d1), color='b', alpha=0.2)
	fig.add_subplot(axes[0, 1])
	d2 = data["LossEnt"]
	sns.lineplot(x="Iteration", y="LossEnt", data=data, ax=axes[0,1], color='red')
	plt.fill_between(data["Iteration"], d2 - np.std(d2), d2 + np.std(d2), color='r', alpha=0.2)
	fig.add_subplot(axes[1, 0])
	d3 = data["LossVF"]
	sns.lineplot(x="Iteration", y="LossVF", data=data, ax=axes[1,0], color='darkred')
	plt.fill_between(data["Iteration"], d3 - np.std(d3), d3 + np.std(d3), color='darkred', alpha=0.2)
	fig.add_subplot(axes[1, 1])
	d4 = data["PolSur"]
	sns.lineplot(x="Iteration", y="PolSur", data=data, ax=axes[1,1], color='orange')
	plt.fill_between(data["Iteration"], d4 - np.std(d4), d4 + np.std(d4), color='orange', alpha=0.2)
	fig.tight_layout()

	plt.savefig(os.path.join(SAVE_PATH + 'rew_values.png'))
	if debug:
		plt.show()


def plot_spl(debug=False):
	"Plotting iterations vs reward, entrophy loss,value loss graphs"
	C1 = '\033[94m'
	C1END = '\033[0m'
	print(C1 + "PLOTTING SUCCESS:" + C1END)

	data = pd.read_csv(os.path.join(os.path.expanduser("~"),"PycharmProjects/Gibson_Exercise/gibson/utils/models/success/spl.csv"))
	sns.set(style="darkgrid", context="paper")
	fig, axes = plt.subplots(figsize=(8,4),nrows=1, ncols=2)
	plt.subplots_adjust(wspace=1, hspace=1)
	d1 = data["Success Rate"]
	d2 = data["SPL"]
	fig.add_subplot(axes[0])
	sns.lineplot(x="Episode", y=d1, data=data, ax=axes[0], color='blue', label="Success Rate")
	plt.fill_between(data["Episode"], d1 - np.std(d1), d1 + np.std(d1), color='b', alpha=0.2)
	fig.add_subplot(axes[1])
	sns.distplot(d2, ax=axes[1], color='red', label="SPL for each eps.", kde_kws={'bw':1.5})
	plt.legend()
	fig.tight_layout()

	plt.savefig(os.path.join(SAVE_PATH + 'success_rate.png'))
	if debug:
		plt.show()

def main(raw_args=None):
	"This function shows that analysis of training process"
	deb = bool(1)

	plot_csv(debug=deb) #Reward Plotting
	plot_spl(debug=deb) #Success Rate Plotting

	for x in range(raw_args.map):
		mesh(model_id=raw_args.model, episode=raw_args.eps, waypoint=True)
		read_file(ep_n=raw_args.eps)
		raw_args.eps += 1

if __name__ == '__main__':
	main()
