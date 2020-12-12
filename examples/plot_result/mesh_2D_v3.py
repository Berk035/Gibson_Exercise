import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
import pandas as pd
import sys, os

path_main = os.path.join(os.path.expanduser("~"),"PycharmProjects/Gibson_Exercise/examples/plot_result/results")
SAVE_PATH = os.path.expanduser("~")
name_file = ["CNN_DEPTH","Fuse_RGB","ELM_SENSOR","Fuse_DEPTH","Fuse_DEPTH_CL(False)","MLP_SENSOR","ODE_DEPTH_150","ResNet_DEPTH"]
#path_file = os.path.join(path_main,(name_file+"/models/iterations/values.csv"))


def plot_csv(debug=False):
	"Plotting iterations vs reward, entrophy loss,value loss graphs"
	C1 = '\033[94m'
	C1END = '\033[0m'
	print(C1 + "PLOTTING ITERATION:" + C1END)

	sns.set(style="whitegrid", context="paper")

	fig, axes = plt.subplots(figsize=(12,12), nrows=2, ncols=2)
	plt.subplots_adjust(wspace=1, hspace=1)


	for entry in name_file:
		path_file = os.path.join(path_main,(entry+"/models/iterations/values.csv"))

		data = pd.read_csv(path_file)
		fig.add_subplot(axes[0, 0])
		sns.lineplot(x="Iteration", y="Reward", data=data, ax=axes[0,0], label=entry)
		plt.legend(loc='lower right',fontsize='x-small', title_fontsize='40')
		d1 = data["Reward"]
		plt.fill_between(data["Iteration"], d1 - np.std(d1), d1 + np.std(d1), alpha=0.1)

		fig.add_subplot(axes[0, 1])
		d2 = data["LossEnt"]
		sns.lineplot(x="Iteration", y="LossEnt", data=data, ax=axes[0,1], label=entry)
		plt.legend(loc='lower right',fontsize='x-small', title_fontsize='40')
		plt.fill_between(data["Iteration"], d2 - np.std(d2), d2 + np.std(d2), alpha=0.1)

		fig.add_subplot(axes[1, 0])
		d3 = data["LossVF"]
		sns.lineplot(x="Iteration", y="LossVF", data=data, ax=axes[1,0], label=entry)
		plt.legend(loc='upper left',fontsize='x-small', title_fontsize='40')
		plt.fill_between(data["Iteration"], d3 - np.std(d3), d3 + np.std(d3), alpha=0.1)

		fig.add_subplot(axes[1, 1])
		d4 = data["PolSur"]
		sns.lineplot(x="Iteration", y="PolSur", data=data, ax=axes[1,1], label=entry)
		plt.legend(loc='upper left',fontsize='x-small', title_fontsize='40')
		plt.fill_between(data["Iteration"], d4 - np.std(d4), d4 + np.std(d4), alpha=0.1)
		fig.tight_layout()

	#plt.savefig(os.path.join(SAVE_PATH + '/rew_values.png'))
	if debug:
		plt.show()


def main(raw_args=None):
	"This function shows that analysis of training process"
	deb = bool(1)

	plot_csv(debug=deb) #Reward Plotting

if __name__ == '__main__':
	main()
