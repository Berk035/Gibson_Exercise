import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
import pandas as pd
import sys, os

path_main = os.path.join(os.path.expanduser("~"),"PycharmProjects/Gibson_Exercise/examples/plot_result/results_new_arch")
SAVE_PATH = os.path.expanduser("~")
# name_file_d = ["(DEPTH)_(CNN)","(DEPTH+SENSOR)_(CNN+MLP)","(DEPTH+SENSOR)_(ODE+MLP)","(DEPTH+SENSOR)_(RESNET+MLP)"]
name_file_d = cl = ["RESNET+SENSOR","NODE+SENSOR","RESNET+SENSOR(nodrop)","NODE+SENSOR(nodrop)"]
# name_file_d = cl = ["(DEPTH)_(CNN)","(DEPTH)_(ODE)"]
# name_file_r = ["(RGB+DEPTH+SENSOR)_(CNN+MLP)","(RGB+DEPTH+SENSOR)_(ODE+MLP)","(RGB+DEPTH)_(CNN)"]
# cl = ["(DEPTH+SENSOR)_(CNN+MLP)","(DEPTH+SENSOR)_(CNN+MLP)_NCL"]
# cl = ["(DEPTH)_(CNN)","(DEPTH)_(ODE)"]

name_file = os.path.join(os.path.expanduser("~"),"PycharmProjects/Gibson_Exercise")
path_file = os.path.join(path_main,(name_file+"/models/iterations/values.csv"))


def plot_csv(name_file=None, debug=False):
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
		plt.legend(loc='lower right',fontsize='x-small', title_fontsize='40')
		plt.fill_between(data["Iteration"], d3 - np.std(d3), d3 + np.std(d3), alpha=0.1)

		fig.add_subplot(axes[1, 1])
		d4 = data["PolSur"]
		sns.lineplot(x="Iteration", y="PolSur", data=data, ax=axes[1,1], label=entry)
		plt.legend(loc='lower right',fontsize='x-small', title_fontsize='40')
		plt.fill_between(data["Iteration"], d4 - np.std(d4), d4 + np.std(d4), alpha=0.1)
		fig.tight_layout()

	plt.savefig(os.path.join(SAVE_PATH + '/final_values.png'))
	if debug:
		plt.show()


def plot_spl(name_file=None, debug=False):
	"Plotting iterations vs reward, entrophy loss,value loss graphs"
	C1 = '\033[94m'
	C1END = '\033[0m'
	print(C1 + "PLOTTING SUCCESS:" + C1END)

	sns.set(style="darkgrid", context="paper")
	fig, axes = plt.subplots(figsize=(8, 4), nrows=1, ncols=2)
	plt.subplots_adjust(wspace=1, hspace=1)

	for entry in name_file:
		path_file = os.path.join(path_main,(entry+"/models/success/spl.csv"))
		data = pd.read_csv(path_file,nrows=8000)

		d1 = data["Success Rate"]
		d2 = data["SPL"]

		fig.add_subplot(axes[0])
		sns.lineplot(x="Episode", y=d1, data=data, ax=axes[0], label=entry)
		plt.legend(loc='upper left', fontsize='x-small', title_fontsize='40')
		plt.fill_between(data["Episode"], d1 - np.std(d1), d1 + np.std(d1), alpha=0.2)


		fig.add_subplot(axes[1])
		sns.distplot(d2, ax=axes[1], label=entry, kde_kws={'bw':1.5})
		plt.legend(loc='upper left', fontsize='x-small', title_fontsize='40')
		fig.tight_layout()

	#plt.savefig(os.path.join(SAVE_PATH + 'success_rate.png'))
	if debug:
		plt.show()


def main(raw_args=None):
	"This function shows that analysis of training process"
	deb = bool(1)

	plot_csv(name_file=name_file_d, debug=deb) #Reward Plotting
	#plot_csv(name_file=name_file, debug=deb)  # Reward Plotting
	plot_spl(name_file=cl,debug=deb) #Success Rate Plotting

if __name__ == '__main__':
	main()
