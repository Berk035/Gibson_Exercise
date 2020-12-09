import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
import pandas as pd
import sys, os

path = examples/plot_result/results

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
	sns.distplot(d2, ax=axes[1], color='red', label="SPL for each eps.")
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

if __name__ == '__main__':
	main()
