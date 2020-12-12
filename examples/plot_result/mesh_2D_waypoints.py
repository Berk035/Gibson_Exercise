import argparse
import matplotlib.pyplot as plt
import meshcut
import numpy as np
import pandas
import seaborn as sns
import pandas as pd
import sys, os
import math
#from scipy.stats import norm

SAVE_PATH = os.path.join(os.path.expanduser("~"),'PycharmProjects/Gibson_Exercise/gibson/utils/models/')
WAY_PATH = os.path.join(os.path.expanduser("~"),'PycharmProjects/Gibson_Exercise/gibson/core/physics/waypoints/')

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

def add_arrow(line, position=None, direction='right', size=15, color='dodgerblue'):

	xdata = line.get_xdata()
	ydata = line.get_ydata()

	if position is None:
		position = xdata.mean()

	start_ind = np.argmin(np.absolute(xdata - position))
	if direction == 'right':
		end_ind = start_ind + 1
	else:
		end_ind = start_ind - 1

	line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )

def mesh(model_id="", waypoint=False):
	C1 = '\033[91m'
	C1END = '\033[0m'
	print(C1 + "PLOTTING EPISODE:" + C1END)
	plt.style.use('default') # switches back to matplotlib style

	fn = os.path.join(os.path.expanduser("~"),
					  "PycharmProjects/Gibson_Exercise/gibson/assets/dataset/") + str(model_id) + "/mesh_z_up.obj"
	verts, faces = load_obj(fn)
	z = np.min(verts[:, -1]) + 0.5  # your robot height
	cross_section = meshcut.cross_section(verts, faces, plane_orig=(0, 0, z), plane_normal=(0, 0, 1))

	plt.figure(figsize=(8,8))
	for item in cross_section:
		for i in range(len(item) - 1):
			plt.plot(item[i:i + 2, 0], item[i:i + 2, 1], 'k')

	plt.title('Map of Navigation')
	plt.xlabel('X Position'); plt.ylabel('Y Position')
	plt.grid(True)

	if waypoint:
		df = pandas.read_csv(WAY_PATH + str('euharlee_waypoints_sort_test.csv'))
		#df = pandas.read_csv(WAY_PATH + str('euharlee_waypoints_clipped_sort.csv'))
		points = df.values
		length = len(points)
		sp = np.zeros((length, 3)); ang = np.zeros((length, 1)); gp = np.zeros((length, 3))
		complexity = np.zeros((length, 1))
		for r in range(length):
			sp[r] = np.array([points[r][2], points[r][3], points[r][4]])
			ang[r] = np.array([points[r][5]])
			gp[r] = np.array([points[r][6], points[r][7], points[r][8]])
			complexity[r] = np.array([points[r][10]/points[r][9]])

		for k in range(length):
			plt.plot(sp[k][0], sp[k][1], 'r*')
			plt.plot(gp[k][0], gp[k][1], 'g*')
			line=plt.plot([sp[k][0], gp[k][0]], [sp[k][1], gp[k][1]], color='dodgerblue', linewidth=1)
			m1 = (sp[k][0] + gp[k][0]) / 2
			m2 = (sp[k][1] + gp[k][1]) / 2
			plt.annotate(s='', xy=(gp[k][0],gp[k][1]), xytext=(sp[k][0],sp[k][1]), arrowprops=dict(arrowstyle='->',color='grey'))
			#plt.arrow([sp[k][0], gp[k][0]], [sp[k][1], gp[k][1]], [dx],[dy], shape='full', lw=0, length_includes_head=True, head_width=.05)
			print("%i Waypoint Navigation Complexity ---> %.3f" % (k+1,complexity[k]))

	debug=1
	if debug:
		plt.savefig(os.path.join(SAVE_PATH + 'waypoints_map_test.png'))
		#plt.savefig(os.path.join(SAVE_PATH + 'waypoints_map.png'))
		plt.show()

def main(raw_args=None):
	"This function shows that analysis of training process"
	deb = bool(0)

	mesh(model_id=raw_args.model, waypoint=True)

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--model', type=str, default="Euharlee")
	args = parser.parse_args()
	main(args)
