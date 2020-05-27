import matplotlib.pyplot as plt
import meshcut
import numpy as np
import pandas
import seaborn as sns


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


def mesh(model_id="", episode=0):
	fn = "/home/berk/PycharmProjects/Gibson_Exercise/gibson/assets/dataset/" + str(model_id) + "/mesh_z_up.obj"
	verts, faces = load_obj(fn)
	z = np.min(verts[:, -1]) + 0.5  # your robot height
	cross_section = meshcut.cross_section(verts, faces, plane_orig=(0, 0, z), plane_normal=(0, 0, 1))

	# plt.figure()
	plt.figure(figsize=(12, 8))
	plt.subplot(1, 2, 1)
	for item in cross_section:
		for i in range(len(item) - 1):
			plt.plot(item[i:i + 2, 0], item[i:i + 2, 1], 'k')

	plt.title('Map of Navigation')
	plt.xlabel('X Position');
	plt.ylabel('Y Position')
	plt.grid(True)

	show_waypoint = 1
	if show_waypoint:
		df = pandas.read_csv('euharlee_waypoints.csv')
		points = df.values
		length = len(points)
		sp = np.zeros((length, 3));
		ang = np.zeros((length, 1));
		gp = np.zeros((length, 3))
		complexity = np.zeros((length, 1))
		for r in range(length):
			sp[r] = np.array([points[r][2], points[r][3], points[r][4]])
			ang[r] = np.array([points[r][5]])
			gp[r] = np.array([points[r][6], points[r][7], points[r][8]])
			complexity[r] = np.array([points[r][10] / points[r][9]])

		print("ModelID:", points[0][1])
		plt.plot(sp[episode % len(sp) - 1][0], sp[episode % len(sp) - 1][1], 'r*')
		plt.plot(gp[episode % len(sp) - 1][0], gp[episode % len(sp) - 1][1], 'g*')
		print("(%i) Nav Complexity ---> %.3f" % (episode, complexity[episode % len(sp) - 1]))

		'''for k in range(length):
			plt.plot(sp[k][0], sp[k][1], 'r*')
			plt.plot(gp[k][0], gp[k][1], 'r*')
			print("%i Nav Complexity ---> %.3f"%(k,complexity[k]))'''


# plt.show()

def read_file(ep_n=0, target=None):
	target_pos = target
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

	counter = 0;
	sum = 0
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

	i_x = x_pos[0];
	i_y = y_pos[0]
	# t_x, t_y, _ = target_pos

	# i_x=-5; i_y=7

	print("Episode: %i" % ep_n)
	print("Mean Rew: %.2f" % np.mean(tot_ret))
	# print("Start Pos:", [x_pos[0], y_pos[0], target_pos[2]])
	# print("Goal Pos:", target_pos)
	print("---------------------------")

	plt.title('Husky Path <%i>' % ep_n)
	plt.xlabel('X Pos');
	plt.ylabel('Y Pos')
	plt.annotate('S: ({:.3g}),({:.3g})'.format(i_x, i_y), xy=(i_x, i_y), xytext=(i_x, i_y),
				 arrowprops=dict(facecolor='black', shrink=0.05))
	# plt.annotate('T: ({:.3g}),({:.3g})'.format(t_x,t_y), xy=(t_x, t_y), xytext=(t_x, t_y), arrowprops=dict(facecolor='blue', shrink=0.05))
	plt.plot(x_pos, y_pos, 'r')
	# plt.plot(target_pos[0], target_pos[1], 'g*')
	# plt.plot(target_pos[1][0], target_pos[1][1], 'g*')
	plt.grid(True)

	plt.subplot(1, 2, 2)
	plt.title('Episode Reward <%i>' % ep_n)
	plt.xlabel('Frames')
	plt.plot(fn, tot_ret, 'b')
	plt.grid(True)
	# plt.tight_layout()
	ep_pos.close()
	plt.show()


def plot_iter(fast):
	"Plotting iterations vs reward, entrophy loss,value loss graphs"
	C1 = '\033[93m'
	C1END = '\033[0m'
	print(C1 + "PLOTTING:" + C1END)

	ep_pos = open(r"/home/berk/PycharmProjects/Gibson_Exercise/gibson/utils/models/iterations/values" + "_" +
				  ".txt", "r")

	'''ep_pos_elm = open(r"/home/berk/PycharmProjects/Gibson_Exercise/gibson/utils/models/iterations/values" + "_" +
				  "elm.txt", "r")

	counter_elm = 0; rew_elm = []; it_zone_elm = []; std_elm = []
	for line in ep_pos_elm:
		raw_elm = line.split(";")
		iter_elm = raw_elm[0]; temp_1_elm = int(iter_elm.replace('Iteration:', ''))
		step_elm = raw_elm[1]; temp_2_elm = int(step_elm.replace('TimeSteps:', ''))
		rewards_elm = raw_elm[2]; temp_3_elm = rewards_elm.replace('Rew:', ''); temp_4_elm = float(temp_3_elm.replace('\n', ''))

		it_zone_elm.append(temp_1_elm)
		rew_elm.append(temp_4_elm)
		std_elm.append(np.std(rew_elm[-10:]))
		counter_elm += 1
'''
	if fast:
		counter = 0; rew = [];	it_zone = [];	std = [];	ep_length = [];	hold = 0
		for line in ep_pos:
			raw = line.split(";")
			iter = raw[0];	temp_1 = int(iter.replace('Iteration:', ''))
			step = raw[1];	temp_2 = int(step.replace('TimeSteps:', ''))
			rewards = raw[2];	temp_3 = rewards.replace('Rew:', '');	temp_4 = float(temp_3.replace('\n', ''))
			ep_length.append(temp_2 - hold)
			hold = temp_2

			it_zone.append(temp_1)
			rew.append(temp_4)
			std.append(np.std(rew[-10:]))
			counter += 1


		plt.figure(figsize=(8, 8))
		plt.subplot(2, 2, 1)
		plt.title('Total Reward')
		plt.xlabel('Iterations')
		#sns.lineplot(it_zone,rew,legend='brief',label="Average Reward",palette=True)
		#plt.plot(it_zone_elm, rew_elm, color='red', label="AÖM Ödül Ort.")
		plt.plot(it_zone, rew, color='blue', label="MLP Mean Reward")
		plt.grid(True)
		plt.legend()
		plt.subplot(2, 2, 2)
		plt.title('Standard Deviations')
		plt.xlabel('Iterations')
		plt.plot(it_zone[-10:], std[-10:], color='blue', label="MLP Std. Dev")
		# plt.plot(it_zone_elm[-10:], std_elm[-10:], color='red', label="AÖM Standart Sapma")
		plt.grid(True)
		plt.legend()

		'''
		plt.subplot(2, 2, 3)
		plt.title('Timesteps per Iteration, Rew Mean %.2f' % np.mean(rew[:]))
		plt.xlabel('Timesteps')
		plt.hist(ep_length, bins=50, color='black')
		plt.grid(True)
		plt.tight_layout()'''

	else:
		counter = 0;
		rew = [];
		it_zone = [];
		std = [];
		ent = [];
		vf = [];
		time = []
		for line in ep_pos:
			raw = line.split(";")
			iter = raw[0];
			temp_1 = int(iter.replace('Iteration:', ''))
			step = raw[1];
			temp_2 = int(step.replace('TimeSteps:', ''))
			rewards = raw[2];
			temp_3 = float(rewards.replace('Rew:', ''))
			ent_t = raw[3];
			temp_4 = float(ent_t.replace('LossEnt:', ''))
			vf_t = raw[4];
			temp_5 = vf_t.replace('LossVF:', '');
			temp_6 = float(temp_5.replace('\n', ''))
			# time_t = raw[5]; hold = time_t.replace('Time:',''); temp_6 = float(hold.replace('\n', ''))

			it_zone.append(temp_1)
			rew.append(temp_3)
			std.append(np.std(rew[-10:]))
			# std.append(np.std(rew))
			ent.append(temp_4)
			vf.append(temp_5)
			time.append(temp_6)

			counter += 1



		plt.figure(figsize=(12, 8))
		plt.subplot(2, 2, 1)
		plt.title('Total Reward')
		plt.xlabel('Iterations')
		plt.plot(it_zone, rew, 'b')
		plt.grid(True)
		plt.subplot(2, 2, 2)
		plt.title('Std Dev')
		plt.xlabel('Iterations')
		plt.plot(it_zone[-10:], std[-10:], 'r')
		# plt.plot(it_zone, std, 'r')
		plt.grid(True)
		plt.subplot(2, 2, 3)
		plt.title('Total Entrophy Loss')
		plt.xlabel('Iterations')
		plt.plot(it_zone, ent, 'k')
		plt.grid(True)
		plt.subplot(2, 2, 4)
		plt.title('Total Value Loss')
		plt.xlabel('Iterations')
		plt.plot(it_zone, vf, 'g')
		plt.grid(True)
		plt.tight_layout()

	# time_x = (time)
	# plt.figure(2)
	# plt.plot(it_zone, time_x)

	plt.show()


def main():
	r = 50  # Number of Navigation Scenario
	k = 8034  # Number of episode
	use_data = False
	map = 3

	C1 = '\033[91m'
	C1END = '\033[0m'
	print(C1 + "PLOTTING:" + C1END)

	# Reward Plotting
	fast_mode = True
	plot_iter(fast_mode)

	if use_data:
		df = pandas.read_csv('pointgoal_gibson_full_v2.csv', skiprows=1100, nrows=100)  # Read DataFrame for Aloha
		points = df.values
		start = [points[r][0], points[r][1], points[r][2]]
		goal = [points[r][3], points[r][4], points[r][5]]
		mesh(model_id="Aloha")
		read_file(ep_n=3000, target=goal)
	else:
		start = [0, -5, 0.3]  # For room1
		start_2 = [0, 2, 0.3]  # For room2
		goal = [-0.77, -4.571, 0.3]
		for x in range(map):
			mesh(model_id="Euharlee", episode=k)
			read_file(ep_n=k, target=goal)
			k += 1


if __name__ == '__main__':
	# mesh(model_id="Euharlee")
	main()
