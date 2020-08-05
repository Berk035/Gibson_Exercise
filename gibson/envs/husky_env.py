from gibson.envs.env_modalities import CameraRobotEnv, BaseRobotEnv, SemanticRobotEnv
from gibson.envs.env_bases import *
from gibson.core.physics.robot_locomotors import Husky
from transforms3d import quaternions
import os
import numpy as np
import sys
import pybullet as p
from gibson.core.physics.scene_stadium import SinglePlayerStadiumScene
import pybullet_data
import cv2

THRESHOLD = 0.4
FLAG_LIMIT = 3000
CALC_OBSTACLE_PENALTY = 1
CALC_GEODESIC_REW = 0

tracking_camera = {
    'yaw': 110,
    'z_offset': 0.5,
    'distance': 1,
    'pitch': -20
}

tracking_camera_top = {
    'yaw': 20,  # demo: living room, stairs
    'z_offset': 0.5,
    'distance': 1,
    'pitch': -20
}


class HuskyNavigateEnv(CameraRobotEnv):
    """Specfy navigation reward
    """

    def __init__(self, config, gpu_idx=0):
        self.config = self.parse_config(config)
        assert (self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")

        CameraRobotEnv.__init__(self, self.config, gpu_idx,
                                scene_type="stadium" if self.config["model_id"] == "stadium" else "building",
                                tracking_camera=tracking_camera)

        self.robot_introduce(Husky(self.config, env=self))
        #WARNING:Robot tanımının yapıldığı yer devam etmeli, aksi taktirde 'Bad inertia hatası'
        self.scene_introduce()
        self.total_reward = 0
        self.total_frame = 0
        self.eps_so_far = 0
        self.hold_rew = 0
        self.success = 0
        self.SR = 0
        self.SPL = 0

        self.position = []
        self.old_pos = []
        self.shortest_path = 0
        self.actual_path = 0  # Offset for beggining


    def add_text(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        # font = cv2.FONT_HERSHEY_PLAIN
        # x,y,z = self.robot.get_position()
        # r,p,ya = self.robot.get_rpy()

        cv2.putText(img, 'Reward:{0:.3f}'.format(self.hold_rew), (10, 110), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(img, 'x:{0:.2f} y:{1:.2f} z:{2:.2f}'.format(x,y,z), (10, 100), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(img, 'ro:{0:.4f} pth:{1:.4f} ya:{2:.4f}'.format(r,p,ya), (10, 40), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(img, 'potential:{0:.4f}'.format(self.potential), (10, 60), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(img, 'fps:{0:.4f}'.format(self.fps), (10, 80), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        return img

    def _rewards(self, action=None, debugmode=False):


        a = action
        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        wall_contact = []
        for i, f in enumerate(self.parts):
            if self.parts[f] not in self.robot.feet:
                wall_contact += [pt for pt in self.robot.parts[f].contact_list() if pt[6][2] > 0.15]
        wall_collision_cost = self.wall_collision_cost * len(wall_contact)

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        # joints_at_limit_cost = 0

        self.shortest_path = np.linalg.norm([np.array(self.robot.target_pos)
                                             - np.array(self.robot.initial_pos)])

        self.position = self.robot.get_position()
        if len(self.old_pos)==0:
            displacement = 0
        else:
            displacement = np.linalg.norm([self.position[1] - self.old_pos[1], self.position[0] - self.old_pos[0]])
        self.actual_path += displacement
        self.old_pos = self.position

        close_to_target = 0
        # x_tar, y_tar, z_tar = self.robot.target_pos
        if self.robot.dist_to_target() <= THRESHOLD:
            close_to_target = 0.5

        steering_cost = self.robot.steering_cost(a)
        angle_cost = self.robot.angle_cost()
        feet_collision_cost = self.robot.feet_col(self.ground_ids, self.foot_collision_cost)

        obstacle_penalty = 1
        if CALC_OBSTACLE_PENALTY and self._require_camera_input:
            obstacle_penalty = get_obstacle_penalty(self.robot, self.render_depth)

        roll, pitch, yaw = self.robot.get_rpy()
        # (vx,vy,vz) = self.robot.get_velocity()
        height = self.position[2]
        alive = float(self.robot.alive_bonus(height, pitch))

        # progress = -0.1

        rewards = [
            # WARNING:all rewards have rew/frame units and close to 1.0
            alive,  # It has 1 or 0 values
            progress,  # It calculates between two frame for target distance
            obstacle_penalty,  # TODO: Aldığı değerlerin etkisi çok düşük
            angle_cost,  # It has -0.6~0 values for tend to target
            wall_collision_cost,  # It  has 0.3~0.1 values edit:0.5
            steering_cost,  # It has -0.1 values when the agent turns
            close_to_target,  # It returns reward step by step between 0.25~0.75
            # feet_collision_cost, #Tekerlerin model üzerinde iç içe girmesini engellemek için yazılmış ancak hata var..
            # joints_at_limit_cost #Jointlerin 0.99 üzerindeki herbir değeri için ceza
        ]

        # Episode Recording
        record = 0
        if record:
            file_path = "/home/berk/PycharmProjects/Gibson_Exercise/gibson/utils/models/rewards"
            try:
                os.mkdir(file_path)
            except OSError:
                pass

            if self.nframe == 1:
                ep_pos = open(r"/home/berk/PycharmProjects/Gibson_Exercise/gibson/utils/models/rewards/positions" +
                              "_" + str(self.eps_count) + ".txt", "w")
            else:
                ep_pos = open(r"/home/berk/PycharmProjects/Gibson_Exercise/gibson/utils/models/rewards/positions" +
                              "_" + str(self.eps_count) + ".txt", "a")
            ep_pos.write("%i;%.3f" % (self.nframe, path_ratio) + "\n")
            ep_pos.close()

        '''img_depth = self.add_text(self.render_depth)
        path="/home/berk/PycharmProjects/Gibson_Exercise/examples/train/output_frames"
        try:
            os.mkdir(path)
        except OSError:
            pass
        cv2.imshow('Depth', img_depth)
        cv2.waitKey(1)
        cv2.imwrite(os.path.join(path, 'FRAME_%i.jpg') %self.nframe, img_depth)'''

        if (debugmode):
            print("------------------------")
            # print("Episode Frame: {}".format(self.nframe))
            # print("Target Position: x={:.3f}, y={:.3f}, z={:.3f}".format(x_tar,y_tar,z_tar))
            # print("Position: x={:.3f}, y={:.3f}, z={:.3f}".format(self.position[0],self.position[1],self.position[2]))
            # print(self.robot.geo_dist_target([2,1],[1,1]))
            # print("Orientation: r={:.3f}, p={:.3f}, y={:.3f}".format(roll, pitch, yaw))
            # print("Velocity: x={:.3f}, y={:.3f}, z={:.3f}".format(vx, vy, vz))
            # print("Progress: {:.3f}".format(progress))
            # print("Steering cost: {:.3f}" .format(steering_cost))
            # print("Angle Cost: {:.3f}".format(angle_cost))
            # print("Joints_at_limit_cost: {:.3f}" .format(joints_at_limit_cost))
            # print("Feet_collision_cost: {:.3f}" .format(feet_collision_cost))
            # print("Wall contact points: {:.3f}" .format(len(wall_contact)))
            # print("Collision cost: {:.3f}" .format(wall_collision_cost))
            # print("Obstacle penalty: {:.3f}".format(obstacle_penalty))
            # print("Close to target: {:.2f}".format(close_to_target))
            # print("ACTUAL:%.2f\t"%self.actual_path + str("SHORTEST:%.2f"%self.shortest_path))
            # print("Rewards: {:.3f} " .format(sum(rewards)))
            # print("Total Eps Rewards: {:.3f} ".format(self.eps_reward))
            # print("-----------------------")

        self.hold_rew = sum(rewards)
        return rewards

    def _termination(self, debugmode=True):
        height = self.robot.get_position()[2]
        pitch = self.robot.get_rpy()[1]
        alive = float(self.robot.alive_bonus(height, pitch)) > 0
        # alive = len(self.robot.parts['top_bumper_link'].contact_list()) == 0

        done = not alive or self.nframe > (self.config['n_step'] - 1) or height < 0 or \
        self.robot.dist_to_target() <= THRESHOLD
        if done:
            self.eps_so_far += 1
            self.actual_path = 0
            self.position = []
            self.old_pos = []
            self.shortest_path = 0
            self.SPL = 0

            if self.robot.dist_to_target() <= THRESHOLD:
                self.success=1
                self.SR+= self.success
                #TODO:This part requires revising
                self.SPL = self.success * (self.shortest_path/max(self.actual_path,self.shortest_path))

            if debugmode:
                CRED = '\033[91m'
                CEND = '\033[0m'
                print(CRED + "Episode reset!" + CEND)
                print("Episodes -----> %i/%s" % ((self.eps_so_far % int(self.config["n_episode"])),
                                                  str(self.config["n_episode"])))
                print("SR: %.2f" % (self.SR/self.eps_so_far*100))
                print("SPL: %.2f" % self.SPL)

        return done

    def _flag_reposition(self):
        target_pos = self.robot.target_pos

        self.flag = None
        if self.gui and not self.config["display_ui"]:
            self.visual_flagId = p.createVisualShape(p.GEOM_MESH,
                                                     fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'),
                                                     meshScale=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 0.7])
            self.last_flagId = p.createMultiBody(baseVisualShapeIndex=self.visual_flagId, baseCollisionShapeIndex=-1,
                                                 basePosition=[target_pos[0], target_pos[1], 0.5])

    def _reset(self):
        self.total_frame = 0
        self.total_reward = 0
        obs = CameraRobotEnv._reset(self)
        self._flag_reposition()
        return obs

    ## openai-gym v0.10.5 compatibility
    step = CameraRobotEnv._step


class HuskyNavigateSpeedControlEnv(HuskyNavigateEnv):
    """Specfy navigation reward
    """

    def __init__(self, config, gpu_idx=0):
        # assert(self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")
        HuskyNavigateEnv.__init__(self, config, gpu_idx)
        self.robot.keys_to_action = {
            (ord('s'),): [-0.5, 0],  ## backward
            (ord('w'),): [0.5, 0],  ## forward
            (ord('d'),): [0, -0.5],  ## turn right
            (ord('a'),): [0, 0.5],  ## turn left
            (): [0, 0]
        }

        self.base_action_omage = np.array([-0.001, 0.001, -0.001, 0.001])
        self.base_action_v = np.array([0.001, 0.001, 0.001, 0.001])
        self.action_space = gym.spaces.Discrete(5)
        # control_signal = -0.5
        # control_signal_omega = 0.5
        self.v = 0
        self.omega = 0
        self.kp = 100
        self.ki = 0.1
        self.kd = 25
        self.ie = 0
        self.de = 0
        self.olde = 0
        self.ie_omega = 0
        self.de_omega = 0
        self.olde_omage = 0

    def _step(self, action):
        control_signal, control_signal_omega = action
        self.e = control_signal - self.v
        self.de = self.e - self.olde
        self.ie += self.e
        self.olde = self.e
        pid_v = self.kp * self.e + self.ki * self.ie + self.kd * self.de

        self.e_omega = control_signal_omega - self.omega
        self.de_omega = self.e_omega - self.olde_omage
        self.ie_omega += self.e_omega
        pid_omega = self.kp * self.e_omega + self.ki * self.ie_omega + self.kd * self.de_omega

        obs, rew, env_done, info = HuskyNavigateEnv.step(self,
                                                         pid_v * self.base_action_v + pid_omega * self.base_action_omage)

        self.v = obs["nonviz_sensor"][3]
        self.omega = obs["nonviz_sensor"][-1]

        return obs, rew, env_done, info

    ## openai-gym v0.10.5 compatibility
    step = _step


class HuskyGibsonFlagRunEnv(CameraRobotEnv):
    """Specfy flagrun reward
    """

    def __init__(self, config, gpu_idx=0):
        self.config = self.parse_config(config)
        print(self.config["envname"])
        assert (self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")
        CameraRobotEnv.__init__(self, self.config, gpu_idx,
                                scene_type="building",
                                tracking_camera=tracking_camera)

        self.robot_introduce(Husky(self.config, env=self))
        self.scene_introduce()

        # WARNING: While debugging mode, it must be pass directly!!

        self.total_reward = 0
        self.total_frame = 0
        self.flag_timeout = 1
        self.visualid = -1
        self.lastid = None
        self.gui = self.config["mode"] == "gui"
        self.waypoint = 0

        if self.gui:
            self.visualid = p.createVisualShape(p.GEOM_MESH,
                                                fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'),
                                                meshScale=[0.2, 0.2, 0.2], rgbaColor=[1, 0, 0, 0.7])
        self.colisionid = p.createCollisionShape(p.GEOM_MESH,
                                                 fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'),
                                                 meshScale=[0.2, 0.2, 0.2])

        self.lastid = None
        self.obstacle_dist = 100

    def _reset(self):
        obs = CameraRobotEnv._reset(self)
        return obs

    def _flag_reposition(self):
        # self.walk_target_x = self.np_random.uniform(low=-self.scene.stadium_halflen,
        #                                            high=+self.scene.stadium_halflen)
        # self.walk_target_y = self.np_random.uniform(low=-self.scene.stadium_halfwidth,
        #                                            high=+self.scene.stadium_halfwidth)
        force_x = self.np_random.uniform(-150, 150)
        force_y = self.np_random.uniform(-150, 150)

        # x_range = [-2.0,2.0]
        # y_range = [-2.0,2.0]
        # z_range = [0,0]
        # more_compact = 0.5  # set to 1.0 whole football field
        # self.walk_target_x *= more_compact
        # self.walk_target_y *= more_compact

        startx, starty, _ = self.robot.get_position()

        self.flag = None
        # self.flag = self.scene.cpp_world.debug_sphere(self.walk_target_x, self.walk_target_y, 0.2, 0.2, 0xFF8080)
        self.flag_timeout = 3000 / self.scene.frame_skip
        # print('targetxy', self.flagid, self.walk_target_x, self.walk_target_y, p.getBasePositionAndOrientation(self.flagid))
        # p.resetBasePositionAndOrientation(self.flagid, posObj = [self.walk_target_x, self.walk_target_y, 0.5], ornObj = [0,0,0,0])
        if self.lastid:
            p.removeBody(self.lastid)

        self.lastid = p.createMultiBody(baseMass=1, baseVisualShapeIndex=self.visualid,
                                        baseCollisionShapeIndex=self.colisionid, basePosition=[startx, starty, 0.5])
        # p.applyExternalForce(self.lastid, -1, [force_x,force_y,50], [0,0,0], p.LINK_FRAME)
        _, orn = p.getBasePositionAndOrientation(self.lastid)

        p.resetBasePositionAndOrientation(self.lastid, [-3.883, 0.993, 0.5], orn)
        if self.waypoint == 1:
            p.resetBasePositionAndOrientation(self.lastid, [-3.5, -1, 0.5], orn)

        '''
        force_x = self.np_random.uniform(-300, 300)
        force_y = self.np_random.uniform(-300, 300)
        p.applyExternalForce(self.lastid, -1, [force_x, force_y, 25], [0, 0, 0], p.LINK_FRAME)

        pos = self.robot.get_position()
        new_pos = [pos[0] + self.np_random.uniform(low=x_range[0], high=x_range[1]),
                   pos[1] + self.np_random.uniform(low=y_range[0], high=y_range[1]),
                   pos[2] + self.np_random.uniform(low=z_range[0], high=z_range[1])]
        self.lastid = p.createMultiBody(baseMass=1, baseVisualShapeIndex=self.visualid,
                                        baseCollisionShapeIndex=self.colisionid,
                                        basePosition=[new_pos[0], new_pos[1], new_pos[2]])
        '''

        '''self.lastid = p.createMultiBody(baseMass=1, baseVisualShapeIndex=self.visualid,
                                        baseCollisionShapeIndex=self.colisionid,
                                        basePosition=[-3.883, 0.993, 0.5])'''

        # self.lastid = p.createMultiBody(baseMass=1, baseVisualShapeIndex=self.visualid,
        #                                baseCollisionShapeIndex=self.colisionid,
        #                               basePosition=[startx, starty, 0.5])

        '''if self.waypoint == 1:
            self.lastid = p.createMultiBody(baseMass=1, baseVisualShapeIndex=self.visualid,
                                            baseCollisionShapeIndex=self.colisionid,
                                            basePosition=[-3.5, -1, 0.5])'''

        # ball_xyz, _ = p.resetBasePositionAndOrientation(self.lastid,new_pos,orn)
        ball_xyz, _ = p.getBasePositionAndOrientation(self.lastid)

        # self.robot.set_target_position(ball_xyz)
        self.robot.walk_target_x = ball_xyz[0]
        self.robot.walk_target_y = ball_xyz[1]

    def _rewards(self, action=None, debugmode=False):
        a = action
        potential_old = self.potential
        self.potential = self.robot.calc_potential()

        if self.flag_timeout > FLAG_LIMIT:  # Is it time required for being cube is totally placed to env?
            progress = 0
        else:
            progress = float(self.potential - potential_old)

        # prog_scale = 5
        # progress = prog_scale * float(self.potential - potential_old)

        alive = len(self.robot.parts['top_bumper_link'].contact_list())
        if alive == 0:
            alive_score = 0.1
        else:
            alive_score = -0.1

        wall_contact = []
        for i, f in enumerate(self.parts):
            if self.parts[f] not in self.robot.feet:
                wall_contact += [pt for pt in self.robot.parts[f].contact_list() if pt[6][2] > 0.15]
        wall_collision_cost = self.wall_collision_cost * len(wall_contact)

        close_target = 0
        if self.robot.walk_target_dist < 0.8:
            close_target = 0.5

        obstacle_penalty = 0
        if self.obstacle_dist < 0.7:
            obstacle_penalty = self.obstacle_dist - 0.7

        # print("Obs dist: %.3f, Obs Pen: %.3f" % (self.obstacle_dist, obstacle_penalty))

        '''
        obstacle_penalty = 0
        if CALC_OBSTACLE_PENALTY and self._require_camera_input:
            obstacle_penalty = get_obstacle_penalty(self.robot, self.render_depth)
        '''

        rewards = [
            alive_score,
            progress,
            obstacle_penalty,
            # wall_collision_cost,
            # close_target
        ]
        return rewards

    def _termination(self, debugmode=False):
        alive = len(self.robot.parts['top_bumper_link'].contact_list())
        done = alive > 0 or self.nframe > self.config['n_step']
        if (debugmode):
            print("alive=")
            print(alive)
        # print(len(self.robot.parts['top_bumper_link'].contact_list()), self.nframe, done)
        self.waypoint = 0
        return done

    def _step(self, a):
        state, reward, done, meta = CameraRobotEnv._step(self, a)
        if self.flag_timeout <= 0 or (self.flag_timeout < 225 and self.robot.walk_target_dist < 0.4):
            if self.robot.walk_target_dist < 0.4:
                self.waypoint = 1
            self._flag_reposition()
        self.flag_timeout -= 1

        depth_size = 16
        if "depth" in self.config["output"]:
            depth_obs = self.get_observations()["depth"]
            x_start = int(self.windowsz / 2 - depth_size)
            x_end = int(self.windowsz / 2 + depth_size)
            y_start = int(self.windowsz / 2 - depth_size)
            y_end = int(self.windowsz / 2 + depth_size)
            self.obstacle_dist = (np.mean(depth_obs[x_start:x_end, y_start:y_end, -1]))

        # state, reward, done, meta = CameraRobotEnv._step(self, a)
        # if self.flag_timeout <= 0 or (self.flag_timeout < FLAG_LIMIT and self.robot.walk_target_dist < 0.8):
        #    self._flag_reposition()
        # self.flag_timeout -= 1

        debug = 0
        if debug:
            print("Frame: {}, FlagTimeOut: {}, Reward: {:.3f}, Distance: {:.3f}, "
                  .format(self.nframe, self.flag_timeout, reward, self.robot.walk_target_dist), done)

        return state, reward, done, meta

    ## openai-gym v0.10.5 compatibility
    step = _step


class HuskySemanticNavigateEnv(SemanticRobotEnv):
    """Specfy navigation reward
    """

    def __init__(self, config, gpu_idx=0):
        # assert(self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")
        self.config = self.parse_config(config)
        SemanticRobotEnv.__init__(self, self.config, gpu_idx,
                                  scene_type="building",
                                  tracking_camera=tracking_camera)
        self.robot_introduce(Husky(self.config, env=self))
        self.scene_introduce()

        self.total_reward = 0
        self.total_frame = 0
        self.flag_timeout = 1
        self.visualid = -1
        self.lastid = None
        self.gui = self.config["mode"] == "gui"

        if self.gui:
            self.visualid = p.createVisualShape(p.GEOM_MESH,
                                                fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'),
                                                meshScale=[0.2, 0.2, 0.2], rgbaColor=[1, 0, 0, 0.7])
        self.colisionid = p.createCollisionShape(p.GEOM_MESH,
                                                 fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'),
                                                 meshScale=[0.2, 0.2, 0.2])

        self.lastid = None
        self.obstacle_dist = 100

        self.semantic_flagIds = []

        debug_semantic = 1
        if debug_semantic and self.gui:
            for i in range(self.semantic_pos.shape[0]):
                pos = self.semantic_pos[i]
                pos[2] += 0.2  # make flag slight above object
                visualId = p.createVisualShape(p.GEOM_MESH,
                                               fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'),
                                               meshScale=[0.1, 0.1, 0.1], rgbaColor=[1, 0, 0, 0.7])
                flagId = p.createMultiBody(baseVisualShapeIndex=visualId, baseCollisionShapeIndex=-1, basePosition=pos)
                self.semantic_flagIds.append(flagId)

    def step(self, action):
        obs, rew, env_done, info = SemanticRobotEnv.step(self, action=action)
        self.close_semantic_ids = self.get_close_semantic_pos(dist_max=1.0, orn_max=np.pi / 5)
        for i in self.close_semantic_ids:
            flagId = self.semantic_flagIds[i]
            p.changeVisualShape(flagId, -1, rgbaColor=[0, 1, 0, 1])
        return obs, rew, env_done, info

    def _rewards(self, action=None, debugmode=False):
        a = action
        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        if self.flag_timeout > 225:
            progress = 0
        else:
            progress = float(self.potential - potential_old)

        if not a is None:
            electricity_cost = self.electricity_cost * float(np.abs(
                a * self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
            electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        else:
            electricity_cost = 0

        alive = len(self.robot.parts['top_bumper_link'].contact_list())
        if alive == 0:
            alive_score = 0.1
        else:
            alive_score = -0.1

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("progress")
            print(progress)

        obstacle_penalty = 0

        # print("obs dist %.3f" %self.obstacle_dist)
        if self.obstacle_dist < 0.7:
            obstacle_penalty = self.obstacle_dist - 0.7

        rewards = [
            alive_score,
            progress,
            obstacle_penalty
        ]
        return rewards

    def _termination(self, debugmode=False):
        alive = len(self.robot.parts['top_bumper_link'].contact_list())
        done = alive > 0 or self.nframe > (self.config['n_step'] - 1)
        if (debugmode):
            print("alive=")
            print(alive)
        # print(len(self.robot.parts['top_bumper_link'].contact_list()), self.nframe, done)
        return done

    def _reset(self):
        CameraRobotEnv._reset(self)
        for flagId in self.semantic_flagIds:
            p.changeVisualShape(flagId, -1, rgbaColor=[1, 0, 0, 1])


def get_obstacle_penalty(robot, depth):
    screen_sz = robot.obs_dim[0]
    screen_delta = int(screen_sz / 8)
    screen_half = int(screen_sz / 2)
    height_offset = int(screen_sz / 4)

    # obstacle_dist = (np.mean(depth))
    obstacle_dist = (
        np.mean(depth[screen_half + height_offset - screen_delta: screen_half + height_offset + screen_delta,
                screen_half - screen_delta: screen_half + screen_delta, -1]))

    obstacle_penalty = 0
    OBSTACLE_LIMIT = 1.5
    if obstacle_dist < OBSTACLE_LIMIT:
        obstacle_penalty = (obstacle_dist - OBSTACLE_LIMIT)

    debugmode = 0
    if debugmode:
        print("Obstacle screen", screen_sz, screen_delta)
        print("Obstacle distance: {:.3f}".format(obstacle_dist))
        print("Obstacle penalty: {:.3f}".format(obstacle_penalty))

        path = "/home/berk/PycharmProjects/Gibson_Exercise/examples/train/frame_penalty"
        try:
            os.mkdir(path)
        except OSError:
            pass

        clip = depth[screen_half + height_offset - screen_delta: screen_half + height_offset + screen_delta,
               screen_half - screen_delta: screen_half + screen_delta, -1]
        width, height = int(depth.shape[0]), int(depth.shape[1])
        dim = (width, height)

        # resized_clip = cv2.convertScaleAbs(cv2.resize(clip, dim, interpolation=cv2.INTER_AREA), alpha=(255.0))
        # cv2.imwrite(os.path.join(path, 'Frame_Dist_Penalty_{:.3g}-{:.3g}.jpg') .format(obstacle_dist,obstacle_penalty), resized_clip)
        depth = cv2.convertScaleAbs(depth, alpha=(255.0))
        cv2.imwrite(os.path.join(path, 'Frame_Dist_Penalty_{:.3g}-{:.3g}.jpg').format(obstacle_dist, obstacle_penalty),
                    depth)

    return obstacle_penalty
