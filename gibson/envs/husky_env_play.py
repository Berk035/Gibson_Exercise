from gibson.envs.env_mod_2 import CameraRobotEnv, BaseRobotEnv, SemanticRobotEnv
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

CALC_OBSTACLE_PENALTY = 1
FLAG_LIMIT = 225

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
        self.scene_introduce()
        self.total_reward = 0
        self.total_frame = 0

        #self.flag_timeout = 1
        #self.visualid = -1
        #self.lastid = None
        self.gui = self.config["mode"] == "gui"

        '''if self.gui:
            self.visualid = p.createVisualShape(p.GEOM_MESH,
                                                fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'),
                                                meshScale=[0.2, 0.2, 0.2], rgbaColor=[1, 0, 0, 0.7])
        self.colisionid = p.createCollisionShape(p.GEOM_MESH,
                                                 fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'),
                                                 meshScale=[0.2, 0.2, 0.2])

        '''
        #self.lastid = None

    def _rewards(self, action=None, debugmode=False):
        a = action
        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i, f in enumerate(
                self.robot.feet):  # TODO: Maybe calculating feet contacts could be done within the robot code
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # contact_ids = set([x[2] for x in f.contact_list()])
            if (self.ground_ids & contact_ids):
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0



        wall_contact = []
        for i, f in enumerate(self.parts):
            if self.parts[f] not in self.robot.feet:
                wall_contact += [pt for pt in self.robot.parts[f].contact_list() if pt[6][2] > 0.15]

        steering_cost = self.robot.steering_cost(a)
        wall_collision_cost = self.wall_collision_cost * len(wall_contact)
        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        close_to_target = 0

        if self.robot.dist_to_target() < 1.25:
            close_to_target = 0.5

        angle_cost = self.robot.angle_cost()

        obstacle_penalty = 0
        if CALC_OBSTACLE_PENALTY and self._require_camera_input:
            obstacle_penalty = get_obstacle_penalty(self.robot, self.render_depth)


        height = self.robot.get_position()[2]
        pitch = self.robot.get_rpy()[1]
        alive = float(self.robot.alive_bonus(height, pitch))

        debugmode = 0
        if (debugmode):
            # print("Wall contact points", len(wall_contact))
            print("Collision cost", wall_collision_cost)
            # print("electricity_cost", electricity_cost)
            print("close to target", close_to_target)
            print("Obstacle penalty", obstacle_penalty)
            print("Steering cost", steering_cost)
            print("progress", progress)
            # print("electricity_cost")
            # print(electricity_cost)
            # print("joints_at_limit_cost")
            # print(joints_at_limit_cost)
            # print("feet_collision_cost")
            # print(feet_collision_cost)

        rewards = [
            alive,
            progress,
            wall_collision_cost,
            close_to_target,
            steering_cost,
            angle_cost,
            obstacle_penalty,
            # electricity_cost,
            # joints_at_limit_cost,
            feet_collision_cost
        ]
        return rewards

    def _termination(self, debugmode=False):
        height = self.robot.get_position()[2]
        pitch = self.robot.get_rpy()[1]
        alive = float(self.robot.alive_bonus(height, pitch)) > 0
        # alive = len(self.robot.parts['top_bumper_link'].contact_list()) == 0

        done = not alive or self.nframe > 250 or height < 0
        # if done:
        #    print("Episode reset")
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

    '''def _step(self, a):
        #Agent Step Function

        state, reward, done, meta = CameraRobotEnv._step(self, a)
        if self.flag_timeout <= 0 or (self.flag_timeout < FLAG_LIMIT and self.robot.walk_target_dist < 0.8):
            self._flag_reposition()
        self.flag_timeout -= 1

        return state, reward, done, meta'''

    ## openai-gym v0.10.5 compatibility
    step = CameraRobotEnv._step

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

    def _reset(self):
        obs = CameraRobotEnv._reset(self)
        return obs

    def _flag_reposition(self):
        self.lastid = None
        self.flag = None
        self.flag_timeout = 1000 / self.scene.frame_skip
        if self.lastid:
            p.removeBody(self.lastid)

        pos = self.robot.target_pos
        self.lastid = p.createMultiBody(baseMass=1, baseVisualShapeIndex=self.visualid,
                                        baseCollisionShapeIndex=self.colisionid,
                                        basePosition=[pos[0], pos[1], pos[2]])


    def _rewards(self, action=None, debugmode=False):
        a = action
        potential_old = self.potential
        self.potential = self.robot.calc_potential()


        #if self.flag_timeout > FLAG_LIMIT: #Is it time required for being cube is totally placed to env?
        #    progress = 0
        #else:
        #    progress = float(self.potential - potential_old)

        prog_scale = 5
        progress = prog_scale * float(self.potential - potential_old)

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

        #print("Obs dist: %.3f, Obs Pen: %.3f" % (self.obstacle_dist, obstacle_penalty))

        '''
        obstacle_penalty = 0
        if CALC_OBSTACLE_PENALTY and self._require_camera_input:
            obstacle_penalty = get_obstacle_penalty(self.robot, self.render_depth)
        '''

        rewards = [
            alive_score,
            progress,
            obstacle_penalty,
            #wall_collision_cost,
            close_target
        ]
        return rewards

    def _termination(self, debugmode=False):
        alive = len(self.robot.parts['top_bumper_link'].contact_list())
        done = alive > 0 or self.nframe > self.config['n_step']
        if (debugmode):
            print("alive=")
            print(alive)
        #print(len(self.robot.parts['top_bumper_link'].contact_list()), self.nframe, done)
        return done

    def _step(self, a):
        '''Agent Step Function'''

        depth_size=16
        if "depth" in self.config["output"]:
            depth_obs = self.get_observations()["depth"]
            x_start = int(self.windowsz / 2 - depth_size)
            x_end = int(self.windowsz / 2 + depth_size)
            y_start = int(self.windowsz / 2 - depth_size)
            y_end = int(self.windowsz / 2 + depth_size)
            self.obstacle_dist = (np.mean(depth_obs[x_start:x_end, y_start:y_end, -1]))

        state, reward, done, meta = CameraRobotEnv._step(self, a)
        if self.flag_timeout <= 0 or (self.flag_timeout < FLAG_LIMIT and self.robot.walk_target_dist < 0.8):
            self._flag_reposition()
        self.flag_timeout -= 1

        debug=0
        if debug:
            print("Frame: {}, FlagTimeOut: {}, Reward: {:.3f}, Distance: {:.3f}, "
                  .format(self.nframe,self.flag_timeout, reward, self.robot.walk_target_dist),done)

        return state, reward, done, meta

    ## openai-gym v0.10.5 compatibility
    step = _step


def get_obstacle_penalty(robot, depth):
    screen_sz = robot.obs_dim[0]
    screen_delta = int(screen_sz / 8)
    screen_half = int(screen_sz / 2)
    height_offset = int(screen_sz / 4)

    obstacle_dist = (np.mean(depth[screen_half + height_offset - screen_delta : screen_half + height_offset + screen_delta,
                             screen_half - screen_delta : screen_half + screen_delta, -1]))
    obstacle_penalty = 0
    OBSTACLE_LIMIT = 1.5
    if obstacle_dist < OBSTACLE_LIMIT:
        obstacle_penalty = (obstacle_dist - OBSTACLE_LIMIT)

    debugmode = 0
    if debugmode:
        # print("Obstacle screen", screen_sz, screen_delta)
        print("Obstacle distance", obstacle_dist)
        print("Obstacle penalty", obstacle_penalty)
    return obstacle_penalty