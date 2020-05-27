import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0,parentdir)
import pybullet_data

from gibson.data.datasets import get_model_path
from gibson.core.physics.scene_abstract import Scene
import pybullet as p


class BuildingScene(Scene):
    def __init__(self, robot, model_id, gravity, timestep, frame_skip, env = None):
        Scene.__init__(self, gravity, timestep, frame_skip, env)

        # contains cpp_world.clean_everything()
        # stadium_pose = cpp_household.Pose()
        # if self.zero_at_running_strip_start_line:
        #    stadium_pose.set_xyz(27, 21, 0)  # see RUN_STARTLINE, RUN_RAD constants
        
        filename = os.path.join(get_model_path(model_id), "mesh_z_up.obj")
        #filename = os.path.join(get_model_path(model_id), "3d", "blender.obj")
        #textureID = p.loadTexture(os.path.join(get_model_path(model_id), "3d", "rgb.mtl"))

        if robot.model_type == "MJCF":
            MJCF_SCALING = robot.mjcf_scaling
            scaling = [1.0/MJCF_SCALING, 1.0/MJCF_SCALING, 0.6/MJCF_SCALING]
        else:
            scaling  = [1, 1, 1]
        magnified = [2, 2, 2]

        collisionId = p.createCollisionShape(p.GEOM_MESH, fileName=filename, meshScale=scaling, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)


        view_only_mesh = os.path.join(get_model_path(model_id), "mesh_view_only_z_up.obj")
        if os.path.exists(view_only_mesh):
            visualId = p.createVisualShape(p.GEOM_MESH,
                                       fileName=view_only_mesh,
                                       meshScale=scaling, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        else:
            visualId = -1

        boundaryUid = p.createMultiBody(baseCollisionShapeIndex = collisionId, baseVisualShapeIndex = visualId)
        p.changeDynamics(boundaryUid, -1, lateralFriction=1)
        #print(p.getDynamicsInfo(boundaryUid, -1))
        self.scene_obj_list = [(boundaryUid, -1)]       # baselink index -1
        

        planeName = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        self.ground_plane_mjcf = p.loadMJCF(planeName)
        
        if "z_offset" in self.env.config:
            z_offset = self.env.config["z_offset"]
        else:
            z_offset = -10 #with hole filling, we don't need ground plane to be the same height as actual floors

        p.resetBasePositionAndOrientation(self.ground_plane_mjcf[0], posObj = [0,0,z_offset], ornObj = [0,0,0,1])
        p.changeVisualShape(boundaryUid, -1, rgbaColor=[168 / 255.0, 164 / 255.0, 92 / 255.0, 1.0],
                            specularColor=[0.5, 0.5, 0.5])
        
    def episode_restart(self):
        Scene.episode_restart(self)

    #WARNING: This part must be snipped from
    # https://github.com/StanfordVL/iGibson/blob/master/gibson2/core/physics/scene.py#L335-L365 to here with auxilary functions
    def get_shortest_path(self, floor, source_world, target_world, entire_path=False):
        assert self.build_graph, 'cannot get shortest path without building the graph'
        source_map = tuple(self.world_to_map(source_world))
        target_map = tuple(self.world_to_map(target_world))

        g = self.floor_graph[floor]

        if not g.has_node(target_map):
            nodes = np.array(g.nodes)
            closest_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - target_map, axis=1))])
            g.add_edge(closest_node, target_map, weight=l2_distance(closest_node, target_map))

        if not g.has_node(source_map):
            nodes = np.array(g.nodes)
            closest_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - source_map, axis=1))])
            g.add_edge(closest_node, source_map, weight=l2_distance(closest_node, source_map))

        path_map = np.array(nx.astar_path(g, source_map, target_map, heuristic=l2_distance))

        path_world = self.map_to_world(path_map)
        geodesic_distance = np.sum(np.linalg.norm(path_world[1:] - path_world[:-1], axis=1))
        path_world = path_world[::self.waypoint_interval]

        if not entire_path:
            path_world = path_world[:self.num_waypoints]
            num_remaining_waypoints = self.num_waypoints - path_world.shape[0]
            if num_remaining_waypoints > 0:
                remaining_waypoints = np.tile(target_world, (num_remaining_waypoints, 1))
                path_world = np.concatenate((path_world, remaining_waypoints), axis=0)

        return path_world, geodesic_distance


class SinglePlayerBuildingScene(BuildingScene):
    multiplayer = False
    def __init__(self, robot, model_id, gravity, timestep, frame_skip, env = None):
        BuildingScene.__init__(self, robot, model_id, gravity, timestep, frame_skip, env)



