from gym.envs.robotics import rotations
from gym import utils, spaces
from gym.envs.robotics import utils as gym_robotics_utils
import numpy as np
import os
import xml.etree.ElementTree as ET
from gym import error
import copy
from gym.utils import seeding
from envs.mujoco.mujoco_utils import MujocoTrait
import datetime
import sys
import string
import logging

DEFAULT_SIZE = 500
OBJECT_OHE = {'grip': np.array([1, 0, 0, 0, 0]), 'ball': np.array([0, 1, 0, 0, 0]), 'box': np.array([0, 0, 1, 0, 0]), 
              'desk': np.array([0, 0, 0, 1, 0]), 'hammer': np.array([0, 0, 0, 0, 1])}

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("""{}. (HINT: you need to install mujoco_py, and also perform the setup 
                                       instructions here: https://github.com/openai/mujoco-py/.)""".format(e))

class MultipleFetchPickAndPlaceEnv(MujocoTrait, utils.EzPickle):
    def __init__(self, seed = None, obs_type = 'state', reward_type = 'sparse', unsupervised = True,
                 object_qty = 4, with_repeat = True, object_names = ['ball', 'box', 'desk', 'hammer']):
        self.colors = ['1 0 0 1', '0 1 0 1', '0 0 1 1', '1 1 0 1', '0 1 1 1', '1 0 1 1']
        self.tints = ['0.25 0 0 1', '0 0.25 0 1', '0 0 0.25 1', '0.25 0.25 0 1', '0 0.25 0.25 1', '0.25 0 0.25 1']
        
        self.gripper_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0
        }
        self.unsupervised = unsupervised
        self.n_substeps = 20
        self.n_actions = 4
        self.gripper_extra_height = 0.2
        self.block_gripper = False
        self.has_object = True
        self.distance_threshold = 0.1
        for object_name in object_names:
            assert object_name in ['ball', 'box', 'desk', 'hammer'],\
            'Supported items are: ball, box, desk, hammer. {} is not supported'.format(object_name)
        self.object_names = object_names
        
        self.object_qty = object_qty
        self.with_repeat = with_repeat
        self._seed = seed
        self.seed(seed)
        date_and_time = str(datetime.datetime.now()).replace(' ', '_')
        self.path_to_xmls_folder = sys.argv[0][:-7] + 'env_xmls/' + date_and_time
        os.mkdir(self.path_to_xmls_folder)

        self.created_object_names, self.goal = self._initialize_sim()

        self.action_space = spaces.Box(-1., 1., shape=(self.n_actions,), dtype='float32')
        
        self.log_path = sys.argv[0][:-7] + 'logs/' + date_and_time
        os.mknod(self.log_path)
        
        self.obs_type = obs_type
        obs, _ = self._get_obs()
        if obs_type in ['state', 'decoupled_state']:
            self.observation_space = spaces.Box(-np.inf, np.inf, shape = obs.shape, dtype = 'float32')
        elif obs_type == 'pixels':
            self.observation_space = spaces.Box(0, 255, shape = obs.shape, dtype = 'uint8')
        else:
            assert False, 'Only states or pixels are supported.'

        self.ob_info = dict(
            type='pixel',
            pixel_shape=(64, 64, 3),
        )
        self.reward_range = (-1, 0)


    def _create_multiobject_xml(self):
        prefix_to_folder = os.path.join(os.path.dirname(__file__), 'gripper', 'xmls')
        model_path = os.path.join(prefix_to_folder, 'pick_and_place.xml')
        xml_tree = ET.parse(model_path)
        root = xml_tree.getroot()
        worldbody_idx = None

        for i, child in enumerate(root):
            if 'worldbody' == child.tag:
                worldbody_idx = i
                break
        
        worldbody = root[worldbody_idx]
        
        sampled_colors_idx = self.np_random.choice(len(self.colors), self.object_qty, replace=False)
        sampled_colors = [self.colors[i] for i in sampled_colors_idx]
        sampled_tints = [self.tints[i] for i in sampled_colors_idx]
        sampled_object_names = self.np_random.choice(self.object_names, self.object_qty, replace = self.with_repeat)
        created_object_names = []
        self.created_geom_names = []
        object_postfixes = {'{}'.format(object_name): 0 for object_name in sampled_object_names}
        
        for object_name, object_color, object_tint in zip(sampled_object_names, sampled_colors, sampled_tints):
            # Modify names for conflict resolution in case two same objects present in one scene
            # And change colors of geoms
            cur_object = ET.parse(os.path.join(prefix_to_folder, '{}.xml'.format(object_name))).getroot()
            old_cur_object_name = cur_object.attrib['name']
            new_cur_object_name = old_cur_object_name + str(object_postfixes[old_cur_object_name])
            for child in cur_object:
                if child.tag in ['joint', 'site']:
                    name, *postfix = child.attrib['name'].split(':')
                    child.attrib['name'] = ':'.join([new_cur_object_name, *postfix])
                if child.tag == 'geom':
                    child.attrib['rgba'] = object_color
                    name, *postfix = child.attrib['name'].split(':')
                    child.attrib['name'] = ':'.join([new_cur_object_name, *postfix])
                    self.created_geom_names.append(child.attrib['name'])
                if child.tag == 'body':
                    for childs_child in child:
                        if childs_child.tag == 'geom':
                            childs_child.attrib['rgba'] = object_tint
                            name, *postfix = childs_child.attrib['name'].split(':')
                            childs_child.attrib['name'] = ':'.join([new_cur_object_name, *postfix])
                            self.created_geom_names.append(childs_child.attrib['name'])

            object_postfixes[old_cur_object_name] += 1
            cur_object.attrib['name'] = new_cur_object_name

            created_object_names.append(new_cur_object_name)
            worldbody.append(cur_object)
        
        # Fetch table size; needed for sampling of objects in scene
        table_pos, table_size = None, None
        for child in worldbody:
            if 'name' in child.attrib.keys() and child.attrib['name'] == 'table0':
                table_geom = child.find('geom')
                table_pos, table_size = child.attrib['pos'], table_geom.attrib['size']
                table_pos = np.array([float(coord) for coord in table_pos.split(' ')])
                table_size = np.array([float(half_length) for half_length in table_size.split(' ')])
                break

        # Change size of target cylinder accordingly to self.distance_threshold
        for child in worldbody:
            if 'name' in child.attrib.keys() and child.attrib['name'] == 'floor0':
                for childs_child in child:
                    if childs_child.attrib['name'] == 'target0':
                        radius, height = childs_child.attrib['size'].split(' ')
                        childs_child.attrib['size'] = str(self.distance_threshold) + ' ' + height

        env_xml_path = os.path.join(self.path_to_xmls_folder, 'env{}.xml'.\
                                    format(self._seed))
        self.env_xml_path = env_xml_path

        with open(env_xml_path, 'w') as f:
            xml_tree.write(f, encoding='unicode')
        return created_object_names, env_xml_path, table_pos, table_size
    
    def _prepare_object_positions(self, table_pos, table_size):
        # All center of masses must be at least on distance 0.07 from end of table
        # 0.07 - maximal distance from center of masses in created objects
        safe_margin = 0.08

        # All objects must be on distance of 0.2 from each other (in order to not overlap)
        safe_distance = 0.15

        # Ascension above table, in order for all objects to be above it
        safe_ascension = 0.04

        done = False
        while not done:
            done = True
            # Get uniform distribution in [-1., 1.]
            points = (self.np_random.uniform(size = (5, 2)) - 0.5) * 2
            # Convert uniform distribution into distribution with table size, inside safe zone
            points[:, 0] = points[:, 0] * (table_size[0] - safe_margin) + table_pos[0]
            points[:, 1] = points[:, 1] * (table_size[1] - safe_margin) + table_pos[1]
            
            # Check if all objects are on safe distance from eachother
            for i in range(points.shape[0]):
                for j in range(points.shape[0]):
                    if i != j:
                        if np.linalg.norm(points[i] - points[j]) < safe_distance:
                            done = False
        points = np.concatenate([points, np.ones([points.shape[0], 1]) * (table_pos[2] + table_size[2] + safe_ascension)],
                                axis = -1)
        
        goal = points[-1:, :2]
        goal = np.concatenate([goal, np.zeros((1, 1))], axis = -1)
        return points, goal

    def _initialize_sim(self):
        created_object_names, xml_path, table_pos, table_size = self._create_multiobject_xml()
        model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(model, nsubsteps = self.n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array']
        }

        initial_qpos = copy.deepcopy(self.gripper_qpos)
        object_positions, goal = self._prepare_object_positions(table_pos = table_pos, table_size = table_size)
        for i, object_name in enumerate(created_object_names):
            initial_qpos['{}:joint'.format(object_name)] = [*object_positions[i], 1., 0., 0., 0.]

        self._env_setup(initial_qpos = initial_qpos)
        return created_object_names, goal
        
    def _get_obs(self):
        # positions
        grip_desc = get_gripper_description(sim = self.sim)
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep

        # Description of gripper head (on which spatulas are connected)
        grip_pos, grip_rot, grip_velp, grip_velr = grip_desc['pos'], grip_desc['rot'], grip_desc['velp'], grip_desc['velr']
        grip_velp, grip_velr = grip_velp * dt, grip_velr * dt

        # Description of gripper grippers (spatulas, rectangular thing with which gripper grasps object)
        robot_qpos, robot_qvel = gym_robotics_utils.robot_get_obs(self.sim)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        objects_description = {'object_pos': [grip_pos], 'object_rot': [grip_rot], 
                               'object_velp': [grip_velp], 'object_velr': [grip_velr], 
                               'object_meta': [np.concatenate([gripper_state, gripper_vel], axis = 0)],
                               'object_ohe': [OBJECT_OHE['grip']]}
        
        if self.obs_type == 'state':
            objects_description['object_rel_pos'] = [np.zeros(objects_description['object_pos'][0].shape)]

        for name in self.created_object_names:
            objects_description['object_pos'].append(self.sim.data.get_site_xpos(name))
            # rotations
            objects_description['object_rot'].append(rotations.mat2euler(self.sim.data.get_site_xmat(name)))
            
            # velocities
            objects_description['object_velp'].append(self.sim.data.get_site_xvelp(name) * dt)
            objects_description['object_velr'].append(self.sim.data.get_site_xvelr(name) * dt)
            
            # relative characteristics
            if self.obs_type == 'state':
                objects_description['object_rel_pos'][-1] = objects_description['object_pos'] - grip_pos
                objects_description['object_velp'][-1] -= grip_velp

            objects_description['object_meta'].append(np.zeros(objects_description['object_meta'][0].shape))
            
            # Remove number from name, thus excluding last char of name
            objects_description['object_ohe'].append(OBJECT_OHE[name[:-1]])
        
        achieved_goal = np.stack(objects_description['object_pos'][1:].copy(), axis = 0)
        assert len(achieved_goal.shape) == 2, 'Something wrong with achieved goal: expected 2 dimensions'
        
        ori_placeholder = {}
        for key in objects_description.keys():
            ori_placeholder[key] = np.concatenate(objects_description[key], axis = 0)
        
        ori_obs = np.concatenate([
            ori_placeholder['object_pos'], ori_placeholder['object_rot'], 
            ori_placeholder['object_velp'], ori_placeholder['object_velr'], 
            ori_placeholder['object_meta']])
        
        if self.obs_type == 'state':
            obs = np.concatenate([ori_placeholder['object_pos'], ori_placeholder['object_velp'], objects_description['object_meta'][0]])
        elif self.obs_type == 'decoupled_state':
            obs = np.concatenate([objects_description[key] for key in objects_description.keys()], axis = -1)
        elif self.obs_type == 'pixels':
            obs = self.render({'width': 64, 'height': 64, 'mode': 'rgb_array', 'segmentation': False})
        else:
            assert False, 'Only state, decoupled_state and pixels modes are supported'

        info_dict = {
            'ori_obs': ori_obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
            'grip_pos': objects_description['object_pos'][0]
        }
        return obs, info_dict
    
    def _is_success(self, achieved_goal, desired_goal):
        desired_goal = desired_goal.copy()[0, :2]
        for object_pos in achieved_goal[:, :2]:
            if np.linalg.norm(object_pos - desired_goal, axis=-1) > self.distance_threshold:
                return False
        return True
    
    def compute_reward(self, achieved_goal, goal, info):
        total_objects_in_zone = 0
        goal = goal.copy()[0, :2]
        for object_pos in achieved_goal[:, :2]:
            total_objects_in_zone += (np.linalg.norm(object_pos - goal) < self.distance_threshold)
        reward = (total_objects_in_zone / len(achieved_goal))
        return reward - 1
    
    def reset(self):
        os.remove(self.env_xml_path)
        self.created_object_names, self.goal = self._initialize_sim()
        obs = self._get_obs()[0]
        return obs
    
    def step(self, action):
        prev_obs, prev_info_dict = self._get_obs()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action, prev_info_dict['grip_pos'])
        self.sim.step()
        cur_obs, cur_info_dict = self._get_obs()

        info = {}
        info['internal_state'], info['ori_obs'] = prev_info_dict['ori_obs'].copy(), prev_info_dict['ori_obs'].copy()
        if self.obs_type == 'pixels':
            info['render'] = cur_info_dict['render']
        info['next_ori_obs'] = cur_info_dict['ori_obs'].copy()
        info['coordinates'], info['next_coordinates'] = np.zeros(3 * (len(self.created_object_names) + 1)),\
            np.zeros(3 * (len(self.created_object_names) + 1))
        for obj in range(len(self.created_object_names) + 1):
            info['coordinates'][obj * 3: (obj + 1) * 3] = prev_info_dict['ori_obs'][obj * 3: (obj + 1) * 3]
            info['next_coordinates'][obj * 3: (obj + 1) * 3] = cur_info_dict['ori_obs'][obj * 3: (obj + 1) * 3]
        info['achieved_goal'], info['desired_goal'] = cur_info_dict['achieved_goal'], cur_info_dict['desired_goal']
        
        reward, done = 0, False
        if not self.unsupervised:
            reward = self.compute_reward(cur_info_dict["achieved_goal"], self.goal, info)
            done = (reward == 0)
        return cur_obs, reward, done, info
    
    def render_step(self, action, resolution = (140, 140)):
        obs, reward, done, info = self.step(action)
        pic = self.render({'width': resolution[0], 'height': resolution[1], 'mode': 'rgb_array'})
        return obs, reward, done, info, pic
    
    def render(self, kwargs):
        from mujoco_py.generated import const as CONST

        mode = kwargs.get('mode', 'rgb_array')
        width, height = kwargs.get('width', DEFAULT_SIZE), kwargs.get('height', DEFAULT_SIZE)
        segmentation = kwargs.get('segmentation', False)
        
        self._render_callback()
        if mode == 'rgb_array':
            if segmentation:
                site_id = self.sim.model.site_name2id('target0')
                old_site_xpos = self.sim.data.get_site_xpos('target0').copy()
                self.sim.model.site_pos[site_id] = np.array([100, 100, 100])
                old_mocap_pos = self.sim.data.get_mocap_pos('robot0:mocap').copy()
                self.sim.data.set_mocap_pos('robot0:mocap', np.array([100, 100, 100]))
                self.sim.forward()
            self._get_viewer(mode).render(width, height, segmentation = segmentation)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth = False, segmentation = segmentation).astype(np.uint8)
            if segmentation:
                pixel_types = data[:, :, 0]
                pixel_idxs = data[:, :, 1]
                geom_idxs = []
                unique_items = {}

                for created_geom_name in self.created_geom_names:
                    created_geom_idx = self.sim.model.geom_name2id(created_geom_name)
                    geom_idxs.append(created_geom_idx)
                
                for geom_idx in np.unique(pixel_idxs[pixel_types == CONST.OBJ_GEOM]):
                    if not (geom_idx in geom_idxs):
                        geom_idxs.append(geom_idx)
                geom_idxs = np.array(geom_idxs)

                for geom_idx in geom_idxs:
                    full_name = self.sim.model.geom_id2name(geom_idx)
                    object_name = full_name.split(':')[0]
                    if not (object_name in unique_items):
                        unique_items[object_name] = []
                    unique_items[object_name].append(geom_idx)
                
                unique_items['background'] = [-1]
                unique_items['background'] = [*unique_items['floor0'], *unique_items['table0'], *unique_items['background']]
                del unique_items['floor0'], unique_items['table0']

                counter = 0
                new_data = data[:, :, 1].copy()
                for idxs in unique_items.values():
                    for idx in idxs:
                        new_data[data[:, :, 1]==idx] = counter
                    counter += 1
                data[:, :, 1] = new_data
                self.sim.model.site_pos[site_id] = old_site_xpos
                self.sim.data.set_mocap_pos('robot0:mocap', old_mocap_pos)
                self.sim.forward()
                return data[::-1, :, :].astype(np.uint8), len(unique_items) 
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()
    
    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        gym_robotics_utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

    @property
    def unwrapped(self):
        """Completely unwrap this env.

        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self
    
    @property
    def n_types(self):
        # Include gripper type as well
        return len(self.object_names) + 1
    
    @property
    def n_obj(self):
        # Include gripper as well
        return self.object_qty + 1

    def _remove_pressure(self, pos_ctrl):
        # Adapted from: https://gist.github.com/machinaut/209c44e8c55245c0d0f0094693053158
        
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            geom_1, geom_2 = self.sim.model.geom_id2name(contact.geom1), self.sim.model.geom_id2name(contact.geom2)
            target_geom_id = None
            if geom_1 in ['robot0:gripper_link', 'robot0:r_gripper_finger_link', 'robot0:l_gripper_finger_link']:
                target_geom_id = contact.geom1
            if geom_2 in ['robot0:gripper_link', 'robot0:r_gripper_finger_link', 'robot0:l_gripper_finger_link']:
                target_geom_id = contact.geom2
            if target_geom_id is not None:
                c_array = np.zeros(6, dtype=np.float64)
                mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, target_geom_id, c_array)

                ref = np.reshape(contact.frame, (3, 3))
                c_force = np.dot(np.linalg.inv(ref), c_array[0:3])[2]
                if c_force < -1e-03:
                    pos_ctrl = np.clip(pos_ctrl, a_min = [-np.inf, -np.inf, 0], a_max = [np.inf, np.inf, np.inf])
                return pos_ctrl
        
        return pos_ctrl

    def _set_action(self, action, obs):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]
        
        pos_ctrl = self._remove_pressure(pos_ctrl)

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        gym_robotics_utils.ctrl_set_action(self.sim, action)
        gym_robotics_utils.mocap_set_action(self.sim, action)

# ============= Override of MujocoTrait methods =============

    def _get_coordinates_trajectories(self, trajectories):
        coordinates_trajectories = {}
        for trajectory in trajectories:
            for element in range(trajectory['env_infos']['coordinates'].shape[1] // 3):
                if element not in coordinates_trajectories:
                    coordinates_trajectories[element] = []
                coordinates_trajectories[element].append(\
                    trajectory['env_infos']['coordinates'][:, element * 3: (element * 3 + 2)])
                coordinates_trajectories[element][-1] = np.concatenate([coordinates_trajectories[element][-1], 
                                                                   trajectory['env_infos']['next_coordinates'][:, element * 3: (element * 3 + 2)]],
                                                                   axis = 0)
        return coordinates_trajectories
        

# ============= No change from fetch_env.PickAndPlaceEnv =============

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('table0')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.5
        self.viewer.cam.azimuth = 180.
        self.viewer.cam.elevation = -55.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        # TODO Why are they substract sites_offset[0] (only by x position)
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}
        os.remove(self.env_xml_path)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer
    
def get_gripper_description(sim):
    return {'pos': sim.data.get_site_xpos('robot0:grip'),
            'rot': rotations.mat2euler(sim.data.get_site_xmat('robot0:grip')),
            'velp': sim.data.get_site_xvelp('robot0:grip'),
            'velr': sim.data.get_site_xvelr('robot0:grip')}