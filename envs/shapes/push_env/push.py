import copy
import numpy as np
from gym import spaces
from matplotlib import colors
from spriteworld import renderers as spriteworld_renderers
from spriteworld.sprite import Sprite
from copy import deepcopy

from .base import BaseEnv
from .utils import norm, l1_norm

class PushEnv(BaseEnv):
    def __init__(self, seed, arena_size = 1.,
                 render_mode = 'rgb_array', render_info = False, obs_size = 64, obs_channels = 3,
                 num_objects_range = [4, 4], moving_step_size = 0.05, 
                 wo_agent = False, max_steps = 100, agent_pos = [0.5, 0.5],
                 use_bg = False, distance_to_agent = 0.08, distance_to_objs = 0.08, distance_to_wall = 0.08):
        super(PushEnv, self).__init__(seed = seed, arena_size = arena_size, render_mode = render_mode, 
                                      render_info = render_info, obs_size = obs_size, obs_channels = obs_channels, 
                                      num_objects_range = num_objects_range, moving_step_size = moving_step_size, 
                                      wo_agent = wo_agent, max_steps = max_steps, agent_pos = agent_pos, use_bg = use_bg)
        _state_size = 5
        self._distance_to_agent, self._distance_to_objs, self._distance_to_wall = \
            distance_to_agent / arena_size, distance_to_objs / arena_size, distance_to_wall / arena_size

    def _set_objs(self):
        objs = super()._set_objs()
        colors = self.np_rng.choice(self._COLORS, size = self._num_objects, replace = False)
        shapes = self.np_rng.choice(self._SHAPES, size = self._num_objects, replace = False)
        scales = self.np_rng.choice(self._SCALES, size = self._num_objects, replace = True)
        for n_idx in range(self._num_objects):
            objs[n_idx][0] = colors[n_idx]
            objs[n_idx][1] = shapes[n_idx]
            objs[n_idx][2] = scales[n_idx]
        objs = self._fill_positions(
            objs,
            agent_eps=self._distance_to_agent,
            objs_eps=self._distance_to_objs,
            wall_eps=self._distance_to_wall,
        )
        return objs

    def _check_can_move(self, obj_idx, eps=1e-6):
        for i in range(self._num_objects):
            if i == obj_idx:
                continue
            if (
                l1_norm(self._objs[i, 3: 5] - self._objs[obj_idx, 3: 5]) + eps
                < self._objs[i, 2] / 2 + self._objs[obj_idx, 2] / 2
            ):
                return False
        return True

    def _move_objs(self, delta, eps=1e-6):
        self._objs[-1, 3] += delta[0]
        self._objs[-1, 4] += delta[1]
        moves = [delta]
        agent_size = self._AGENT[2]
        for i in range(self._num_objects):
            obj_size = self._objs[i, 2]
            collision_distance = obj_size / 2 + agent_size / 2
            obj_movement = np.zeros_like(delta)
            
            if (
                l1_norm(self._objs[i, 3:5] - self._objs[-1, 3:5]) + eps
                < collision_distance
            ):
                # when object is near the wall
                allowed_movements = self._objs[i, 3: 5] - self._objs[-1, 3: 5]
                is_x_move_aligned = (allowed_movements[0] * delta[0]) > 0
                are_objs_collided_by_x = np.abs(self._objs[i, 3] - self._objs[-1, 3]) < collision_distance
                is_y_move_aligned = (allowed_movements[1] * delta[1]) > 0
                are_objs_collided_by_y = np.abs(self._objs[i, 4] - self._objs[-1, 4]) < collision_distance
                if (self._objs[i, 3] == obj_size / 2) or (self._objs[i, 3] == 1 - obj_size / 2) or \
                   (self._objs[i, 4] == obj_size / 2) or (self._objs[i, 4] == 1 - obj_size / 2):
                    # TODO Act according to discrete push logic
                    moves.append(np.array([0, 0]))
                    break
                if is_x_move_aligned and are_objs_collided_by_x:
                    obj_movement[0] = delta[0]
                if is_y_move_aligned and are_objs_collided_by_y:
                    obj_movement[1] = delta[1]
                before_pos = copy.deepcopy(self._objs[i, 3: 5])
                self._objs[i, 3: 5] += obj_movement
                if not self._check_can_move(i):
                    self._objs[i, 3: 5] -= obj_movement
                    moves.append(np.array([0, 0]))
                    break
                self._objs[i, 3: 5] = np.clip(self._objs[i, 3: 5], obj_size / 2, 1 - obj_size / 2)
                moves.append(self._objs[i, 3: 5] - before_pos)
        moves = np.array(moves)
        smallest_move_idxs = np.argmin(np.abs(moves), axis = 0)
        smallest_move = np.array([moves[smallest_move_idxs[0], 0], moves[smallest_move_idxs[1], 1]])
        self._objs[-1, 3: 5] = self._objs[-1, 3: 5] - delta + smallest_move

    def step(self, act):
        """
        act: {0,1,2,3} <- up, left, down, right
        """
        truncated = False
        # move
        assert np.all(act <= 1.) and np.all(act >= -1), 'Out-of-bounds action is supplied'
        prev_coords = deepcopy(self._objs[:, 3: 5])
        self._move_objs(act * self._moving_step_size)
        self._objs[-1, 3: 5] = np.clip(
            self._objs[-1, 3: 5], self._AGENT[2] / 2, 1 - self._AGENT[2] / 2
        )
        next_coords = deepcopy(self._objs[:, 3: 5])
        self.step_count += 1
        if self.step_count >= self._max_steps:
            truncated = True
        # dense reward type
        obs = self.render()
        return (
            obs,
            0.,
            False,
            self._prepare_info(obs, truncated, prev_coordinares = prev_coords, next_coordinates = next_coords)
        )
    
    def _prepare_info(self, obs, truncated, prev_coordinares, next_coordinates):
        info = super()._prepare_info(obs, truncated, prev_coordinares, next_coordinates)
        info['objects'] = deepcopy(self._objs[:-1])
        info['agent'] = deepcopy(self._objs[-1])
        return info
