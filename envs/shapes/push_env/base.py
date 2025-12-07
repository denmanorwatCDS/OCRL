import os
import numpy as np
from copy import deepcopy
from PIL import Image
from gym import spaces
from pathlib import Path
from matplotlib import colors
from spriteworld import renderers as spriteworld_renderers
from spriteworld.sprite import Sprite

from envs.shapes.push_env.utils import l1_norm

COLORS = ["blue", "green", "yellow", "cyan", "pink", "brown"]
SHAPES = ["square", "triangle", "star_4", "pentagon", "hexagon", "octagon", "spoke_6"]
SCALES = [0.15]

class BaseEnv:
    metadata = {"render.modes": ["rgb_array", "state", "image", "mask"]}

    def __init__(self, seed, arena_size = 1.,
                 render_mode = 'rgb_array', render_info = False, obs_size = 64, obs_channels = 3,
                 num_objects_range = [4, 4], moving_step_size = 0.05, 
                 wo_agent = False, max_steps = 100, agent_pos = [0.5, 0.5],
                 use_bg = False):
        agent_params = ['red', 'circle', 0.15 / arena_size]
        self._state_size = 3
        self.render_mode = render_mode
        self._obs_size = obs_size
        self._obs_channels = obs_channels
        self._num_objs_range = num_objects_range
        assert num_objects_range[0] == num_objects_range[1], '''It is not clear yet if all code will work correctly
        if quantities of objects are able to change...'''
        self._renderer = spriteworld_renderers.PILRenderer(
            image_size=(obs_size, obs_size),
            anti_aliasing=10,
        )
        self._moving_step_size = moving_step_size / arena_size
        self._wo_agent = wo_agent
        self._max_steps = max_steps
        self._agent_pos = agent_pos
        self.arena_size = arena_size
        self._COLORS = COLORS
        self._SHAPES = SHAPES
        self._SCALES = [shape_scale / arena_size for shape_scale in SCALES]
        self._AGENT = agent_params
        self.render_info = render_info
        self.seed_ = seed
        self.np_rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(low = -np.array([1, 1]),
                                       high = np.array([1, 1]))
        if self.render_mode in ['state', 'simple_state']:
            self.state_fetcher = self.prepare_states(self._SHAPES + ([agent_params[1]] if agent_params[1] not in self._SHAPES else []))
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(self._num_objs_range[1] + 1, self._state_size),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(
                    self._obs_size,
                    self._obs_size,
                    self._obs_channels,
                ),
                dtype=np.uint8,
            )

        self._objs = None
        self.step_count = 0

    def prepare_states(self, preset_shapes):
        # 3 - is color vector size, 
        # len(shapes) - Quantity of shapes
        # 1 - is scale vector size,
        # 2 - is position vector size
        if self.render_mode == 'state':
            self._state_size = 3 + len(preset_shapes) + 1 + 2
            def state_fetcher(color, shape, scale, position):
                color_vec = np.array([c for c in colors.to_rgb(color)])
                shape_vec = np.zeros(len(preset_shapes))
                shape_vec[preset_shapes.index(shape)] = 1
                scale_vec = np.array([scale])
                return np.concatenate([color_vec, shape_vec, scale_vec, position])
        elif self.render_mode == 'simple_state':
            self._state_size = 3
            def state_fetcher(color, shape, scale, position):
                is_agent = np.array([0])
                if color == 'red' and shape == 'circle':
                    is_agent = np.array([1])
                return np.concatenate([is_agent, position])
        else:
            raise NotImplementedError
        return state_fetcher

    def _get_position(self, pos_min, pos_max, radius, eps):
        if pos_min == pos_max:
            return pos_min
        _min = pos_min + radius + eps
        _max = pos_max - radius - eps
        return self.np_rng.uniform(_min, _max)

    def _fill_positions(
        self,
        objs,
        agent_eps=0.08,
        objs_eps=0.08,
        wall_eps=0.08
    ):
        if self._agent_pos is not None:
            objs[-1, 3] = float(self._agent_pos[0])
            objs[-1, 4] = float(self._agent_pos[1])
        for i, _obj in enumerate(objs):
            # agent
            if (i == len(objs) - 1) and (self._agent_pos is not None or self._wo_agent):
                continue
            x_min, x_max, y_min, y_max = self._obj_poses[i]
            radius = _obj[2] / 2
            found = False
            while not found:
                x = self._get_position(x_min, x_max, radius, wall_eps)
                y = self._get_position(y_min, y_max, radius, wall_eps)
                found = True
                for j in range(objs.shape[0]):
                    threshold = radius + objs[j, 2] / 2 + objs_eps
                    if l1_norm(objs[j, 3:5] - [x, y]) < threshold:
                        found = False
                        break
                if self._agent_pos is not None:
                    threshold = radius + objs[-1, 2] / 2 + agent_eps
                    if l1_norm(objs[-1, 3:5] - [x, y]) < threshold:
                        found = False
            objs[i, 3] = x
            objs[i, 4] = y
        return objs

    def _set_objs(self):
        self._num_objects = self.np_rng.choice(
            list(range(self._num_objs_range[0], self._num_objs_range[1] + 1))
        )
        offset = (1 - 1 / self.arena_size) / 2
        self._obj_poses = [[offset, 1.0 - offset, offset, 1.0 - offset]] * (self._num_objects + 1)
        # color, shape, scale, x, y
        objs = np.zeros((self._num_objects + 1, 5), dtype=object)
        objs[-1, :3] = self._AGENT

        return objs

    def _get_masks(self, objs):
        masks = []
        bg = self._renderer.render([])
        for _obj in objs[:-1] if self._wo_agent else objs:
            rgb = [int(c * 255) for c in colors.to_rgb(_obj[0])]
            sprites= [
                Sprite(
                    _obj[3],
                    _obj[4],
                    _obj[1],
                    c0=rgb[0],
                    c1=rgb[1],
                    c2=rgb[2],
                    scale=_obj[2],
                )
            ]
            obs = self._renderer.render(sprites)
            obs = np.sum(np.abs(obs - bg), axis=-1)
            _mask = np.zeros((self._obs_size, self._obs_size, 1), dtype=int)
            _mask[obs!=0] = 1
            masks.append(_mask)
        fg_mask = np.sum(np.array(masks), axis=0)
        bg_mask = np.zeros((self._obs_size, self._obs_size, 1), dtype=int)
        bg_mask[fg_mask==0] = 1
        masks.append(bg_mask)
        return np.array(masks)

    def _draw_objs(self, objs, mode="rgb_array"):
        sprites = []
        for _obj in objs[:-1] if self._wo_agent else objs:
            if _obj[0] == -1:
                continue
            rgb = [int(c * 255) for c in colors.to_rgb(_obj[0])]
            sprites.append(
                Sprite(
                    _obj[3],
                    _obj[4],
                    _obj[1],
                    c0=rgb[0],
                    c1=rgb[1],
                    c2=rgb[2],
                    scale=_obj[2],
                )
            )
        obs = self._renderer.render(sprites)
        if mode == "rgb_array":
            return obs

    def reset(self):
        self._objs = self._set_objs()
        self.step_count = 0
        return self.render()

    def step(self, act):
        """
        act: {0,1,2,3} <- up, left, down, right
        """
        # move
        assert np.all(act <= 1.) and np.all(act >= -1), 'Out-of-bounds action is supplied'
        prev_pos = deepcopy(self._objs[:, 3: 5])
        self._objs[-1, 3] += act[0] * self._moving_step_size
        self._objs[-1, 4] += act[1] * self._moving_step_size

        self._objs[-1, 3] = np.clip(
            self._objs[-1, 3], self._AGENT[2] / 2, 1 - self._AGENT[2] / 2
        )
        self._objs[-1, 4] = np.clip(
            self._objs[-1, 4], self._AGENT[2] / 2, 1 - self._AGENT[2] / 2
        )
        next_pos = deepcopy(self._objs[:, 3: 5])
        self.step_count += 1
        if self.step_count >= self._max_steps:
            truncated = True
        obs = self.render()
        return obs, 0, self._prepare_info(self, obs, truncated, prev_coordinares = prev_pos,
                                                    next_coordinates = next_pos), False
    
    def _prepare_info(self, obs, truncated, prev_coordinares, next_coordinates):
        info = dict(
            truncated = truncated,
            coordinates = prev_coordinares.astype(np.float32),
            next_coordinates = next_coordinates.astype(np.float32)
        )
        if self.render_info:
            if self.render_mode in ['state', 'simple_state']:
                info['render'] = self.render(mode = 'rgb_array')
            elif self.render_mode == 'rgb_array':
                info['render'] = obs
            info['render'] = np.moveaxis(info['render'], source = 2, destination = 0)
        return info

    def render(self, mode=None, fill_empty=True):
        if mode is None:
            mode = self.render_mode
        if mode in ['state', 'simple_state']:
            gt_states = np.zeros((self._objs.shape[0], self._state_size))
            for i in range(gt_states.shape[0]):
                if self._objs[i, 0] == -1:
                    gt_states[i] = np.zeros(self._state_size) - 1
                    continue
                gt_states[i] = self.state_fetcher(color = self._objs[i, 0], shape = self._objs[i, 1],
                                                  scale = self._objs[i, 2], position = self._objs[i, 3:5])
            # Indexing Agent
            gt_states = np.array(gt_states, dtype=np.float32)
            if fill_empty:
                zero_padding_size = self._num_objs_range[1] + 1 - gt_states.shape[0]
                if zero_padding_size > 0:
                    zero_padding = np.zeros((zero_padding_size, self._state_size))
                    gt_states = np.concatenate([gt_states, zero_padding], axis=0)
            return gt_states
        elif mode == "mask":
            masks = self._get_masks(self._objs) # objs, agent and bg
            if fill_empty:
                zero_padding_size = self._num_objs_range[1] + 2 - masks.shape[0]
                if zero_padding_size > 0:
                    zero_padding = np.zeros((zero_padding_size, self._obs_size, self._obs_size, 1))
                    masks = np.concatenate([masks, zero_padding], axis = 0)
            return masks
        else:
            return self._draw_objs(self._objs, mode)

    def close(self):
        self._objs = None
        self.step_count = 0

    @property
    def n_obj(self):
        # Account for agent
        return self._num_objs_range[1] + 1

    def env_discretizer(self):
        return lambda x: np.floor(x / self._moving_step_size)
    