from collections import OrderedDict

import akro
import numpy as np
from gym import spaces


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = akro.Box(low=low, high=high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class MujocoTrait:
    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = akro.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def render(self,
               mode='human',
               width=100,
               height=100,
               camera_id=None,
               camera_name=None):
        if hasattr(self, 'render_hw') and self.render_hw is not None:
            width = self.render_hw
            height = self.render_hw
        return super().render(mode, width, height, camera_id, camera_name)

    def calc_eval_metrics(self, trajectories, coord_dims=None):
        eval_metrics = {}

        if coord_dims is not None:
            coords = []
            for trajectory_coord, trajectory_next_coord in zip(trajectories['coordinates'], trajectories['next_coordinates']):
                traj1 = trajectory_coord[:, coord_dims]
                traj2 = trajectory_next_coord[-1:, coord_dims]
                coords.append(traj1)
                coords.append(traj2)
            coords = np.concatenate(coords, axis=0)
            uniq_coords = np.unique(np.floor(coords), axis=0)
            eval_metrics.update({
                'MjNumTrajs': len(trajectories),
                'MjAvgTrajLen': len(coords) / len(trajectories) - 1,
                'MjNumCoords': len(coords),
                'MjNumUniqueCoords': len(uniq_coords),
            })

        return eval_metrics
