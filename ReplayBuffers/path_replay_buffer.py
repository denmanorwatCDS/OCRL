import collections
import numpy as np
import scipy
import copy

def discount_cumsum(x, discount):
    """Discounted cumulative sum.

    See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering  # noqa: E501
    Here, we have y[t] - discount*y[t+1] = x[t]
    or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]

    Args:
        x (np.ndarrary): Input.
        discount (float): Discount factor.

    Returns:
        np.ndarrary: Discounted cumulative sum.

    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1],
                                axis=0)[::-1]

def calculate_GAE(rewards, values, next_values, dones, options, discount, gae_lambda):
    # All arrays are expected to be of size len. 
    # This is because second dimension is always one for rewards, values, next_values and dones. 
    # All inputs are lists.
    
    # Preprocess obs so they can be processed in parallel

    max_length = -1
    for traj in rewards:
        max_length = max(traj.shape[0], max_length)
    
    # Pad all trajectories to the same size (If needed)
    trajectory_info = {'rewards': [], 'values': [], 'next_values': [], 'dones': []}
    for name, data_array in zip(['rewards', 'values', 'next_values'], [rewards, values, next_values]):
        for data_traj in data_array:
            if data_traj.shape[0] < max_length:
                trajectory_info[name].append(np.concatenate([data_traj, np.zeros(max_length - data_traj.shape[0])]))
            else:
                trajectory_info[name].append(data_traj)
        trajectory_info[name] = np.stack(trajectory_info[name], axis = 0)

    for done_traj in dones:
        if done_traj.shape[0] < max_length:
            trajectory_info['dones'].append(np.concatenate([done_traj, np.ones(max_length - done_traj.shape[0])\
                                                            .astype(np.bool_)]))
        else:
            trajectory_info['dones'].append(done_traj)
    trajectory_info['dones'] = np.stack(trajectory_info['dones'], axis = 0)
    
    # Calculate GAE
    last_gae_lambda = np.zeros(trajectory_info['dones'].shape[0])
    advantages = np.zeros(trajectory_info['dones'].shape)
    rewards, dones = trajectory_info['rewards'], trajectory_info['dones']
    values, next_values = trajectory_info['values'], trajectory_info['next_values']
    options = np.array(options)
    for t in reversed(range(max_length)):
        nonterminal = 1 - trajectory_info['dones'][:, t]
        
        # (Q(s, a) - V(s)): single-sample advantage
        delta = (rewards[:, t] + discount * next_values[:, t] * nonterminal) - values[:, t]
        
        # Generalized Advantage Estimation
        advantages[:, t] = last_gae_lambda = delta + discount * gae_lambda * nonterminal * last_gae_lambda
        if t != 0:
            last_gae_lambda = last_gae_lambda * (np.all(np.abs(options[:, t] - options[:, t-1]) < 1e-03, 
                                                        axis=-1))
    returns = advantages + values
    
    # Remove padding (if, for current state, done for previous timestep and current timestep are both true,
    # then this entry is a padding one)
    prev_entry = np.zeros(dones[:, 0].shape).astype(np.bool_)
    mask = []
    for t in range(dones.shape[1]):
        mask.append(~np.bitwise_and(prev_entry, dones[:, t]))
        prev_entry = dones[:, t]
    mask = np.stack(mask, axis = 1)
    
    trajectory_info.update({'returns': [], 'advantages': []})
    for name, data_batch in zip(['returns', 'advantages'], [returns, advantages]):
        for i, data_trajectory in enumerate(data_batch):
            trajectory_info[name].append(data_trajectory[mask[i]])
    
    return trajectory_info['returns'], trajectory_info['advantages']
    

class PathBuffer:
    """A replay buffer that stores and can sample whole paths.

    This buffer only stores valid steps, and doesn't require paths to
    have a maximum length.

    Args:
        capacity_in_transitions (int): Total memory allocated for the buffer.

    """

    def __init__(self, capacity_in_transitions, batch_size, pixel_keys, discount, gae_lambda, on_policy):
        self._capacity = capacity_in_transitions
        self.batch_size = batch_size
        self._transitions_stored = 0
        self._first_idx_of_next_path = 0
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.on_policy = on_policy
        # Each path in the buffer has a tuple of two ranges in
        # self._path_segments. If the path is stored in a single contiguous
        # region of the buffer, the second range will be range(0, 0).
        # The "left" side of the deque contains the oldest path.
        self._path_segments = collections.deque()
        self._buffer = {}
        self.recent_paths = []

        self._pixel_keys = pixel_keys

    def add_path(self, path):
        """Add a path to the buffer.

        Args:
            path (dict): A dict of array of shape (path_len, flat_dim).

        Raises:
            ValueError: If a key is missing from path or path has wrong shape.

        """
        path_len = self._get_path_length(path)
        first_seg, second_seg = self._next_path_segments(path_len)
        # Remove paths which will overlap with this one.
        while (self._path_segments and self._segments_overlap(
                first_seg, self._path_segments[0][0])):
            self._path_segments.popleft()
        while (self._path_segments and self._segments_overlap(
                second_seg, self._path_segments[0][0])):
            self._path_segments.popleft()
        self._path_segments.append((first_seg, second_seg))
        for key, array in path.items():
            buf_arr = self._get_or_allocate_key(key, array)
            buf_arr[first_seg.start: first_seg.stop] = array[:len(first_seg)]
            buf_arr[second_seg.start: second_seg.stop] = array[len(first_seg):]
        if second_seg.stop != 0:
            self._first_idx_of_next_path = second_seg.stop
        else:
            self._first_idx_of_next_path = first_seg.stop
        self._transitions_stored = min(self._capacity,
                                       self._transitions_stored + path_len)

    def fetch_transitions(self, idxs):
        batch = {key: buf_arr[idxs] for key, buf_arr in self._buffer.items()}
        for key in self._pixel_keys:
            batch[key] = ((batch[key] - (255 / 2)) / (255 / 2)).astype(np.float32)
        return batch

    def sample_transitions(self, batch_size = None):
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            dict: A dict of arrays of shape (batch_size, flat_dim).

        """
        if batch_size is None:
            batch_size = self.batch_size
        idxs = np.random.choice(self._transitions_stored, batch_size)
        return self.fetch_transitions(idxs)
    
    def next_batch(self):
        idxs = np.arange(self._transitions_stored)
        np.random.shuffle(idxs)
        for i in range(0, self._transitions_stored // self.batch_size, 1):
            yield self.fetch_transitions(idxs[i * self.batch_size: (i + 1) * self.batch_size])

    def _next_path_segments(self, n_indices):
        """Compute where the next path should be stored.

        Args:
            n_indices (int): Path length.

        Returns:
            tuple: Lists of indices where path should be stored.

        Raises:
            ValueError: If path length is greater than the size of buffer.

        """
        if n_indices > self._capacity:
            raise ValueError('Path is too long to store in buffer.')
        start = self._first_idx_of_next_path
        end = start + n_indices
        if end > self._capacity:
            second_end = end - self._capacity
            return (range(start, self._capacity), range(0, second_end))
        else:
            return (range(start, end), range(0, 0))

    def _get_or_allocate_key(self, key, array):
        """Get or allocate key in the buffer.

        Args:
            key (str): Key in buffer.
            array (numpy.ndarray): Array corresponding to key.

        Returns:
            numpy.ndarray: A NumPy array corresponding to key in the buffer.

        """
        buf_arr = self._buffer.get(key, None)
        if buf_arr is None:
            buf_arr = np.zeros((self._capacity,) + array.shape[1:], array.dtype)
            self._buffer[key] = buf_arr
        return buf_arr

    def clear(self):
        """Clear buffer."""
        self._transitions_stored = 0
        self._first_idx_of_next_path = 0
        self._path_segments.clear()
        self._buffer.clear()
        self.delete_recent_paths()

    def preprocess_data(self, paths):
        data = collections.defaultdict(list)
        for path in paths:
            if 'obs' in self._pixel_keys:
                assert np.bitwise_and(np.all(path['observations'] > -1.01), np.all(path['observations'] < 1.01)),\
                    'Expected normalized images'
                path['observations'] = np.rint((path['observations'] * 255 / 2) + 255 / 2).astype(np.uint8)

            if 'next_obs' in self._pixel_keys:
                assert np.bitwise_and(np.all(path['next_observations'] > -1.01), np.all(path['next_observations'] < 1.01)),\
                    'Expected normalized images'
                path['next_observations'] = np.rint((path['next_observations'] * 255 / 2) + 255 / 2).astype(np.uint8)

            data['obs'].append(path['observations'])
            data['next_obs'].append(path['next_observations'])
            data['actions'].append(path['actions'])
            data['rewards'].append(path['rewards'])
            data['dones'].append(path['dones'])
            if 'ori_obs' in path['env_infos'].keys():
                data['ori_obs'].append(path['env_infos']['ori_obs'])
            if 'next_ori_obs' in path['env_infos'].keys():
                data['next_ori_obs'].append(path['env_infos']['next_ori_obs'])
            if 'pre_tanh_actions' in path['agent_infos']:
                data['pre_tanh_actions'].append(path['agent_infos']['pre_tanh_actions'])
            if 'log_probs' in path['agent_infos']:
                data['log_probs'].append(path['agent_infos']['log_probs'])
            if 'options' in path['agent_infos']:
                data['options'].append(path['agent_infos']['options'])
                data['next_options'].append(np.concatenate([path['agent_infos']['options'][1:], 
                                                            path['agent_infos']['options'][-1:]], axis=0))
            if 'obj_idxs' in path['agent_infos']:
                data['obj_idxs'].append(path['agent_infos']['obj_idxs'])
                data['next_obj_idxs'].append(np.concatenate([path['agent_infos']['obj_idxs'][1:], 
                                                             path['agent_infos']['obj_idxs'][-1:]], axis=0))
        if self.on_policy and 'values' in paths[0]['agent_infos'].keys() and 'next_values' in paths[0]['agent_infos'].keys():
            rewards = [paths[i]['rewards'] for i in range(len(paths))]
            values = [paths[i]['agent_infos']['values'] for i in range(len(paths))]
            next_values = [paths[i]['agent_infos']['next_values'] for i in range(len(paths))]
            dones = [paths[i]['dones'] for i in range(len(paths))]
            options = [paths[i]['agent_infos']['options'] for i in range(len(paths))]
            rets, advs = calculate_GAE(rewards, values, next_values, dones, options, self.discount, self.gae_lambda)
            for ret, adv, vals, next_vals, in zip(rets, advs, values, next_values):
                data['returns'].append(ret)
                data['advantages'].append(adv)
                data['values'].append(vals)
                data['next_values'].append(next_vals)
        return data
    
    def store_recent_paths(self, paths):
        for path in paths:
            self.recent_paths.append(path)

    def get_recent_paths(self):
        return copy.deepcopy(self.recent_paths)
        
    def delete_recent_paths(self):
        del self.recent_paths
        self.recent_paths = []

    def update_replay_buffer(self, data):
        if self.on_policy:
            self.store_recent_paths(copy.deepcopy(data))

        data = dict(self.preprocess_data(data))
        for i in range(len(data['actions'])):
            path = {}
            for key in data.keys():
                path[key] = data[key][i]
            self.add_path(path)

    def fetch_trajectories(self):
        qty_of_path_segments = len(self._path_segments)
        path_batch = []
        for i in range(qty_of_path_segments):
            path_segments = list(self._path_segments)[i]
            first_seg, second_seg = path_segments[0], path_segments[1]
            idxs = list(first_seg) + list(second_seg)
            path = self.fetch_transitions(idxs)
            path_batch.append(path)
        return path_batch

    @staticmethod
    def _get_path_length(path):
        """Get path length.

        Args:
            path (dict): Path.

        Returns:
            length: Path length.

        Raises:
            ValueError: If path is empty or has inconsistent lengths.

        """
        length_key = None
        length = None
        for key, value in path.items():
            if length is None:
                length = len(value)
                length_key = key
            elif len(value) != length:
                raise ValueError('path has inconsistent lengths between '
                                 '{!r} and {!r}.'.format(length_key, key))
        if not length:
            raise ValueError('Nothing in path')
        return length

    @staticmethod
    def _segments_overlap(seg_a, seg_b):
        """Compute if two segments overlap.

        Args:
            seg_a (range): List of indices of the first segment.
            seg_b (range): List of indices of the second segment.

        Returns:
            bool: True iff the input ranges overlap at at least one index.

        """
        # Empty segments never overlap.
        if not seg_a or not seg_b:
            return False
        first = seg_a
        second = seg_b
        if seg_b.start < seg_a.start:
            first, second = seg_b, seg_a
        assert first.start <= second.start
        return first.stop > second.start

    @property
    def n_transitions_stored(self):
        """Return the size of the replay buffer.

        Returns:
            int: Size of the current replay buffer.

        """
        return int(self._transitions_stored)