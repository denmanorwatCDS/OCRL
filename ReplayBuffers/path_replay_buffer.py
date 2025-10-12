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
    
    # Calculate GAE
    last_gae_lambda = np.zeros(dones.shape[0])
    advantages = np.zeros(dones.shape)
    for t in reversed(range(dones.shape[1])):
        nonterminal = 1 - dones[:, t]
        
        # (Q(s, a) - V(s)): single-sample advantage
        delta = (rewards[:, t] + discount * next_values[:, t] * nonterminal) - values[:, t]
        
        # Generalized Advantage Estimation
        advantages[:, t] = last_gae_lambda = delta + discount * gae_lambda * nonterminal * last_gae_lambda
        if t != 0:
            last_gae_lambda = last_gae_lambda * (np.all(np.abs(options[:, t] - options[:, t-1]) < 1e-03, 
                                                        axis=-1))
    returns = advantages + values
    
    return returns, advantages
    

class PathBuffer:
    """A replay buffer that stores and can sample whole paths.

    This buffer only stores valid steps, and doesn't require paths to
    have a maximum length.

    Args:
        capacity_in_transitions (int): Total memory allocated for the buffer.

    """

    def __init__(self, capacity_in_transitions, batch_size, pixel_keys, discount, gae_lambda, seed):
        self._capacity = capacity_in_transitions
        self.batch_size = batch_size
        self._transitions_stored = 0
        self._first_idx_of_next_path = 0
        self.discount = discount
        self.gae_lambda = gae_lambda
        # Each path in the buffer has a tuple of two ranges in
        # self._path_segments. If the path is stored in a single contiguous
        # region of the buffer, the second range will be range(0, 0).
        # The "left" side of the deque contains the oldest path.
        self._path_segments = collections.deque()
        self._buffer = {}
        self.rng = np.random.default_rng(seed)
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

    def preprocess_data(self, paths):
        data = copy.deepcopy(paths)
        if 'obs' in self._pixel_keys:
            assert np.bitwise_and(np.all(data['observations'] > -1.01), np.all(data['observations'] < 1.01)),\
                'Expected normalized images'
            data['observations'] = np.rint((data['observations'] * 255 / 2) + 255 / 2).astype(np.uint8)

        if 'next_obs' in self._pixel_keys:
            assert np.bitwise_and(np.all(data['next_observations'] > -1.01), np.all(data['next_observations'] < 1.01)),\
                'Expected normalized images'
            data['next_observations'] = np.rint((data['next_observations'] * 255 / 2) + 255 / 2).astype(np.uint8)

        return data

    def update_replay_buffer(self, data):
        data = dict(self.preprocess_data(data))
        for i in range(len(data['actions'])):
            path = {}
            for key in data.keys():
                path[key] = data[key][i]
            self.add_path(path)

    def update_rollout_buffer(self, data):
        rets, advs = calculate_GAE(data['rewards'], data['values'], data['next_values'], 
                                   data['dones'], data['options'], self.discount, self.gae_lambda)
        data['returns'], data['advantages'] = rets, advs
        self.rollout_buffer = data

    def get_rollout_iterator(self, rollout_batch_size):
        trajectory_qty, trajectories_length = self.rollout_buffer['dones'].shape[:2]
        global_to_local = np.stack([np.concatenate([np.zeros(trajectories_length, dtype=np.int32) + i for i in range(trajectory_qty)], axis = 0), 
                                    np.concatenate([np.arange(0, trajectories_length, dtype=np.int32) for i in range(trajectory_qty)], axis = 0)],
                                    axis = 1)
        shuffled_idxs = np.arange(trajectory_qty * trajectories_length)
        self.rng.shuffle(shuffled_idxs)
        for i in range(0, shuffled_idxs.shape[0] // rollout_batch_size * rollout_batch_size, rollout_batch_size):
            batch = {}
            for key in self.rollout_buffer.keys():
                traj_idxs, step_idxs = np.split(global_to_local[shuffled_idxs[i: i + rollout_batch_size]], 
                                                indices_or_sections = 2, axis = 1)
                batch[key] = np.squeeze(self.rollout_buffer[key][traj_idxs, step_idxs], axis=1)
            yield batch

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