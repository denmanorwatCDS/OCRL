import h5py
import torch
from torch.utils.data import Dataset

class H5Dataset(Dataset):
    def __init__(self, datafile, uint_to_float, use_future, future_steps, 
                 augment = None, is_train = True, debug = False):
        f = h5py.File(datafile, "r")
        self.is_train = is_train
        self.extract_masks = False
        self.augment = augment
        self.uint_to_float = uint_to_float
        self.use_future, self.future_steps = use_future, future_steps
        if is_train:
            self._data = {}
            if debug:
                self._data['dones'] = f['TrainingSet']['dones'][:10_000]
                self._data['obss'] = f['TrainingSet']['obss'][:10_000]
            else:
                self._data = f['TrainingSet']
            if self.use_future:
                self.future_step_proba = self.precalculate_data()
        else:
            self._data = f['ValidationSet']
            if 'masks' in self._data.keys():
                self.extract_masks = True
        self._num_samples = self._data["obss"].shape[0]

    def precalculate_data(self):
        future_idx_probs = [[0. for i in range(self.future_steps)] for j in range(len(self._data['dones']))]
        for i in range(len(self._data['dones'])):
            future_steps = min(self.future_steps, len(self._data['dones']) - i)
            for j in range(future_steps):
                future_idx_probs[i][j] = 1.
                if self._data['dones'][i + j]:
                    continue
        return torch.tensor(future_idx_probs)

    def __getitem__(self, index):
        sample = self._data['obss'][index]
        if self.is_train:
            concat_sample = torch.unsqueeze(torch.from_numpy(sample), dim = 0)
            if self.use_future:
                future_sample_index = torch.multinomial(self.future_step_proba[index], num_samples = 1).item()
                future_sample = torch.unsqueeze(torch.from_numpy(self._data['obss'][index + future_sample_index]), dim = 0)
                concat_sample = torch.cat((concat_sample, future_sample), dim = 0)
            if self.augment:
                concat_sample = self.augment(concat_sample)

            if self.use_future:
                sample, future_sample = concat_sample[0], self.uint_to_float(concat_sample[1]).permute(2, 0, 1)
            else:
                sample, future_sample = concat_sample[0], torch.tensor(float('nan'))
            sample = self.uint_to_float(sample).permute(2, 0, 1)
            return sample, future_sample
        else:
            outp = {'obss': self.uint_to_float(torch.from_numpy(sample)).permute(2, 0, 1)}
            if self.extract_masks:
                outp['masks'] = self._data['masks'][index]
            return outp

    def __len__(self):
        return self._num_samples