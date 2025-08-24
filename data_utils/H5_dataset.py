import h5py
import torch
from torch.utils.data import Dataset

class H5Dataset(Dataset):
    def __init__(self, datafile, uint_to_float, augment = None, is_train = True):
        f = h5py.File(datafile, "r")
        self.is_train = is_train
        self.extract_masks = False
        if is_train:
            self._data = f['TrainingSet']
        else:
            self._data = f['ValidationSet']
            if 'masks' in self._data.keys():
                self.extract_masks = True
        self._num_samples = self._data["obss"].shape[0]
        self.augment = augment
        self.uint_to_float = uint_to_float

    def __getitem__(self, index):
        sample = self._data['obss'][index]
        if self.is_train:
            if self.augment:
                sample = self.augment(sample)
            return torch.from_numpy(self.uint_to_float(sample)).permute(2,0,1)
        else:
            outp = {'obss': torch.from_numpy(self.uint_to_float(sample)).permute(2,0,1)}
            if self.extract_masks:
                outp['masks'] = self._data['masks'][index]
            return outp

    def __len__(self):
        return self._num_samples