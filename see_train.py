from PIL import Image
import h5py
import numpy as np

f = h5py.File('/media/denis/data/orig_dataset/sh.hdf5', "r")['TrainingSet']
print(f['obss'].shape)
img_path = '/media/denis/data/orig_dataset/sh_inspection'
mean_r, mean_g, mean_b = [], [], []
zeros = []
for i in range(4):
    Image.fromarray(f["obss"][i]).save(f'{img_path}/Example_{i}.png')
    mean_r.append(f['obss'][i, :, :, 0])
    mean_g.append(f['obss'][i, :, :, 1])
    mean_b.append(f['obss'][i, :, :, 2])
    zeros.append(f['obss'][i]==np.array([0, 0, 0]))
print(np.mean(np.array(mean_r)), np.mean(np.array(mean_g)), np.mean(np.array(mean_b)), np.mean(np.array(zeros)))
print(np.std(np.array(mean_r)), np.std(np.array(mean_g)), np.std(np.array(mean_b)), np.std(np.array(zeros)))