from PIL import Image
import h5py
import numpy as np

#f = h5py.File('/media/denis/data/64x64/HighDiversityOcclusionSH/dataset/data.hdf5', "r")['TrainingSet']
f = h5py.File('/media/denis/data/orig_dataset/RandomObjsEnv-N5C4S4-AgentPosNo-WoAgentTrue-OcclusionTrue-SkewedFalse-Tr1000000-Val10000.hdf5')['TrainingSet']
print(f['obss'].shape)
mean_r, mean_g, mean_b = [], [], []
zeros = []
for i in range(10_000):
    mean_r.append(f['obss'][i, :, :, 0])
    mean_g.append(f['obss'][i, :, :, 1])
    mean_b.append(f['obss'][i, :, :, 2])
    zeros.append(f['obss'][i]==np.array([0, 0, 0]))
print(np.mean(np.array(mean_r)), np.mean(np.array(mean_g)), np.mean(np.array(mean_b)), np.mean(np.array(zeros)))
print(np.std(np.array(mean_r)), np.std(np.array(mean_g)), np.std(np.array(mean_b)), np.std(np.array(zeros)))