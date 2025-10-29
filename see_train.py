from PIL import Image
import h5py

f = h5py.File('/media/denis/data/orig_dataset/RandomObjsEnv-N5C4S4-AgentPosNo-WoAgentTrue-OcclusionTrue-SkewedFalse-Tr1000000-Val10000.hdf5', "r")['TrainingSet']
for i in range(1000):
    Image.fromarray(f['obss'][i]).save(f'/home/denis/Work/OCRL_refactored/inspection/{i}.png')