import pickle
import h5py
import numpy as np
import os
import sys

if len(sys.argv) < 2:
    sys.exit("Usage: python format_hdf5.py [FILENAME]")
dir_ = os.path.dirname(os.path.abspath(__file__)) + '/../../logs'
#dir_ = '/nfs/diskstation/ryanhoque/vismpc_data_logs'
files = [f for f in os.listdir(dir_) if '.pkl' in f]
combined_pkl = []
for file_ in files:
    p_ = pickle.load(open(dir_ + '/' + file_, 'rb'))
    p1 = [ep for ep in p_ if len(ep['act']) == 10]
    combined_pkl.extend(p1)
print("Total episodes:", len(combined_pkl))
with h5py.File(sys.argv[1],'w') as f:
    d_images = f.create_dataset('images', (len(combined_pkl), 11, 56, 56, 3))
    d_actions = f.create_dataset('actions', (len(combined_pkl), 10, 4))
    for i in range(len(combined_pkl)):
        d_images[i:i+1] = np.array(combined_pkl[i]['obs'])
        d_actions[i:i+1] = np.array(combined_pkl[i]['act'])
        if i % 100 == 0:
            f.flush()
    f.flush()
    f.close()
