import sys
import pickle
import numpy as np
import cv2

files = sys.argv[1:]
all_data = []
for f in files:
    p = pickle.load(open(f, 'rb'))
    if 'predictions' in f: # imagined
        for ep in p:
            all_data.extend(ep['pred'])
    else:
        for ep in p:
            all_data.extend(ep['obs'])
all_data = np.array(all_data).astype(np.uint8)
data2 = np.zeros((all_data.shape[0], 128, 128, all_data.shape[3]))
for i in range(all_data.shape[0]):
    data2[i] = cv2.resize(all_data[i][7:-7,7:-7,:], (128,128)) # trim black border and resize for autoencoder CNN
data2 = data2.astype(np.uint8)
print(data2.shape)
pickle.dump(data2, open('vae_data.pkl', 'wb'))
