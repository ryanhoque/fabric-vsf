"""
Run inference with a trained SV2P model.
"""
from vismpc.SV2P import SV2P
from dotmap import DotMap
import pickle
import numpy as np
import sys
import argparse

pp = argparse.ArgumentParser()
pp.add_argument("--data_dir", type=str, default="sv2p_data_cloth")
pp.add_argument("--model_dir", type=str, default="sv2p_model_cloth")
pp.add_argument("--horizon", type=int, default=5)
pp.add_argument("--input_img", type=str, help="filepath of input image as .pkl file")
pp.add_argument("--batch", action="store_true", default=False) # input_img is pkl of images and actions
args = pp.parse_args()
params = DotMap()
params.name = 'cloth'
params.model_dir = args.model_dir
params.data_dir = args.data_dir
params.popsize = 1
params.nparts = 1
params.plan_hor = args.horizon
params.adim = 4
params.stochastic_model = True
sys.argv = sys.argv[:1]
sv2p = SV2P(params)
if args.batch:
    # pkl is a list of length num_episodes with entries of the form {obs: (steps+1, 56, 56, 3), acts: (steps, 4)}
    # output is of the same form with entries {preds: (steps+1-horizon, horizon, 56, 56, 3), acts: (steps+1-horizon, horizon, 4)}
    pkl = pickle.load(open(args.input_img, 'rb'))
    output = list()
    for episode in pkl:
        num_steps = len(episode['act'])
        all_acts = np.array(episode['act'])
        preds = []
        acts = []
        for i in range(num_steps + 1 - args.horizon):
            currim = episode['obs'][i]
            curracts = all_acts[i:i + args.horizon][np.newaxis,:]
            pred = sv2p.predict(currim, curracts)[0][0]
            preds.append(pred[args.horizon - 1]) # for VAE data gen just get the last image in the horizon
            acts.append(curracts[0])
        # CAREFUL!! these will not make sense visually unless they are np.uint8, but that can mess with L2 metrics
        curr_output = {'pred': np.array(preds).astype(np.uint8), 'act': np.array(acts)}
        output.append(curr_output)
    pickle.dump(output, open('predictions.pkl', 'wb'))
else:
    # input_img is a single image, and we take random actions
    currim = pickle.load(open(args.input_img, 'rb')) # input image
    acts1 = np.random.uniform(-1, 1, (1, 5, 2)) # modify as desired
    acts2 = np.random.uniform(-0.3, 0.3, (1, 5, 2))
    acts = np.dstack((acts1, acts2))
    pred = sv2p.predict(currim, acts)[0][0]
    d = {"act": acts[0], "pred": pred.astype(np.uint8)}
    pickle.dump(d, open('predictions.pkl', 'wb')) # output
print("Successfully wrote predictions to predictions.pkl")
