import cv2
import numpy as np
import torch
from skimage.measure import compare_ssim
import pickle

# from vismpc.vae import vae
# from vismpc.dqn import dqn
# load_dqn = True
# load_vae = True

def L2(traj, goal_img):
    """
    average L2 difference in the last image
    """
    return np.linalg.norm(traj[-1,7:-7,7:-7,:] - goal_img[7:-7,7:-7,:])

def SSIM(traj, goal_img):
    return 1 - compare_ssim(traj[-1,7:-7,7:-7,:].astype(np.uint8), goal_img[7:-7,7:-7,:], multichannel=True)    

def DQN(trajs, acts, goal_img, model_path):
    # See vismpc/dqn/dqn.py. Assumes batch mode.
    global load_dqn
    if load_dqn:
        policy_net = dqn.get_model(model_path)
        load_dqn = False
    goal_img_reshape = np.tile(goal_img, (len(trajs),1,1,1))
    acts_ = np.zeros(acts.shape).astype(np.float32)
    return -1 * dqn.get_Q(policy_net, trajs[:,1], acts_[:,-1], goal_img_reshape) 

def VAE(trajs, goal_img, model_path, timestep=None):
    # See vismpc/vae/vae.py. Assumes batch mode.
    global load_vae
    if load_vae:
        model = vae.load_model(model_path)
        model.eval()
        load_vae = False
    if not timestep:
        index = -1
    elif timestep and timestep < 6: # assuming ep len 10, horizon 5
        index = -1
    else: # dynamic horizon: shorten near the end of the episode
        index = 4 - timestep
    z = vae.encode_all(model, trajs[:,index])
    goal = vae.encode_two(model, goal_img, goal_img)[0]
    return np.linalg.norm(z - goal, axis=1)

def coverage(traj, timestep=None):
    """
    (For the smoothing task) return 1 - coverage of the final image
    Assumes no domain randomization.
    traj: H x height x width x channels
    """
    img = traj[-1]
    # 7 pixel padding for empty space that doesn't count
    img = img[7:-7,7:-7,:3]
    # compute fraction of non-orange pixels
    lower = np.array([200, 100, 0], dtype="uint8")
    upper = np.array([255, 200, 100], dtype="uint8")
    mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask=mask)
    tot_pixel = output.size
    orange = np.count_nonzero(output)
    return 1 - orange / tot_pixel
