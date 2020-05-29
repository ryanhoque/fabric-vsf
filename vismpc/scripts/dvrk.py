"""
Script to connect VISMPC to the dVRK. Continually consumes images from a directory and returns actions.
"""
import pkg_resources
import numpy as np
import argparse
import os
from os.path import join
import sys
import time
import yaml
import logging
import pickle
import datetime
import cv2
from gym_cloth.envs import ClothEnv
from collections import defaultdict
from vismpc.mpc import VISMPC
from vismpc.cost_functions import coverage, L2, SSIM
np.set_printoptions(edgeitems=10, linewidth=180, suppress=True)

# adjust as necessary, and pack the goal image (comment out the others).
DVRK_IMG_PATH = '/data/dir_for_imgs/' 
_HEAD = 'vismpc/goal_imgs'
#GOAL_IMG = join(_HEAD, 'smooth.pkl')
#GOAL_IMG = join(_HEAD, 'rgbd_corner_56x56_topleft.pkl')
#GOAL_IMG = join(_HEAD, 'rgbd_corner_56x56_topright.pkl')
#GOAL_IMG = join(_HEAD, 'rgbd_corner_56x56_bottomright.pkl')
#GOAL_IMG = join(_HEAD, 'rgbd_corner_56x56_bottomleft.pkl')
#GOAL_IMG = join(_HEAD, 'rgbd_fileslash_bottomabove_56x56.pkl')
#GOAL_IMG = join(_HEAD, 'rgbd_fileslash_topabove_56x56.pkl')
GOAL_IMG = join(_HEAD, 'rgbd_latexslash_bottomabove_56x56.pkl')
#GOAL_IMG = join(_HEAD, 'rgbd_latexslash_topabove_56x56.pkl')


def get_sorted_imgs():
    res = sorted(
        [join(DVRK_IMG_PATH,x) for x in os.listdir(DVRK_IMG_PATH) \
            if x[-4:]=='.png']
    )
    return res

def get_net_results():
    net_results = sorted(
        [join(DVRK_IMG_PATH,x) for x in os.listdir(DVRK_IMG_PATH) \
            if 'result_' in x and '.txt' in x]
    )
    return net_results


if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument("--model_path", type=str, help="SV2P model's parent directory", default="/data/pure_random")
    pp.add_argument("--cost_fn", type=str, help="SV2P cost function among ('L2', 'SSIM', 'coverage')", default="L2")
    pp.add_argument("--goal_img", type=str, help="goal image filepath", default=GOAL_IMG)
    args = pp.parse_args()
    raw_img_shape = (56,56,3) 

    dvrk_img_paths = get_sorted_imgs()
    net_results = get_net_results()
    if len(dvrk_img_paths) > 0:
        print('There are {} images in {}. Please remove it/them.'.format(
                len(dvrk_img_paths), DVRK_IMG_PATH))
        print('It should be empty to start an episode.')
        sys.exit()
    if len(net_results) > 0:
        print('There are {} results in {}. Please remove it/them.'.format(
                len(net_results), DVRK_IMG_PATH))
        print('It should be empty to start an episode.')
        sys.exit()

    goal_img = pickle.load(open(args.goal_img,'rb'))

    if args.cost_fn == 'L2':
        cost_fn = lambda traj: L2(traj, goal_img)
    elif args.cost_fn == 'coverage':
        cost_fn = coverage
    elif args.cost_fn == 'SSIM':
        cost_fn = lambda traj: SSIM(traj, goal_img)
    else:
        print("Invalid cost function.")
        sys.exit()

    # Daniel: adjust the time horizon if desired!! Ryan set it to 5 by default.
    vismpc = VISMPC(cost_fn,
                    join(args.model_path, 'sv2p_data_cloth'),
                    join(args.model_path, 'sv2p_model_cloth'),
                    horizon=8,
                    batch=False)
    vismpc.reset()

    print('\nNetwork loaded successfully!')
    print('Now we\'re waiting for images in: {}'.format(DVRK_IMG_PATH))
    t_start = time.time()
    beeps = 0
    nb_prev = 0
    nb_curr = 0

    while True:
        time.sleep(1)
        beeps += 1
        t_elapsed = (time.time() - t_start)
        if beeps % 5 == 0:
            print('  time: {:.2f}s (i.e., {:.2f}m)'.format(t_elapsed, t_elapsed/60.))
        
        # -------------------------------------------------------------------- #
        # HUGE ASSUMPTION: assume we store image sequentially and do not
        # override them. That means the images should be appearing in
        # alphabetical order in chronological order. We can compute statistics
        # about these and the actions in separate code. Also, the images should
        # be saved by the ZividCamera.py script, which ALREADY does processing!
        # -------------------------------------------------------------------- #
        dvrk_img_paths = get_sorted_imgs()
        #nb_curr = len(dvrk_img_paths) # No, we save MANY images.
        if len(dvrk_img_paths) == 0:
            nb_curr = 0
        else:
            # We really want this to represent 'number of image groups' so 1-idx.
            nb_curr = int((os.path.basename(dvrk_img_paths[-1])).split('-')[0]) + 1
        #print(len(dvrk_img_paths), nb_prev, nb_curr)

        # Usually this equality should be happening. Means we just skip the below code.
        if nb_prev == nb_curr:
            continue
        if nb_prev+1 < nb_curr:
            print('Error, prev {} and curr {}, should only differ by one.'.format(
                    nb_prev, nb_curr))
        nb_prev = nb_curr

        # Now we load! We cannot just load the last one, must load c and d. COMBINE THEM.
        # E.g.: dir_for_imgs/000-c_img_crop_proc.png, dir_for_imgs/000-d_img_crop_proc.png
        # Note that this is one MINUS nb_curr ... that is 1-idx'd.
        time.sleep(1)  # just in case delays happen
        c_path = join(DVRK_IMG_PATH,
                      '{}-c_img_crop_proc_56.png'.format(str(nb_curr-1).zfill(3)))
        d_path = join(DVRK_IMG_PATH,
                      '{}-d_img_crop_proc_56.png'.format(str(nb_curr-1).zfill(3)))
        print('reading from: {} and {}'.format(c_path, d_path))
        dvrk_c_img = cv2.imread(c_path)
        dvrk_d_img = cv2.imread(d_path)
        assert dvrk_c_img.shape == dvrk_d_img.shape == raw_img_shape
        img = np.dstack( (dvrk_c_img, dvrk_d_img[:,:,0]) )
        policy_action = vismpc.get_next_action(img) # get action from vismpc controller
        net_results = get_net_results()
        nb_calls = len(net_results)+1
        date = '{}'.format( datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') )
        pol = args.model_path.split('/')[-1]
        tail = 'result_{}_{}_{}_num_{}.txt'.format(date, pol, args.cost_fn, str(nb_calls).zfill(3))
        save_path = join(DVRK_IMG_PATH, tail)
        np.savetxt(save_path, policy_action, fmt='%f')
        print('Just did action #{}, with result: {}'.format(nb_calls, policy_action))
        print('Saving to: {}'.format(save_path))
