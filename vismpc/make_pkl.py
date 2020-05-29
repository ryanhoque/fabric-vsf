"""
Daniel: Visual foresight requires pickle files of the 4-channel image.
"""
import pickle                                                                                                                                                                                       
import cv2                                                                                                                                                                                          
import numpy as np                                                                                                                                                                                  

# in 'vismpc/goal_imgs'.
# Use these for the suffix of g_file:
#   000 = topleft
#   001 = topright
#   002 = bottomright
#   003 = bottomleft
# then need dstack to combine along third axis.
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.dstack.html

# With single fold.
#c_file = 'goal_imgs/003-c_img_crop_proc_56.png'
#d_file = 'goal_imgs/003-d_img_crop_proc_56.png'
#g_file = 'goal_imgs/rgbd_corner_56x56_bottomleft.pkl'

# With two folds.

# latex slash. Top above.
#c_file = 'goal_imgs/000-c_img_crop_proc_56_diagslash_1.png'
#d_file = 'goal_imgs/000-d_img_crop_proc_56_diagslash_1.png'
#g_file = 'goal_imgs/rgbd_latexslash_topabove_56x56.pkl'

# bottom part above.
#c_file = 'goal_imgs/001-c_img_crop_proc_56_diagslash_2.png'
#d_file = 'goal_imgs/001-d_img_crop_proc_56_diagslash_2.png'
#g_file = 'goal_imgs/rgbd_latexslash_bottomabove_56x56.pkl'

# filename slash. top above
#c_file = 'goal_imgs/002-c_img_crop_proc_56_backslash_1.png'
#d_file = 'goal_imgs/002-d_img_crop_proc_56_backslash_1.png'
#g_file = 'goal_imgs/rgbd_fileslash_topabove_56x56.pkl'

# bottom part above
c_file = 'goal_imgs/003-c_img_crop_proc_56_backslash_2.png'
d_file = 'goal_imgs/003-d_img_crop_proc_56_backslash_2.png'
g_file = 'goal_imgs/rgbd_fileslash_bottomabove_56x56.pkl'

c_img = cv2.imread(c_file)
d_img = cv2.imread(d_file)
assert c_img.shape == d_img.shape == (56,56,3)
combo = np.dstack( (c_img, d_img[:,:,0]) )
assert combo.shape == (56,56,4)

with open(g_file, 'wb') as fh:
    pickle.dump(combo, fh)
print('just saved: {}'.format(g_file))
