import moviepy.editor as mpy
import pickle, sys, os

OUTPUT_DIR = 'episodes'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if len(sys.argv) < 2:
    sys.exit("Usage: python makegif.py [log_file].pkl")

def npy_to_gif(im_list, filename, fps=2):
    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')

file = pickle.load(open(sys.argv[1], 'rb'))

for e in range(len(file)):
    episode = file[e]
    npy_to_gif(episode['obs'], '{}/ep{}'.format(OUTPUT_DIR, e))
