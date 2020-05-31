# VSF Instructions
First run `pip install -r requirements.txt` in this directory and the parent directory, and follow `gym-cloth` installation instructions in the parent directory's `README.md`. Unless otherwise specified, run all commands below from the parent directory `fabric-vsf`.

## Data collection
Run several simulations of the random policy:

`python scripts/run.py --max_episodes [x] --seed [y] random`

You will want at least ~100,000 transitions total, where transitions = episode length * number of episodes. Since SV2P will expect uniform episode length, you'll want to generate more episodes than you need so you have sufficient episodes of maximum length. To speed up data collection, use `screen`, `tmux`, or a similar utility to run `scripts/run.py` on multiple cores simultaneously. After collection, run `python vismpc/scripts/format_hdf5.py [FILEPATH.hdf5]` to create the HDF5 file to be consumed during training.

Parameters are set in `cfg/sim.yaml`. Double check the following values:
- I recommend `env > max_actions = 10`.
- Set `env > force_grab = False`.
- Set `env > use_dom_rand = True` to enable domain randomization.
- Set `env > use_rgbd = True` to collect RGBD images.
- ** Make sure to vary `init > type` roughly evenly across `tier0, tier1, tier2, tier3`.

## Training the visual dynamics model
You will need CUDA 10 and a GPU for this (and the next two steps). One way to do this is to create a Docker container.

First, open `vismpc/t2t/tensor2tensor/data_generators/cloth.py` and make sure `DATA_URL, NUM_EP,` and `EP_LEN` are set appropriately (Lines 39-41). `DATA_URL` is the path to your HDF5 file, and `NUM_EP` is printed when running `vismpc/scripts/format_hdf5.py` in the previous step. Also set `num_channels()` appropriately (Line 49) to 1, 3, or 4 for D, RGB, RGBD respectively. Rerun `python install -r requirements.txt` to install changes.

Then, run the following command:
`CUDA_VISIBLE_DEVICES=[#] t2t-trainer --schedule=train --alsologtostderr --generate_data --tmp_dir /data/sv2p_tmp_cloth --data_dir=[input data path] --output_dir=[output data path] --problem=cloth --model=next_frame_sv2p --hparams_set=next_frame_sv2p --train_steps=1000000 —eval_steps=100 --hparams="video_num_input_frames=[#], video_num_target_frames=[#]”`

This will take a day or two to train. For episode length 10, I used `video_num_input_frames=2, video_num_target_frames=5`, which predicts 5 output frames from 2 context frames. Feel free to tune these.

## Model inference
Run `CUDA_VISIBLE_DEVICES=[#] python vismpc/scripts/predict.py --input_img PATH [--model_dir MODEL_DIR --data_dir DATA_DIR --horizon H]` to see what the trained model predicts at test time. 

## Run the policy in sim
Run `python scripts/run.py --seed [SEED] --max_episodes [#] --model_path [path] vismpc`.

This will use the same `cfg/sim.yaml` as earlier. This time, do the following:
- Set `env > force_grab = True`.
- Set `env > use_dom_rand = False`.
- Set `env > goal_img` to a desired goal image. For example, `vismpc/goal_imgs/smooth.pkl`.
- Set `init > type` as desired.
- Set `env > viz_vismpc = True` to visualize the top 10 CEM elites at every iteration in `logs/debug`.

## Run the policy on the dVRK
Run `python vismpc/scripts/dvrk.py --model_path [model] --goal_img [goal]`. This will wait for robot observations to populate a specific directory and return the policy. See further instructions in our physical experiment repository [here](https://github.com/BerkeleyAutomation/dvrk-vismpc/tree/master).
