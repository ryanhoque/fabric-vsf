# VisuoSpatial Foresight (VSF) for Fabric Manipulation

This repository is built off the [gym-cloth repository](https://github.com/DanielTakeshi/gym-cloth), an RL environment wrapped around a fabric simulator. The repository includes code for fabric manipulation with VSF (see the project website [here](sites.google.com/view/fabric-vsf/home)), primarily in the `vismpc/` directory. The most relevant files for Visual MPC are `scripts/run.py`, `cfg/sim.yaml`, `gym_cloth/envs/cloth_env.py`, and `vismpc/mpc.py`. See `vismpc/README.md` for detailed instructions.

Instructions for gym-cloth in general are below:

## Gym Cloth

This creates a gym environment based on our cloth simulator. The path directory
is structured following [standard gym conventions][1], and we also include our
`.pyx` files here for Cython compilation.

Platforms tested:

- Mac OS X (renderer working)
- Ubuntu 16.04 (renderer not working, unfortunately)
- Ubuntu 18.04 (renderer working)

Python versions tested:

- Python 2.7
- Python 3.6

The code is hopefully agnostic to Python 2 or 3, however we strongly recommend
using Python 3. We have not done any serious Python 2 testing for many months.


## Installation and Code Usage

1. Make a new virtualenv. For example, if you're using Python 2 and you put
your environments in a directory `~/Envs`:

   ```
   virtualenv --python=python2 ~/Envs/py2-clothsim
   ```

2. Run `pip install -r requirements.txt`.

3. Run `python setup.py install`. This should automatically "cythonize" the
Python `.pyx` files and allow you to run `from gym_cloth import ...` regarless
of current directory. You will need to re-run this command after making any
changes in `gym_cloth/` or `vismpc/`, for example:

   ```
   python setup.py install ; python scripts/run.py
   ```

## Renderer Installation

To visualize the simulator in real time, you will need to install the OpenGL renderer,
which is currently an independent C++ program. However this won't be possible if running
Visual MPC on a remote server. In this case, there are other ways to inspect results 
(see `vismpc/README.md`).

These instructions have been tested on Mac OS X and Ubuntu 18.04. For some
reason, we have not been able to get this working for Ubuntu 16.04. For Ubuntu 18.04, you 
might need sudo access for `make -j4 install`.

1. Navigate to `render/ext/libzmq`. Run
```
mkdir build
cd build
cmake ..
make -j4 install
```
2. Navigate to `render/ext/cppzmq`. Again run
```
mkdir build
cd build
cmake ..
make -j4 install
```
3. Navigate to `render`. Run
```
mkdir build
cd build
cmake ..
make
```

Finally you should have an executable `clothsim` in `render/build`. **To test
that it is working, go to `render/build` and run `./clothsim` on the command
line. You should see an empty window appear. There should be no segmentation
faults.** Occasionally I have seen it fail on installed machines, but normally
rebooting fixes it.

To activate the renderer, set `init: render_opengl` to `True` in your cfg file.

Notes:

- If you make changes to `width`, `height`, or `render_port` in
  `cfg/env_config.yaml`, please also update `num_width_points`,
  `num_height_points`, and `render_port` respectively in
  `render/scene/pinned2.json`.

- It's easier to change the viewing angle by directly adjusting values in
  `clothSimulator.cpp`, rather than with the mouse and GUI. When you adjust the
  camera angles, be sure to re-compile the renderer using the instructions
  above. You only need to re-compile `render`, not the other two.

[1]:https://github.com/openai/gym/tree/master/gym/envs
[2]:https://github.com/openai/gym/pull/1314
