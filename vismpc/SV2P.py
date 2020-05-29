from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import sys
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import video_utils
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
from tensor2tensor.bin.t2t_decoder import create_hparams
import h5py
import tensorflow as tf

class SV2P:
    """Forward model for SV2P (from saved-visual repo)
    """
    def __init__(self, params):
        """Initializes a class instance.

        Arguments:
            params (DotMap): A dotmap of model parameters.
                .name (str): Model name, used for logging/use in variable scopes.
                    Warning: Models with the same name will overwrite each other.
                .model_dir (str/None): (optional) Path to directory from which model will be loaded, and
                    saved by default. Defaults to None.
                .sess (tf.Session/None): The session that this model will use.
                    If None, creates a session with its own associated graph. Defaults to None.
        """
        self.name = params.name
        self.model_dir = params.model_dir
        self.data_dir = params.data_dir  
        self.popsize = params.popsize # Number of trajectories evaluated at a time
        self.nparts = params.nparts  # Number of particles
        self.H = params.plan_hor # planning horizon
        self.adim = params.adim # Action space dimension
        
        flags = tf.flags
        FLAGS = flags.FLAGS 
        FLAGS.data_dir = self.data_dir
        FLAGS.output_dir = self.model_dir
        FLAGS.problem = "cloth"
        FLAGS.hparams = "video_num_input_frames=1, video_num_target_frames=" + str(self.H)
        if not params.stochastic_model:
            FLAGS.hparams += ", stochastic_model=False"
        FLAGS.hparams_set = "next_frame_sv2p"
        FLAGS.model = "next_frame_sv2p"
        
        ### Creates initial set of hparams
        hparams = create_hparams()  
   
        self.frame_shape = hparams.problem.frame_shape # 56 x 56 x 3. modify in vismpc/t2t/tensor2tensor/data_generators/cloth.py
        l = [f for f in os.listdir(self.model_dir) if 'model.ckpt' in f]
        ckpt_path = "/" + l[-1][:l[-1].rfind('.')]
        ### Init Graph
        forward_graph = tf.Graph()
        with forward_graph.as_default():
            self._sess = tf.Session()
            input_size = [self.popsize*self.nparts, hparams.video_num_input_frames]
            target_size = [self.popsize*self.nparts, hparams.video_num_target_frames]
            ### Create Placeholders
            self.forward_placeholders = {
                "inputs": tf.placeholder(tf.float32, input_size + self.frame_shape),
                "input_action": tf.placeholder(tf.float32, input_size + [self.adim]),
                "targets": tf.placeholder(tf.float32, target_size + self.frame_shape),
                "target_action": tf.placeholder(tf.float32, target_size + [self.adim]),
            }
            ### Create Model
            forward_model_class = registry.model(FLAGS.model)
            self.forward_model = forward_model_class(hparams, tf.estimator.ModeKeys.PREDICT)
            self.forward_prediction_ops, _ = self.forward_model(self.forward_placeholders)
            ### Restore Weights
            forward_saver = tf.train.Saver()
            forward_saver.restore(self._sess, self.model_dir + ckpt_path)
            
        print("LOADED SV2P")

    @property
    def is_probabilistic(self):
        return True

    @property
    def is_tf_model(self):
        return True

    @property
    def sess(self):
        return self._sess

    def predict(self, currim, acts):      
        # currim: 48 x 64 x 3 tensor
        # actions: popsize*nparts x H x adim tensor
        ### Create Feed Dict
        forward_feed = {
            self.forward_placeholders["inputs"]: np.repeat(np.expand_dims(np.expand_dims(currim, 0),0), self.popsize*self.nparts, axis=0),
            self.forward_placeholders["input_action"]: acts[:, 0:1, :],
            self.forward_placeholders["targets"]: np.zeros(self.forward_placeholders["targets"].shape),
            self.forward_placeholders["target_action"]: acts,
        }
        
        predictions = self._sess.run(self.forward_prediction_ops, forward_feed)
        predictions = predictions.squeeze()
        predictions = predictions.reshape([self.popsize,self.nparts,self.H] + self.frame_shape)
        return predictions # shape = (popsize, nparts, H, 48, 64, 3)


    def create_prediction_tensors(self, currim, acts):
        return tf.reshape(tf.squeeze(self.forward_prediction_ops), [self.popsize,self.nparts,self.H] + self.frame_shape)

    def save(self, savedir=None):
        return None
