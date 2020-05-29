# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Berkeley (BAIR) robot pushing dataset.

Self-Supervised Visual Planning with Temporal Skip Connections
Frederik Ebert, Chelsea Finn, Alex X. Lee, and Sergey Levine.
https://arxiv.org/abs/1710.05268

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import video_utils
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
import h5py
import tensorflow as tf

DATA_URL = ("cloth.hdf5")
NUMEP = 13054
EPLEN = 11
ACTION_DIM = 4

@registry.register_problem
class SavedVisualForesight(video_utils.VideoProblem):

  @property
  def num_channels(self):
    return 3

  @property
  def frame_height(self):
    return 48

  @property
  def frame_width(self):
    return 64

  @property
  def is_generate_per_split(self):
    return True

  # num_train_files * num_videos * num_frames
  @property
  def total_number_of_frames(self):
    return NUMEP*EPLEN

  def max_frames_per_video(self, hparams):
    return 100

  @property
  def random_skip(self):
    return False

  @property
  def only_keep_videos_from_0th_frame(self):
    return False

  @property
  def use_not_breaking_batching(self):
    return True

  @property
  def extra_reading_spec(self):
    """Additional data fields to store on disk and their decoders."""
    data_fields = {
        "frame_number": tf.FixedLenFeature([1], tf.int64),
        "action":tf.FixedLenFeature([ACTION_DIM], tf.float32),
    }
    decoders = {
        "frame_number": tf.contrib.slim.tfexample_decoder.Tensor(
            tensor_key="frame_number"),
        "action": tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="action"),
    }
    return data_fields, decoders

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.modality = {"inputs": modalities.ModalityType.VIDEO,
                  "action":modalities.ModalityType.REAL_L2_LOSS,
                  "targets": modalities.ModalityType.VIDEO}
    p.vocab_size = {"inputs": 256,
                    "action":ACTION_DIM,
                    "targets": 256}

  def parse_frames(self, f, dataset_split):
    ims = f["images"][:]
    acts = f["actions"][:]
    
    if dataset_split == problem.DatasetSplit.TRAIN:
        start_ep, end_ep = 0, int(NUMEP * 0.8)
    else:
        start_ep, end_ep = int(NUMEP * 0.8), NUMEP
    for ep in range(start_ep, end_ep):
        for step in range(EPLEN):
            frame = ims[ep,step]
            action = acts[ep, step]
            yield step, frame, action
            
  def generate_samples(self, data_dir, tmp_dir, dataset_split):
#     path = generator_utils.maybe_download(
#         tmp_dir, os.path.basename(DATA_URL), DATA_URL)
    path= DATA_URL

    f = h5py.File(path, "r")

    for frame_number, frame, action in self.parse_frames(f, dataset_split):
      yield {
          "frame_number": [frame_number],
          "frame": frame,
          "action": action.tolist(),
      }
