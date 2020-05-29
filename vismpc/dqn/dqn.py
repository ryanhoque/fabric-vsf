# -*- coding: utf-8 -*-
"""
Training a DQN for goal-conditioned cost function
Adapted from https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py
"""

import gym
import math
import random
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import h5py
import argparse
import pickle

totensor = T.ToTensor()

def format_new_hdf5(old_filepath, new_filepath):
    """
    Convert output of vismpc/scripts/format_hdf5 into individual transitions
    """
    # assumes ep len = 10
    old = h5py.File(old_filepath, 'r')
    new = h5py.File(new_filepath, 'w')
    num_episodes = len(old['actions'])
    d_s = new.create_dataset('state', (num_episodes * 10, 56, 56, 4))
    d_a = new.create_dataset('action', (num_episodes * 10, 4))
    d_ns = new.create_dataset('next_state', (num_episodes * 10, 56, 56, 4))
    d_g = new.create_dataset('goal', (num_episodes * 10, 56, 56, 4))
    d_r = new.create_dataset('reward', (num_episodes * 10,))
    for i in range(num_episodes):
        d_s[i * 10:(i + 1) * 10] = old['images'][i][:10]
        d_a[i * 10: (i + 1) * 10] = old['actions'][i]
        d_ns[i * 10: (i + 1) * 10] = old['images'][i][1:]
        d_g[i * 10: (i + 1) * 10] = np.tile(old['images'][i][-1], (10,1,1,1))
        d_r[i * 10: (i + 1) * 10] = np.array([0.] * 9 + [1.])
        if i % 100 == 0:
            new.flush()
    new.flush()
    old.close()
    new.close()

def sample_data(dataset, batch_size, device):
    num_transitions = len(dataset['reward'])
    indices = sorted(random.sample(range(int(0.95 * num_transitions)), batch_size))
    states = torch.stack([totensor(img) for img in dataset['state'][indices,:,:,:]],dim=0)
    next_states = torch.stack([totensor(img) for img in dataset['next_state'][indices,:,:,:]],dim=0)
    goals = torch.stack([totensor(img) for img in dataset['goal'][indices,:,:,:]],dim=0)
    actions = torch.from_numpy(dataset['action'][indices,:])
    rewards = torch.from_numpy(dataset['reward'][indices])
    return states.to(device), actions.to(device), next_states.to(device), goals.to(device), rewards.to(device)

######################################################################
# Now, let's define our model
#

class DQN(nn.Module):
    def __init__(self, h, w, dim_actions):
        super(DQN, self).__init__()
        # process state
        self.conv1 = nn.Conv2d(4, 128, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(16)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=2, padding=1):
            return (size - kernel_size + 2*padding) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        self.linear_input_size = convw * convh * 16
        HIDDEN_DIM = 64 # TODO Tune this
        self.state_rep = nn.Linear(self.linear_input_size, HIDDEN_DIM)

        # process action
        self.act1 = nn.Linear(dim_actions, HIDDEN_DIM)
        self.act2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.act_rep = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

        # process goal
        self.conv4 = nn.Conv2d(4, 128, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv6 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(16)
        self.goal_rep = nn.Linear(self.linear_input_size, HIDDEN_DIM)

        # merge
        self.last_linear1 = nn.Linear(3*HIDDEN_DIM, HIDDEN_DIM)
        self.last_linear2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.last_linear3 = nn.Linear(HIDDEN_DIM, 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, s, a, g):
        # get state rep
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.state_rep(s.view(-1, self.linear_input_size)))

        # get action rep
        a = F.relu(self.act1(a))
        a = F.relu(self.act2(a))
        a = self.act_rep(a)

        # get goal rep
        g = F.relu(self.bn4(self.conv4(g)))
        g = F.relu(self.bn5(self.conv5(g)))
        g = F.relu(self.bn6(self.conv6(g)))
        g = F.relu(self.goal_rep(g.view(-1, self.linear_input_size)))

        # merge
        x = torch.cat([s, a, g], 1)
        x = F.relu(self.last_linear1(x))
        x = F.relu(self.last_linear2(x))
        return F.relu(self.last_linear3(x))

def optimize_model(dataset, optimizer, batch_size, gamma, policy_net, target_net, device, print_Q=False):
    s, a, ns, g, r = sample_data(dataset, batch_size, device)

    # Compute a mask of non-final states
    mask = torch.tensor(tuple(map(lambda r: r == 0.,
                                    r)), device=device, dtype=torch.bool)
    # Compute Q(s, a, g)
    state_action_values = policy_net(s, a, g) # detach?
    if print_Q:
        print("Q_val", state_action_values[:,0])
    # Compute V(s_{t+1}) for all next states with target net and action sampling
    next_state_values = torch.zeros(batch_size, device=device)
    NUM_ACTION_SAMPLES = 1000 # tune this
    nonfinal_size = len(ns[mask])
    grid = torch.zeros((nonfinal_size, NUM_ACTION_SAMPLES), device=device)
    a_samples = np.random.uniform([-1,-1,-0.7,-0.7], [1,1,0.7,0.7], (NUM_ACTION_SAMPLES, nonfinal_size, 4)).astype(np.float32)
    a_samples = torch.from_numpy(a_samples).to(device)
    for i in range(NUM_ACTION_SAMPLES):
        grid[:,i] = target_net(ns[mask], a_samples[i], g[mask]).detach()[:,0]
    next_state_values[mask] = torch.max(grid, 1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + r
    if print_Q:
        print("esav", expected_state_action_values)
    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    #print(policy_net.parameters())
#    if print_Q:
#    print(policy_net.act1.bias.grad)
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    #print(policy_net.parameters())
#    if print_Q:
#    print(policy_net.act1.bias.grad)
    #if print_Q:
    #    print(state_action_values.grad)
    optimizer.step()
    #if print_Q:
    #    print("Q grad", state_action_values.grad)
    return loss.item()

def get_model(model_path):
    device = torch.device("cuda",1)
    policy_net = DQN(56, 56, 4).to(device)
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()
    return policy_net

def get_Q(policy_net, s, a, g):
    # s, g is shape (batch_size, 56, 56, 4)
    # a is shape (batch_size, 4)
    device = torch.device("cuda",1)
    s = torch.stack([totensor(img) for img in s],dim=0).to(device)
    a = torch.from_numpy(a).to(device)
    g = torch.stack([totensor(img) for img in g],dim=0).to(device)
    return policy_net(s, a, g).detach()[:,0].cpu().numpy()

if __name__=='__main__':
    #format_new_hdf5('/data/edge_bias/edge_bias.hdf5', 'dqn.hdf5')
    parser = argparse.ArgumentParser(description='PyTorch DQN')
    parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 10)')
    parser.add_argument('--filepath', type=str, default='dqn_data.hdf5')
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor')
    parser.add_argument('--target_update', type=int, default=50,
                    help='how often to update target network')
    parser.add_argument('--continue_from', type=str, help='model path to continue training from')
    args = parser.parse_args()
    dataset = h5py.File(args.filepath, 'r') # load data
    # Hyperparams and other setup
    batch_size = args.batch_size
    epochs = args.epochs
    gamma = args.gamma
    target_update = args.target_update
    device = torch.device("cuda",1)
    policy_net = DQN(56, 56, 4).to(device)
    if args.continue_from:
        policy_net.load_state_dict(torch.load(args.continue_from))
    target_net = DQN(56, 56, 4).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    policy_net.eval()
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters())
    # Training Loop
    losses = []
    for e in range(epochs):
        for i in range(100000 // batch_size): # steps per epoch
            loss_ = optimize_model(dataset, optimizer, batch_size, gamma, policy_net, target_net, device, i%10==0)
            losses.append(loss_)
            print('Train Epoch: {} ({:.0f}%)\tLoss: {:.6f}'.format(
                e, 100. * i / (100000 // batch_size),loss_))
            if i % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
        torch.save(policy_net.state_dict(), 'models/Epoch_{}_loss_{:.4f}.pth'.format(e, loss_))
    pickle.dump(losses, open('losses.pkl', 'wb'))
    dataset.close()
