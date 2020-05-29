"""
Machinery for running MPC with CEM, based on saved-visual/dmbrl/controllers/VIS_MPC.py
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import tensorflow as tf
import numpy as np
import scipy.stats as stats
from vismpc.SV2P import SV2P
from dotmap import DotMap
import sys

class VISMPC():
    def __init__(self, cost_fn, data_dir='vismpc/sv2p_data_cloth', model_dir='vismpc/sv2p_model_cloth', horizon=5, batch=False, viz=None):
        """
        cost_fn: higher-order function 
        """
        params = DotMap()
        params.name = 'cloth'
        params.model_dir = model_dir
        params.data_dir = data_dir
        params.popsize = 2000  # must match _run_cem's popsize!
        params.nparts = 1
        params.plan_hor = horizon
        params.adim = 4
        params.stochastic_model = True
        sys.argv = sys.argv[:1]
        self.model = SV2P(params)
        self.cost_fn = cost_fn
        # TUNE CEM VARIANCE:
        # -0.4/0.4 work better for smoothing, -0.7/0.7 better for folding
        self.ac_lb = np.array([-1., -1., -0.7, -0.7])
        self.ac_ub = np.array([1., 1., 0.7, 0.7])
        self.act_dim = 4
        self.plan_hor = horizon
        self.batch = batch
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        # / 16 works better for smoothing, /8 better for folding
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 8, [self.plan_hor])
        self.viz = viz

    def reset(self):
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])

    def set_cost_function(self, cost_fn):
        self.cost_fn = cost_fn

    def get_next_action(self, obs, timestep=None):
        soln = self._run_cem(obs, mean=self.prev_sol, var=self.init_var, timestep=timestep)
        self.prev_sol = np.concatenate([np.copy(soln)[self.act_dim:], np.zeros(self.act_dim)])
        return soln[:self.act_dim]

    def _run_cem(self, obs, mean, var, num_iters=10, timestep=None):
        num_elites, alpha, popsize = 400, 0.1, 2000
        print('running cem with num iters {}, popsize {}'.format(num_iters, popsize))
        lb = np.tile(self.ac_lb, [self.plan_hor])
        ub = np.tile(self.ac_ub, [self.plan_hor])
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean))
        if self.viz:
            self.viz.set_context(obs)
        for i in range(num_iters):
            lb_dist, ub_dist = mean - lb, ub - mean
            constrained_var = var
            #constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)
            samples = X.rvs(size=[popsize, self.plan_hor * self.act_dim]) * np.sqrt(constrained_var) + mean
            costs, pred_trajs = self._predict_and_eval(obs, samples, timestep=timestep)
            print("CEM Iteration: ", i, "Cost (mean, std): ", np.mean(costs), ",", np.std(costs))
            elites = samples[np.argsort(costs)][:num_elites]
            if self.viz:
                elite_trajs = pred_trajs[np.argsort(costs)][:num_elites]
                self.viz.set_grid(elite_trajs[:10].reshape((10,self.plan_hor,56,56,obs.shape[2])), elites[:10].reshape((10,self.plan_hor,self.act_dim)), np.sort(costs)[:10])
                self.viz.render_image('logs/debug/t={}i={}.jpg'.format(timestep, i))
            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)
            # refit mean/var
            mean, var = alpha * mean + (1 - alpha) * new_mean, alpha * var + (1 - alpha) * new_var
        return mean

    def _predict_and_eval(self, obs, ac_seqs, timestep=None):
        ac_seqs = np.reshape(ac_seqs, [-1, self.plan_hor, self.act_dim])
        pred_trajs = self.model.predict(obs, ac_seqs)
        # since feed_dict in SV2P is going to require np arrays
        if self.batch:
            costs = self.cost_fn(pred_trajs[:,0])
        else:
            costs = []
            for traj in pred_trajs:
                traj = traj[0]
                costs.append(self.cost_fn(traj))
        return np.array(costs), pred_trajs[:,0]

