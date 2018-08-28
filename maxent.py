# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:32:59 2018

Using an MDP solver/RL method, run the Maximum Entropy algorithm

@author: dsbrown
"""

import numpy as np
import rbf
from mcar_sarsa_semigrad_TileSutton import ValueFunction, rollout
import math

class MaxEnt:
    def __init__(self, mdp_solver, rbf_fn, env, num_rollouts, discount, alpha = 0.5, num_tilings = 8):
        self.mdp_solver = mdp_solver
        self.rbf_fn = rbf_fn
        self.env = env
        self.num_rollouts = num_rollouts
        self.discount = discount
        self.alpha = alpha
        self.num_tilings = num_tilings




    def get_opt_policy(self, emp_f_counts, learning_rate, num_steps):
        k = len(emp_f_counts)

        #initialize W
        W = np.zeros(k)
        for t in range(num_steps):
            print("---- MaxEnt iteration : ", t)
            #PRINT OUT W's
            print("weights")
            for i in range(k):
                print(W[i],)
            print()
            #compute an epsilon-optimal policy for the MDP
            #with R(s) = W ^T \phi(s)
            reward_fn = rbf.RbfReward(self.rbf_fn, W, self.env)


            value_fn = ValueFunction(self.alpha, self.num_tilings)
            print("solving mdp with sarsa semigrad")
            self.mdp_solver(self.env, value_fn, reward_fn, max_time = 500)

            #debug watch policy
            #evaluate_policy(self.env, 4, value_fn)


            #compute an epsilon-good estimate of expected feature counts
            fcounts_pi = rbf.get_expected_feature_counts_softmax(self.num_rollouts, self.rbf_fn, value_fn, self.env, self.discount)
            print("expected f counts")
            for f in fcounts_pi:
                print(f,)
            print()

            #print size of the gradient
            grad_mag = 0.0
            for i in range(k):
                grad_mag += (emp_f_counts[i] - fcounts_pi[i]) ** 2;
            print("Gradient = ", math.sqrt(grad_mag))


            #update W
            for i in range(k):
                W[i] += learning_rate * (emp_f_counts[i] - fcounts_pi[i]);
        return value_fn
