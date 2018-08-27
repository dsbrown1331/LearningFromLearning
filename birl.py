# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:32:59 2018

Using an MDP solver/RL method, run the Bayesian IRL algorithm

@author: dsbrown
"""

import numpy as np
import rbf
from mcar_sarsa_semigrad_TileSutton import ValueFunction, rollout, evaluate_policy
import math
from scipy.misc import logsumexp

class BIRL:
    def __init__(self, mdp_solver, rbf_fn, env, discount, alpha = 0.5, num_tilings = 8):
        self.mdp_solver = mdp_solver
        self.rbf_fn = rbf_fn
        self.env = env
        self.discount = discount
        self.alpha = alpha
        self.num_actions = env.action_space.n
        self.num_tilings = num_tilings
        


    def compute_likelihood(self, value_fn, demos, confidence):
        print("computing likelihood")
        log_posterior = 0
        for state, action in demos:
            #compute partition function
            Z = [confidence * value_fn.value(state[0], state[1], a) 
                                        for a in range(self.num_actions)]
            log_posterior += confidence * (value_fn.value(state[0], state[1], action))
            log_posterior -= logsumexp(Z)
        return log_posterior


    def get_opt_policy(self, demos, num_features, confidence, num_steps, step_size, time_limit = 1000):
        
        #initialize W
        W = 2*np.random.random(num_features)-1
        #normalize
        W = W / np.sum(np.abs(W))
        print("initial weights")
        #for i in range(num_features):
        #    print(W[i], end = ",")
        #print()
        #solve for optimal policy with R = W^T \phi(S)
        reward_fn = rbf.RbfReward(self.rbf_fn, W, self.env)
        value_fn = ValueFunction(self.alpha, self.num_tilings)
        print("solving mdp with sarsa semigrad")
        self.mdp_solver(self.env, value_fn, reward_fn, max_time = time_limit)
        likelihood_prev = self.compute_likelihood(value_fn, demos, confidence)
        print("initial likelihood", likelihood_prev)
        #initialize variables to keep track of MAP
        map_value_fn = value_fn
        map_reward_fn = reward_fn
        map_likelihood = likelihood_prev
        
        accept_cnt = 0

        for t in range(num_steps):
            print("---- BIRL iteration : ", t)
            #randomly tweak weights
            W_new = W + step_size * (2*np.random.random(num_features) - 1)
            #renormalize
            W_new = W_new / np.sum(np.abs(W_new))
            #PRINT OUT W's
            print("new weights")
            #for i in range(num_features):
            #    print(W_new[i], end = ",")
            #print()
            #compute an epsilon-optimal policy for the MDP
            #with R(s) = W ^T \phi(s)
            reward_fn_new = rbf.RbfReward(self.rbf_fn, W_new, self.env)


            value_fn_new = ValueFunction(self.alpha, self.num_tilings)
            print("solving mdp with sarsa semigrad")
            self.mdp_solver(self.env, value_fn_new, reward_fn_new, max_time = time_limit)

            likelihood_new = self.compute_likelihood(value_fn_new, demos, confidence)
            print("new likelihood", likelihood_new)

            if np.random.rand() < min(1, np.exp(likelihood_new - likelihood_prev)):
                print("accept")
                accept_cnt += 1
                likelihood_prev = likelihood_new
                value_fn = value_fn_new
                if likelihood_new > map_likelihood:
                    map_likelihood = likelihood_new
                    map_value_fn = value_fn_new
                    map_reward = reward_fn_new
                    print("updating best")
                    print("best likelihood", map_likelihood)


        print("num accepts = ", accept_cnt)
        return map_value_fn, map_reward_fn
