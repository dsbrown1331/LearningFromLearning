# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:32:59 2018

Using an MDP solver/RL method, run the Syed and Schapire algorithm to optimize performance
with respect to signs of features extracted. I'm using the simpler algorithm
described in "Apprenticeship Learning using Linear Programming"

@author: dsbrown
"""

import numpy as np
import rbf
from mcar_sarsa_semigrad_TileSutton import ValueFunction, rollout, evaluate_policy

class MWAL:
    def __init__(self, mdp_solver, rbf_fn, env, num_rollouts, discount, alpha = 0.5, num_tilings = 8):
        self.mdp_solver = mdp_solver
        self.rbf_fn = rbf_fn
        self.env = env
        self.num_rollouts = num_rollouts
        self.discount = discount
        self.alpha = alpha
        self.num_tilings = num_tilings
        self.value_fns = []

        

        
    def get_opt_policy(self, emp_f_counts, T):
        k = len(emp_f_counts)
        beta = 1.0/( 1.0 + np.sqrt(2.0 * np.log(k) / T));
        
        #initialize W
        W = np.ones(k)
        for t in range(1,T+1):
            print("---- MWAL iteration : ", t)
            #normalize the W's 
            W = W / np.sum(W)
            #PRINT OUT W's
            print("weights")
            for i in range(k):
                print(W[i],)
            print()
            print(np.sum(np.abs(W)))
            #compute an epsilon-optimal policy for the MDP
            #with R(s) = W_norm ^T \phi(s)
            reward_fn = rbf.RbfReward(self.rbf_fn, W, self.env)

            
            value_fn = ValueFunction(self.alpha, self.num_tilings)
            self.value_fns.append(value_fn)
            print("solving mdp with sarsa semigrad")
            self.mdp_solver(self.env, value_fn, reward_fn)
                       
            #debug watch policy
            #evaluate_policy(self.env, 4, value_fn)
            
    
            #compute an epsilon-good estimate of expected feature counts
            fcounts_pi = rbf.get_expected_feature_counts(self.num_rollouts, self.rbf_fn, value_fn, self.env, self.discount)
            print("expected f counts")            
            for f in fcounts_pi:
                print(f,)
            print()
    
            #update W values 
            #calculate tildeG
            fcount_diffs = np.array(fcounts_pi - emp_f_counts)
            W = W * (beta ** fcount_diffs)
        return self.value_fns
           
           
            
       
            
    
