#!/usr/bin/env python
from mcar_sarsa_semigrad_TileSutton import ValueFunction, run_episode, getOptimalAction, run_rollout, solve_mdp, evaluate_policy
import gym
import time
import numpy as np
import rbf
from rbf import RBF, normalize_obs, generate_grid_centers, Rbf_2D_Feature_Map, compute_feature_counts
from feature_extrapolation import FeatureSignExtractor
from maxent import MaxEnt
import pickle
import random
import sys


import matplotlib.pyplot as plt

print(sys.argv[1])
print(sys.argv[2])

#testing script for condor
print("hi there")
for i in range(100):
    print(i)
