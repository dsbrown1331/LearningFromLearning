#from mcar_sarsa_semigrad_TileSutton import ValueFunction, run_episode, getAction


from mcar_sarsa_semigrad_TileSutton import ValueFunction, run_episode, getOptimalAction, rollout, solve_mdp, evaluate_policy
import gym
import time
import numpy as np
import rbf
from rbf import RBF, normalize_obs, generate_grid_centers, Rbf_2D_Feature_Map, compute_feature_counts
from feature_extrapolation import FeatureSignExtractor
from mwal import MWAL
import pickle
import random

import matplotlib.pyplot as plt




#rewards = []
if __name__ == "__main__":

    reps = 1000  #number of episodes to train learner on

    env = gym.make('MountainCar-v0')
    env.seed(1234)
    random.seed(12345)

    numOfTilings = 8
    alpha = 0.5
    n = 1


    # use optimistic initial value, so it's ok to set epsilon to 0
    EPSILON = 0
    discount = 0.999 #using high discount factor

    valueFunction = ValueFunction(alpha, numOfTilings)


    for i in range(reps):
        print(">>>>iteration",i)


        reward, states_visited, steps = run_episode(env, valueFunction, n, False, EPSILON)
        print(reward)
    #pickle the controller (value function)
    with open('opt_policy_ss.pickle', 'wb') as f:
        pickle.dump(valueFunction, f, pickle.HIGHEST_PROTOCOL)

    with open('opt_policy_ss.pickle', 'rb') as f:
        vFunc = pickle.load(f)

    for i in range(3):
        rollout(env, valueFunction, True)
