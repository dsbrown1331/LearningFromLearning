#from mcar_sarsa_semigrad_TileSutton import ValueFunction, run_episode, getAction


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




#rewards = []
if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("usage: python mcar_maxent_exp.py seed ndemos")

    #max ent parameter
    learning_rate = 0.01
    num_steps = 5

    percentage_skip = 0.8
    seed = int(sys.argv[1])
    reps = int(sys.argv[2])  #number of episodes to train learner on
    num_fcount_rollouts = 100
    eval_rollouts = 100
    skip_time = int(np.floor(percentage_skip * reps)) #number of episodes to skip when computing demo feature counts
    print(reps, skip_time)

    rbf_grid_size = 8
    assert(skip_time < reps)

    env = gym.make('MountainCar-v0')
    env.seed(seed)
    random.seed(seed)

    numOfTilings = 8
    alpha = 0.5
    n = 1


    # use optimistic initial value, so it's ok to set epsilon to 0
    EPSILON = 0
    discount = 0.999 #using high discount factor

    ##Debugging with optimal demonstrator. Max Ent works like a charm!
    #with open('opt_policy_ss.pickle', 'rb') as f:
    #    valueFunction = pickle.load(f)
    valueFunction = ValueFunction(alpha, numOfTilings)

    ##feature map
    features = []
#    centers = np.array([[0.0, 0.0], [0.0, 0.2], [0.0, 0.4], [0.0, 0.6], [0.0, 0.8], [0.0, 1.0],
#                        [0.25, 0.0], [0.25, 0.25], [0.25, 0.5], [0.25, 0.75], [0.25, 1.0],
#                        [0.5, 0.0], [0.5, 0.25], [0.5, 0.5], [0.5, 0.75], [0.5, 1.0],
#                        [0.75, 0.0], [0.75, 0.25], [0.75, 0.5], [0.75, 0.75], [0.75, 1.0],
#                        [1.0, 0.0], [1.0, 0.25], [1.0, 0.5], [1.0, 0.75], [1.0, 1.0]])
    centers = generate_grid_centers(rbf_grid_size);
    print(centers)
    widths = 0.15*np.ones(len(centers))

    rbfun = RBF(centers, widths, env.action_space.n)
    fMap = Rbf_2D_Feature_Map(rbfun)

    #generate plot of rbf activations

#    x = np.linspace(0,1)
#    y = np.ones(len(x))
#    activations = []
#    for i,x_i in enumerate(x):
#        activations.append(fMap.map_features([x_i,y[i]]))
#    print(activations)
#    plt.plot(x, activations)
#    plt.show()


    writer = open("data/mcar_maxent_seed" + str(seed) + "_demos" + str(reps), "w")
    for i in range(reps):
        print(">>>>iteration",i)


        reward, states_visited, steps = run_episode(env, valueFunction, n, False, EPSILON)
        #compute feature counts
        fcounts = compute_feature_counts(fMap, states_visited, discount, env)
        print("steps = ", steps)
        #print("feature count = ", fcounts)
        features.append(fcounts)
        
    features = np.array(features)

    #compute expected feature counts for demos
    emp_feature_cnts = np.mean(features[skip_time:], axis = 0)
    ment = MaxEnt(solve_mdp, fMap, env, num_fcount_rollouts, discount)
    maxent_value_fn = ment.get_opt_policy(emp_feature_cnts, learning_rate, num_steps)

    #pickle the controller (value function)
    #with open('mcar_maxent_policy_ss.pickle', 'wb') as f:
    #    pickle.dump(maxent_value_fn, f, pickle.HIGHEST_PROTOCOL)

    #with open('mcar_maxent_policy_ss.pickle', 'rb') as f:
    #    vFunc = pickle.load(f)

    #evaluate maxent learned policy
    returns = evaluate_policy(env, eval_rollouts, maxent_value_fn)
    print("average return", np.mean(returns))
    
    for r in returns:
        writer.write(str(r)+"\n")
    writer.close()
