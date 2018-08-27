#from mcar_sarsa_semigrad_TileSutton import ValueFunction, run_episode, getAction


from mcar_sarsa_semigrad_TileSutton import ValueFunction, run_episode, getOptimalAction, run_rollout, solve_mdp, evaluate_policy
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

    reps = 10  #number of episodes to train learner on
    num_fcount_rollouts = 100
    eval_rollouts = 100
    skip_time = 7 #number of episodes to skip when computing demo feature counts
    mwal_iter = 1 #number of times to run MWAL
    rbf_grid_size = 8
    assert(skip_time < reps)

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


    writer = open("data/mountain_car_test_data.txt", "w")
    demo_writer = open("data/mountain_car_demos.txt", "w")
    fcount_writer = open("data/mountain_car_fcounts.txt", "w")
    writer.write("#demos\n")
    for i in range(reps):
        print(">>>>iteration",i)


        reward, states_visited, steps = run_episode(env, valueFunction, n, False, EPSILON)
        for s in states_visited:
            demo_writer.write(str(s))
            demo_writer.write("\n")
        demo_writer.write("----\n")
        #compute feature counts
        fcounts = compute_feature_counts(fMap, states_visited, discount, env)
        print("steps = ", steps)
        #print("feature count = ", fcounts)
        writer.write(str(reward) + "\n")
        features.append(fcounts)
        for i in range(len(fcounts)-1):
            fcount_writer.write(str(fcounts[i]) + ",")
        fcount_writer.write(str(fcounts[-1]) + "\n")
    demo_writer.close()
    fcount_writer.close()

    features = np.array(features)

    flabels = [str(c) for c in centers]
    sign_finder = FeatureSignExtractor(features, flabels)
    slopes = sign_finder.estimate_signs()
    fsigns = np.sign(slopes)

    signedfMap = rbf.SignedRbf_2D_Feature_Map(rbfun, fsigns)
#    for f in range(len(features[0])):
#        plt.figure(f)
#        plt.plot(range(1,reps+1), features[:,f])
#        plt.legend([flabels[f]])
#        plt.xlabel("Number of episodes")
#        plt.ylabel("Feature Counts")
#
#
    print("positive slopes")
    for i,s in enumerate(slopes):
        if s > 0:
            print(flabels[i], s)
#    plt.show()
    #compute expected feature counts for demos
    emp_feature_cnts = np.mean(features[skip_time:], axis = 0)
    mwal = MWAL(solve_mdp, signedfMap, env, num_fcount_rollouts, discount)
    mwal_value_fn = mwal.get_opt_policy(emp_feature_cnts, mwal_iter)

    #pickle the controller (value function)
    with open('mcar_mwal_policy_ss.pickle', 'wb') as f:
        pickle.dump(mwal_value_fn, f, pickle.HIGHEST_PROTOCOL)

    with open('mcar_mwal_policy_ss.pickle', 'rb') as f:
        vFunc = pickle.load(f)

    #evaluate mwal learned policy
    returns = evaluate_policy(env, eval_rollouts, mwal_value_fn)
    print("average return", np.mean(returns))
    writer.write("#learned policy\n")
    for r in returns:
        writer.write(str(r)+"\n")
    writer.close()
