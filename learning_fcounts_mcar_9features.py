#from mcar_sarsa_semigrad_TileSutton import ValueFunction, run_episode, getAction
import mcar_sarsa_semigrad_TileSutton
from mcar_sarsa_semigrad_TileSutton import ValueFunction, run_episode, getOptimalAction, run_rollout
import gym
import time
import numpy as np
from rbf import RBF, normalize_obs


import matplotlib.pyplot as plt

class Constant_Feature_Map():
    def __init__(self):
        self.n_features = 1
    def map(self, state):
        return np.array([1.0])
        
class Rbf_Position_Feature_Map():
    def __init__(self, rbf):
        self.n_features = len(rbf.c)
        self.rbf = rbf
    def map(self, state): 
        #just map position
        #print(state)
        #print(state[0])
        return np.array(self.rbf.get_rbf_activations([state[0]]))
        
class Rbf_2D_Feature_Map():
    def __init__(self, rbf):
        self.n_features = len(rbf.c)
        self.rbf = rbf
    def map(self, state): 
        #just map position
        #print(state)
        #print(state[0])
        return np.array(self.rbf.get_rbf_activations(state))
    
def compute_feature_counts(feature_map, states, discount, env):
    fcounts = np.zeros(feature_map.n_features)
    for i in range(len(states)):
        #normalize state features
        f_normed = normalize_obs(states[i], env)
        fcounts += discount ** i * feature_map.map(f_normed)
    return fcounts


    
    
        
#rewards = []
if __name__ == "__main__":
    
    reps = 40  #number of episodes to train learner on
    env = gym.make('MountainCar-v0')      
    numOfTilings = 8
    alpha = 0.5

    # use optimistic initial value, so it's ok to set epsilon to 0
    EPSILON = 0
    discount = 1.0 #using no discount factor for now

    valueFunction = ValueFunction(alpha, numOfTilings)
  
    ##feature map
    features = []
    centers = np.array([[0.0, 0.0], [0.0, 0.25], [0.0, 0.5], [0.0, 0.75], [0.0, 1.0],
                        [0.25, 0.0], [0.25, 0.25], [0.25, 0.5], [0.25, 0.75], [0.25, 1.0],
                        [0.5, 0.0], [0.5, 0.25], [0.5, 0.5], [0.5, 0.75], [0.5, 1.0],
                        [0.75, 0.0], [0.75, 0.25], [0.75, 0.5], [0.75, 0.75], [0.75, 1.0],   
                        [1.0, 0.0], [1.0, 0.25], [1.0, 0.5], [1.0, 0.75], [1.0, 1.0]])
    widths = 0.15*np.ones(len(centers))
    
    rbf = RBF(centers, widths, env.action_space.n)
    fMap = Rbf_2D_Feature_Map(rbf)    
    
    #generate plot of rbf activations
    
    x = np.linspace(0,1)
    y = np.ones(len(x))
    activations = []
    for i,x_i in enumerate(x):
        activations.append(fMap.map([x_i,y[i]]))
    print(activations)
    plt.plot(x, activations)
    plt.show()
  
    for i in range(reps):
        print(">>>>iteration",i)
 
        
        steps, states_visited = run_episode(env, valueFunction, 1, False, EPSILON)
        
        #compute feature counts
        fcounts = compute_feature_counts(fMap, states_visited, discount, env)
        print("steps = ", steps)
        print("feature count = ", fcounts)
        
        features.append(fcounts)
        
    features = np.array(features)
    legends = [str(c) for c in centers]
    for f in range(len(features[0])):
        plt.figure(f)
        plt.plot(range(1,reps+1), features[:,f])
        plt.legend([legends[f]])
        plt.xlabel("Number of episodes")
        plt.ylabel("Feature Counts")
    
    plt.show()
        

