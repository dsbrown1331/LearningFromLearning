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
    
def compute_feature_counts(feature_map, states, discount, env):
    fcounts = np.zeros(feature_map.n_features)
    for i in range(len(states)):
        #print("state:", states[i])
        #normalize state features
        f_normed = normalize_obs(states[i], env)
        #print("normed:", f_normed)
        #print("f:", feature_map.map(f_normed))
        #print("fcount+=", discount ** i * feature_map.map(f_normed))
        fcounts += discount ** i * feature_map.map(f_normed)
        #print("fcount", fcounts)
    return fcounts


    
    
        
#rewards = []
if __name__ == "__main__":
    
    reps = 100  #number of episodes to train learner on
    env = gym.make('MountainCar-v0')      
    numOfTilings = 8
    alpha = 0.5

    # use optimistic initial value, so it's ok to set epsilon to 0
    EPSILON = 0
    discount = 1.0 #using no discount factor for now

    valueFunction = ValueFunction(alpha, numOfTilings)
  
    features = []
  
    for i in range(reps):
        print(">>>>iteration",i)
        #pick feature map
        #fMap = Constant_Feature_Map()
        rbf = RBF(np.array([[0.0], [0.5], [1.0]]), 0.01*np.ones(3), env.action_space.n)
        fMap = Rbf_Position_Feature_Map(rbf)
        
        steps, states_visited = run_episode(env, valueFunction, 1, False, EPSILON)
        
        #normed = [normalize_obs(s, env) for s in states_visited]
        #s_n = [(states_visited[i],normed[i]) for i in range(len(states_visited))]
        #for thing in s_n:
        #    print(thing)
        #compute feature counts
        fcounts = compute_feature_counts(fMap, states_visited, discount, env)
        print("steps = ", steps)
        print("feature count = ", fcounts)
        
        features.append(fcounts)
        
    features = np.array(features)
    legends = ['RBF(0.0)','RBF(0.5)','RBF(1.0)']
    for f in range(len(features[0])):
        plt.figure(f)
        plt.plot(range(1,reps+1), features[:,f])
        plt.legend([legends[f]])
        plt.xlabel("Number of episodes")
        plt.ylabel("Feature Counts")
    plt.show()
        

