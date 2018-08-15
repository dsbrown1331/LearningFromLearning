#from mcar_sarsa_semigrad_TileSutton import ValueFunction, run_episode, getAction
import mcar_sarsa_semigrad_TileSutton
from mcar_sarsa_semigrad_TileSutton import ValueFunction, run_episode, getOptimalAction, run_rollout
import gym
import time
import numpy as np
from rbf import RBF

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
    
def compute_feature_counts(feature_map, states, discount):
    fcounts = np.zeros(feature_map.n_features)
    for i in range(len(states)):
        fcounts += discount ** i * feature_map.map(states[i])
    return fcounts


def get_feature_half_planes(state_feature_map, env, start_state, valueFunction, horizon, min_margin = 0, discount = 1.0, threshold = 0.0001):
    #run rollouts counting up the features for each possible action
    opt_action = getOptimalAction(start_state[0], start_state[1], valueFunction)
    half_plane_normals = set() #store the constraints in a set
    #get the feature counts from taking the optimal action
    #print("taking opt", opt_action)
    states_visited, opt_reward = run_rollout(env, start_state, opt_action, valueFunction, render=False)
    #print(states_visited)
    #compute feature counts
    opt_fcounts = compute_feature_counts(state_feature_map, states_visited, discount)
    #print(opt_fcounts)
    best_margin = 0
    #take other actions
    for init_action in range(env.action_space.n):
        if init_action != opt_action:
            #print("init action", init_action)
            states_visited, cum_reward = run_rollout(env, start_state, init_action, valueFunction, render=False)
            #make sure that opt_reward is no worse than cum_reward from other actions
            if opt_reward - cum_reward < 0:
                return set()  #return nothing if opt_action isn't really optimal
            #print(states_visited)
            #compute feature counts
            fcounts = compute_feature_counts(state_feature_map, states_visited, discount)
            #print(fcounts)
            #compute half-plane normal vector
            normal = opt_fcounts - fcounts
            if np.linalg.norm(normal, np.inf) > best_margin:
                best_margin = np.linalg.norm(normal, np.inf)
           
            #check if close to zero
            non_zero = False
            for i in range(len(normal)):
                if np.abs(normal[i]) > threshold:
                    non_zero = True
                else:
                    normal[i] = 0.0 #truncate the normal vector if less than threshold
            if non_zero:
                #normalize normal vector
                normal = normal / np.linalg.norm(normal)
                half_plane_normals.add(tuple(normal))
    if best_margin < min_margin:
        return set()
    else:
        return half_plane_normals
        
    
    
        
#rewards = []
if __name__ == "__main__":
    
    reps = 10  #number of episodes to train learner on
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
        rbf = RBF(np.array([[-1.2], [-0.3], [0.6]]), 0.7*np.ones(3), env.action_space.n)
        fMap = Rbf_Position_Feature_Map(rbf)
        
        steps, states_visited = run_episode(env, valueFunction, 1, False, EPSILON)
        
        #compute feature counts
        fcounts = compute_feature_counts(fMap, states_visited, discount)
        print("steps = ", steps)
        print("feature count = ", fcounts)
        
        features.append(fcounts)
        
    plt.plot(range(1,reps+1), features)
    plt.legend(['RBF(-1.2)','RBF(-0.3)','RBF(0.6)'])
    plt.xlabel("Number of episodes")
    plt.ylabel("Feature Counts")
    plt.show()
        

