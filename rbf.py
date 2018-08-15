import numpy as np
import matplotlib.pyplot as plt
import gym
import mcar_sarsa_semigrad_TileSutton as sarsa

#plt.plot(range(10))
#plt.show()

#return observation normalized between 0 and 1 using max and min obs from env
def normalize_obs(obs, env):
    obs_highs = env.observation_space.high
    obs_lows = env.observation_space.low
    return (obs - obs_lows) / (obs_highs - obs_lows)



class RBF:
    #initialize with centers and stdevs
    def __init__(self, centers, sigmas, n_actions):
        self.c = centers
        self.sig = sigmas
        self.num_actions = n_actions
        self.num_features = n_actions * len(centers)

    #get RBF activations for a given state
    def get_rbf_activations(self, state):
        #print(state)
        #print(self.c[0])
        assert(len(state) is len(self.c[0]))
        rbf_activations = []
        for i in range(len(self.c)):
            rbf_activations.append(np.exp(-np.dot(state - self.c[i], state - self.c[i]) / (2 * self.sig[i]**2)))
        return rbf_activations

    #get vectorized features for state and action_indx with zero activations for actions not taken
    def get_features(self, state, action_indx):
        assert(action_indx >= 0 and action_indx < self.num_actions)
        features = []
        for a in range(self.num_actions):
            if a is action_indx:
                features.extend(self.get_rbf_activations(state))
            else:
                features.extend(np.zeros(len(self.c)))
        return np.array(features)


class RbfReward:
    def __init__(self, rbf_fn, weights, env):
        self.rbf_fn = rbf_fn
        self.weights = weights
        self.env = env
    
    def get_reward(self, state):
        #NEED TO NORMALIZE STATE FIRST!
        state_normed = normalize_obs(state, self.env)
        activations = self.rbf_fn.map_features(state_normed)
        #print("activations", activations)
        #print(len(activations))
        #print("reward at {} is {}".format(state, np.dot(activations, self.weights)))
        reward = np.dot(activations, self.weights)
        
        #for a in activations:
        #    print(a)
        #    assert( a < 0 or abs(a) < 1E-9)
        return reward
        
class UniformReward:
    def __init__(self, reward):
        self.reward = reward
    def get_reward(self, state):
        return self.reward
        
class PositionRangeReward:
    def __init__(self, pos_lower, pos_upper, reward_in, reward_out):
        self.pos_lower = pos_lower
        self.pos_upper = pos_upper
        self.reward_in = reward_in
        self.reward_out = reward_out
        
    def get_reward(self, state):
        if self.pos_lower <= state[0] <= self.pos_upper:
            return self.reward_in
        else:
            return self.reward_out
            
class Constant_Feature_Map():
    def __init__(self):
        self.n_features = 1
    def map(self, state):
        return np.array([1.0])
        
class Rbf_Position_Feature_Map():
    def __init__(self, rbf):
        self.n_features = len(rbf.c)
        self.rbf = rbf
    def map_features(self, state): 
        #just map position
        #print(state)
        #print(state[0])
        return np.array(self.rbf.get_rbf_activations([state[0]]))
        
class Rbf_2D_Feature_Map():
    def __init__(self, rbf):
        self.n_features = len(rbf.c)
        self.rbf = rbf
    def map_features(self, state): 
        #just map position
        #print(state)
        #print(state[0])
        return np.array(self.rbf.get_rbf_activations(state))
        
class SignedRbf_2D_Feature_Map():
    """use this one with Syed and Schapire MWAL algorithm so that signs are right """
    def __init__(self, rbf, fsigns):
        self.n_features = len(rbf.c)
        self.rbf = rbf
        self.fsigns = fsigns
    def map_features(self, state): 
        #just map position
        #print(state)
        #print(state[0])
        #print("fsigns", self.fsigns)
        #print(np.array(self.rbf.get_rbf_activations(state)))
        return self.fsigns * np.array(self.rbf.get_rbf_activations(state))
    
def compute_feature_counts(feature_map, states, discount, env):
    fcounts = np.zeros(feature_map.n_features)
    for i in range(len(states)):
        #normalize state features
        f_normed = normalize_obs(states[i], env)
        fcounts += discount ** i * feature_map.map_features(f_normed)
    return fcounts

def generate_grid_centers(size):
    x = np.linspace(0.0, 1.0, size)
    y = np.linspace(0.0, 1.0, size)
    centers = [[x_i,y_j] for x_i in x for y_j in y]
    return np.array(centers)

def get_expected_feature_counts(num_rollouts, fMap, valueFunction, env, discount):
    
    features = []
    for i in range(num_rollouts):
        
        steps, states_visited = sarsa.rollout(env, valueFunction, render=False)
        
        #compute feature counts
        fcounts = compute_feature_counts(fMap, states_visited, discount, env)
        #print("steps = ", steps)
        #print("feature count = ", fcounts)
        
        features.append(fcounts)
    
    features = np.array(features)
    return np.mean(features, axis = 0)

# n_actions = 2
# c = np.array([[0,0],[0,0.5],[0,1],
#               [0.5,0],[0.5,0.5],[0.5,1],
#               [1.0,0],[1.0,0.5],[1.0,1]])
# sigma = 0.4 * np.ones(len(c))

# s = np.array([1.0,1.0])

# #calculate the RBF activation

# rbf = RBF(c,sigma,n_actions)
# print(rbf.get_features(s,0))
