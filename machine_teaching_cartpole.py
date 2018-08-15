import pickle
from _policies import BinaryActionLinearPolicy
from cart_pole_cem_agent import run_rollout
import gym
import time
import numpy as np
from lp_redundancy_removal import remove_redundancies
from rbf import RBF

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

#TODO: get this to work for stochastic domains by running monte carlo esimtations of feature counts?
def get_feature_half_planes(state_feature_map, env, start_state, agent, horizon, min_margin = 0, discount = 1.0, threshold = 0.0001):
    #run rollouts counting up the features for each possible action
    opt_action = agent.act(start_state)
    half_plane_normals = set() #store the constraints in a set
    #get the feature counts from taking the optimal action
    #print("taking opt", opt_action)
    states_visited, opt_reward = run_rollout(env, start_state, opt_action, agent, render=False)
    #print(states_visited)
    #compute feature counts
    opt_fcounts = compute_feature_counts(state_feature_map, states_visited, discount)
    #print(opt_fcounts)
    best_margin = 0
    #take other actions
    for init_action in range(env.action_space.n):
        if init_action != opt_action:
            #print("init action", init_action)
            states_visited, cum_reward = run_rollout(env, start_state, init_action, agent, render=False)
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
        
    
        
def solve_set_cover(nr_constraints, demo_database):
    print("=== Solving set cover ===")
    opt_start_states = [] #store machine teaching solution
    #convert nr_constraints into set of constraints for easy queries and removal
    nr_set = set([tuple(c) for c in nr_constraints])
    #print("to cover", nr_set)
    while(len(nr_set) > 0):
        max_covered = 0
        best_start_state = None
        constraints_covered = set()
        database_to_remove = []
        for start_state in demo_database:
            #print("--start state", start_state)
            #get constraints for trajectory
            new_constraints = demo_database[start_state]  #returns set of tuples
            #print("constraints", new_constraints)
            #check how many constraints it covers
            covered_cnt = 0
            for c_new in new_constraints:
                if c_new in nr_set:
                    covered_cnt += 1
            #print("num covered", covered_cnt)
            if covered_cnt > max_covered:
                #print("updating max_covered to", covered_cnt)
                max_covered = covered_cnt
                constraints_covered = new_constraints
                best_start_state = start_state
            elif covered_cnt == 0:
                #remember to remove this demo option since won't ever be used
                database_to_remove.append(start_state)
        #remove start states that are not useful any more (won't cover any uncovered constraints)
        for start_remove in database_to_remove:
            demo_database.pop(start_remove)
        #print("data base after removal", demo_database)
        #print("**")
        #print("best start state", best_start_state)
        #print("max covered", max_covered)
        #print("constraints covered", constraints_covered)
        #add best start_state to solution
        opt_start_states.append(best_start_state)
        #remove covered constraints from nr_set
        for c_added in constraints_covered:
            if c_added in nr_set:
                nr_set.remove(c_added)
        
        #print("updated uncovered constraints", nr_set)
    return opt_start_states
    
        
#rewards = []
if __name__ == "__main__":
    reps = 20
    alpha = 0.5
    EPSILON = 0
    num_samples = 500

    min_margin = 1
    discount = 1.0
    threshold = 0.001
    horizon = 1
    env = gym.make('CartPoleUniform-v0')
    # use optimistic initial value, so it's ok to set epsilon to 0
    EPSILON = 0
    #get optimal policy that has been previously learned
    outdir = '/tmp/cem-agent-results/'
    env = gym.make("CartPoleUniform-v0")

    #env.seed(0)
    #np.random.seed(0)
    agent = pickle.load( open( outdir + "/agent.pkl", "rb" ) )
        
    #pick feature map
    fMap = Constant_Feature_Map()
    #rbf = RBF(np.array([[-1.2], [-0.3], [0.6]]), 0.7*np.ones(3), env.action_space.n)
    #fMap = Rbf_Position_Feature_Map(rbf)

    #figure out the feasible region

    #sample starting states
    all_constraints = set()
    demo_database = {}
    #TODO could parallelize this!
    for i in range(num_samples):
        if(i % 100 == 0):
           print(i)
        start_state = env.reset() #picks range that you can recover from
        half_planes = get_feature_half_planes(fMap, env, start_state, agent, horizon, min_margin, discount, threshold) #return dictionary s:constraints
        #print(half_planes)
        #check if there are any half-planes and add them to the constraint set and the demo_database
        if len(half_planes) > 0:
            for p in half_planes:
                all_constraints.add(p)
            demo_database[tuple(start_state)] = half_planes #associated half_planes with start_state of demo
            

    print("all constraints", all_constraints)
    print("demo database", demo_database)
    nr_constraints = remove_redundancies(np.array(list(all_constraints)))
    pickle_filename = "pickle filename = "+ 'constraints_' + str(time.time()) + '.pickle'
    print(pickle_filename)
    with open(pickle_filename, 'wb') as f:
        pickle.dump(nr_constraints, f, pickle.HIGHEST_PROTOCOL)
    print("non-redundant constraints")
    print(nr_constraints)
    opt_start_states = solve_set_cover(nr_constraints, demo_database)
    print("opt demos", opt_start_states)
    print("number of demos = ", len(opt_start_states))
    
