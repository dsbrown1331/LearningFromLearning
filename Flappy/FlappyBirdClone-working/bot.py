import pickle
import random
import time
from matplotlib import pyplot as plt
import numpy as np



class Bot:

    def __init__(self):
        self.alpha = 0.2 # Constant Learning rate
        self.epsilonFactor = 1.0 # The reduction factor for epsilon after every sampleRate
        self.discount = .95 # Discount Rate
        self.epsilon_start = 0.0
        self.sampleRate = 500 # The dynamic variables change after every sampleRate and Q-Table + scores are stored in files
        self.fileName = 'Q-table.pkl' # Q-Table file name
        self.scoreFile = 'scores.pkl' # Scores file name
        
        # Try to open a pretrained model
        # If and only if both files exist, we use the pretrained model
        try:
            with open(self.fileName, 'rb') as f:
                self.Q = pickle.load(f)

            with open(self.scoreFile, 'rb') as f:
                self.counter = len(pickle.load(f))
                self.epsilon = self.epsilon_start * pow(self.epsilonFactor,self.counter/self.sampleRate)
        # Else start fresh training
        except FileNotFoundError:
            print('New Learning')
            self.Q = {}
            self.counter = 0
            self.epsilon = self.epsilon_start

        self.pipeReward = -15 # Reward for crashing into an upper pipe
        self.sumScore = 0 # Sum of scores for a sample
        self.scoreList = [] # Scores List for a sample
        #plt.ion()
        
    def set_eps(self, eps):
        self.epsilon = eps

    def maxQ(self, state, eps_greedy = True):
        # Chooses the higher Q-value with a probability of (1-epsilon)
        max_val = max(self.Q[state])
        max_act = self.Q[state].index(max_val)

        if eps_greedy and random.random()<self.epsilon:
            max_act = int(not max_act)
            max_val = self.Q[state][max_act]
        return max_act, max_val

    def appendState(self, state):
        # Add state to Q table if not already added
        if state not in self.Q:
            self.Q[state]=[]
            self.Q[state].append(0)
            self.Q[state].append(0)

# exp is a tuple(old_state, best_action, reward, new_state)
    def updateQ(self, exp, score):
        self.sample(score) # Update the sample statistics for each game played
        
        exp.reverse() # Reverse the list of experiences

        # If crashed with upper pipe then tax the last jump that caused it
        #if upCrash:
        #    for i, xp in enumerate(exp):
        #        if xp[1]==1:
        #            temp = list(exp[i])
        #            temp[2] = self.pipeReward
        #            exp[i] = tuple(temp)
        #            break
        # For each entry calculate and update Q value
        for xp in exp:
            s = xp[0]
            a = xp[1]
            r = xp[2]
            _, fut_r = self.maxQ(xp[3], eps_greedy=False)
            self.Q[s][a] *= 1 - self.alpha
            self.Q[s][a] += self.alpha * (r + self.discount*fut_r )

    def sample(self, score):
        # Updates the sample statistics for each game played
        self.counter += 1
        self.sumScore+= score
        self.scoreList.append(score)

        # If sampleRate is reached then save the statistics to a file and reset sample statistics
        if self.counter%self.sampleRate == 0:
            #plt.plot(self.counter, self.sumScore/self.sampleRate, 'ro')
            print(self.counter, np.mean(self.scoreList), np.std(self.scoreList),
                             np.min(self.scoreList), np.max(self.scoreList))
            self.saveQ()
            self.dumpScores()
            self.epsilon *= self.epsilonFactor
            print(self.epsilon)
            self.sumScore = 0

    def dumpScores(self):
        # Save the scores list to a file and reset it for next sample
        try:
            with open(self.scoreFile, 'rb') as f:
                temp = pickle.load(f)
            temp+=(self.scoreList)
        except FileNotFoundError:
            temp = self.scoreList

        with open(self.scoreFile,'wb') as f:
            pickle.dump(temp,f)

        self.scoreList=[] # Resetting the scores list for next batch of sample

    def saveQ(self):
        # Save the Q-Table to a file
        with open(self.fileName,'wb') as f:
            pickle.dump(self.Q, f)
            
            
if __name__ == "__main__":
#    fileName = 'Q-table.pkl' # Q-Table file name
#    with open(fileName, 'rb') as f:
#        Q = pickle.load(f)
#    states = []
#    for k in Q:
#        states.append(list(k))
#    states = np.array(states)
#    print(np.min(states, axis=0))
#    print(np.max(states, axis = 0))
    # Initialise bot
    bot = Bot()
       

