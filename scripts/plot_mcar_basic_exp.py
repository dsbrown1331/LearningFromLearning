# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 16:08:41 2018

@author: dsbrown
"""
import numpy as np
import matplotlib.pyplot as plt
demonstrations = [5,10,20,30,40,50]
num_reps = 20
max_demos = 50
birl_conf = 1.0

#plot demos
demo_returns = np.zeros((num_reps,max_demos))
for seed in range(num_reps):
    reader = open("data/mcar_demo_seed" + str(seed) + "_demos50")
    cnt = 0
    for line in reader:
        demo_returns[seed, cnt] = float(line)
        cnt += 1
demo_returns = -np.mean(demo_returns[:,4:], axis = 0)

#plot MWAL
mwal = []
for demos in demonstrations:
    mwal_returns = []
    for seed in range(num_reps):
        reader = open("data/mcar_mwal_seed" + str(seed) + "_demos" + str(demos))
        for line in reader:
            mwal_returns.append(float(line)) 
    mwal.append(-np.mean(mwal_returns))

   
#plot MaxEnt
maxent = []
for demos in demonstrations:
    maxent_returns = []
    for seed in range(num_reps):
        reader = open("data/mcar_maxent_seed" + str(seed) + "_demos" + str(demos))
        for line in reader:
            maxent_returns.append(float(line)) 
    maxent.append(-np.mean(maxent_returns))
    
#plot BIRL
birl = []
for demos in demonstrations:
    birl_returns = []
    for seed in range(num_reps):
        reader = open("data/mcar_birl_conf" + str(birl_conf) + "_seed" + str(seed) + "_demos" + str(demos))
        for line in reader:
            birl_returns.append(float(line)) 
    birl.append(-np.mean(birl_returns))

  

plt.plot(demonstrations, mwal, label="GT-IRLfL", lw=3)
plt.plot(demonstrations, birl, label="BIRL", lw=3)
plt.plot(demonstrations, maxent, label="MaxEnt", lw=3)
plt.plot(range(5,max_demos + 1), demo_returns, label="demonstrations", lw=3)  
plt.xlabel("Number of demonstrations",fontsize=18)
plt.ylabel("Average steps to goal",fontsize=18)
plt.xticks(demonstrations,fontsize=18) 
#plt.axis([0,21,-1100,0])
plt.yticks(fontsize=18) 
plt.legend(loc='best', fontsize=18)
plt.tight_layout()

plt.show()
