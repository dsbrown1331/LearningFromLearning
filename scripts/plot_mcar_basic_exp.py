# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 16:08:41 2018

@author: dsbrown
"""
import numpy as np

demo_data = False
learned_data = False
demos = []
learned = []
reader = open("data/mountain_car_test_data.txt")
for line in reader:
    if line.startswith("#demos"):
        demo_data = True
        learned_data = False
    elif line.startswith("#learned"):
        demo_data = False
        learned_data = True
    elif demo_data:
        demos.append(float(line))
    elif learned_data:
        learned.append(float(line))
        
import matplotlib.pyplot as plt
plt.plot(range(1,len(demos)+1),demos, label="Demonstrator", lw=2)
plt.plot([1, len(demos)], np.mean(learned) * np.ones(2), '--', label="LfLD", lw=2)
plt.xlabel("Iteration",fontsize=18)
plt.ylabel("Return",fontsize=18)
plt.xticks([1,5,10,15,20],fontsize=18) 
plt.axis([0,21,-1100,0])
plt.yticks(fontsize=18) 
plt.legend(loc='best', fontsize=19)
plt.tight_layout()

plt.show()
