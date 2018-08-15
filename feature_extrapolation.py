# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:11:45 2018

@author: dsbrown
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


class FeatureSignExtractor:
    def __init__(self, fcounts, flabels):
        self.fcounts = np.array(fcounts)
        self.flabels = flabels
        
    def estimate_signs(self):
        fsigns = []
        for i in range(len(self.fcounts[0])):
            #fit linear regression line
            x_vals = np.arange(len(self.fcounts)).reshape(-1,1)
            regr = linear_model.LinearRegression()
            regr.fit(x_vals, self.fcounts[:,i])
            print('Slope for feature {}: {}'.format(self.flabels[i], regr.coef_))
            fsigns.append(regr.coef_[0])
        return fsigns
   