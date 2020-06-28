# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:36:58 2020

@author: marti
"""

import EinastoSim
import numpy as np

class reparameterizer:
    def __init__(self, parameters):
        '''
        Module that takes a set of profile parameters, generates their statistics and normalizes it
        Alternatively can reparameterize a set of normalized parameters of the same family
        '''
        self.parameters = parameters
        self.num_param_sets = len(parameters)
        self.num_params = len(parameters[0])
        self.means = np.asarray([0 for i in range(self.num_params)])
        self.var = np.asarray([1 for i in range(self.num_params)])
        assert len(parameters) == 5
    def calculate_parameterization(self, parameters = None):
        if parameters is not None:
            self.parameters = parameters
            self.num_params = len(parameters[0])
            self.num_param_sets = len(parameters)
            
            self.means = np.asarray([0 for i in range(self.num_params)])
            self.var = np.asarray([1 for i in range(self.num_params)])
        for params in self.parameters:
            self.means += params
        self.means /= self.num_param_sets
        
        for params in self.parameters:
            self.var += (params - self.means)**2
        self.var /= self.num_param_sets
        
        out = [[0 for j in range(self.num_params)] for k in range(self.num_param_sets)]
        for k in range(self.num_param_sets):
            for j in range(self.num_params):
                out[k,j] = (parameters[k,j]-self.means[j])/np.sqrt(self.var[j])
        return out
    def calculate_deparameterization(self, parameters):
        out = [[0 for j in range(self.num_params)] for k in range(self.num_param_sets)]
        for k in range(self.num_param_sets):
            for j in range(self.num_params):
                out[k,k] = (parameters[k,j]*np.sqrt(self.var[j]))+self.means[j]
        return out
    