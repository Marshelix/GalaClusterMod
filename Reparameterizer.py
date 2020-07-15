# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:36:58 2020

@author: Martin Sanner
"""
import numpy as np
import EinastoSim
import matplotlib.pyplot as plt

class reparameterizer:
    def __init__(self, profiles):
        '''
        Module that takes a set of data profile, generates their statistics and normalizes it to a 0-1 range
        Alternatively can reparameterize a set of normalized parameters of the same family
        '''
        self.profiles = profiles
        self.num_param_sets = len(self.profiles)
        #minmax parameterization
        self.minmax = [[min(p),max(p)] for p in self.profiles]
    def calculate_parameterization(self, profiles = None):
        if profiles != None:
            self.profiles = profiles
            self.num_param_sets = len(self.profiles)
        self.minmax = [[min(p),max(p)] for p in self.profiles]
        return np.asarray([(p-min(p))/(max(p)-min(p)+1e-9) for p in self.profiles])
    def calculate_deparameterization(self, new_profiles):
        assert len(new_profiles) == self.num_param_sets,"This reparameterizer expects {} profiles.".format(self.num_param_sets)
        return np.asarray([new_profiles[i]*(self.minmax[i][1]-self.minmax[i][0]+1e-9)+self.minmax[i][0] for i in range(self.num_param_sets)])


normalization_param = -11

def normalize_profiles(logged_profiles):
    '''
    Instead of minmax normalization, apply a  standard normalization
    '''
    return np.asarray([p/normalization_param for p in logged_profiles])

def renormalize_profiles(renorm_log_profiles):
    return np.asarray([p*normalization_param for p in renorm_log_profiles])

if __name__ == "__main__":
    plt.close("all")
    profiles, parameters, rs = EinastoSim.generate_n_random_einasto_profile_maggie(1)
    r = rs[0]
    profile = profiles[0]
    log_prof = [np.log(p) for p in profiles]
    repa = reparameterizer(log_prof)
    repa_profs = repa.calculate_parameterization()
    
    
    