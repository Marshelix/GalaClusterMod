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
        
        
        ISSUE: This doesnt make sense, always only normalizes locally, never globally, doesnt track stats correctly -> normalize globally instead
        
        
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

#estimate "normalization" parameter as reduction of most plots
normalization_param_einasto = 66


def normalize_profiles(logged_profiles,normalization_param = normalization_param_einasto):
    '''
    Instead of minmax normalization, apply a  standard normalization
    '''
    return np.asarray([p/normalization_param for p in logged_profiles])

def renormalize_profiles(renorm_log_profiles,normalization_param = normalization_param_einasto):
    return np.asarray([p*normalization_param for p in renorm_log_profiles])


if __name__ == "__main__":
    plt.close("all")
    profiles, parameters, rs = EinastoSim.generate_n_random_einasto_profile_maggie(15)
    r = rs[0]
    log_prof = [np.log(p) for p in profiles]
    repa_profs = normalize_profiles(log_prof)
    for i in range(len(rs)):
        r = rs[i]
        log_profile = repa_profs[i]
        parameter = parameters[i]
        plt.figure()
        plt.title(EinastoSim.print_params_maggie(parameter).replace("\t",""))
        plt.xlabel("R [Mpc]")
        plt.ylabel("log({}(r))".format(u"\u03C1"))
        plt.plot(r,log_profile)
    
    