# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:53:31 2020

@author: Martin Sanner
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import random

'''
Task:
        Generate M200, z, r, alpha randomly
        Generate Einasto Profile and derivative corresponding to this

'''

Msol = 1#1.9984*(10)**(30) #kg
Lyr2MPc = 3.066013938E-7 #

h = 0.7 #cosmological parameter
rmax_glob = 52850*Lyr2MPc
rmin_glob = 0.05*rmax_glob
r_gran_max = 100

a0g = 1
a0pg = np.sqrt(100*h)

Om_m = 0.3
Om_l = 0.7

error_epsi  = 1e-10

def generate_set_einasto_profile(rmax, r_granularity, M200,z,alpha,r0,epsi,rho0,k,hz):
    profile = []
    for r in np.linspace(rmin_glob,rmax,r_granularity):
        pval = rho0*np.exp(-(r/(k*r0))**alpha)
        profile.append(pval)
    return profile


def generate_random_einasto_profile(rmax,r_granularity = r_gran_max):
    '''
        Randomly select M, z, r, alpha from appropriate range
        Figure out constants based on original function definition
        
    '''
    profile = []
    M200 = 10**(13)*(1+100*random.random())*Msol #Mass in range 10^13 - 10^15 Msol
    z = 0.6+(0.9*random.random())
    alpha = random.random() # ~1/N -> between 0,1
    r0 = rmax*random.random()
    '''
    Alpha between 0,1 take random here
    
    Take k = 1, assume h = 0.7
    '''
    
    '''
        Alternate approach here:
            assume a0 = 1, a0' = sqrt(100*h), integrate from 0 to z to find h(z) = 1/100 *(az'/a)^2
        Would be more accurate...
    '''
    a = a0g
    ap = a0pg
    for zc in np.linspace(rmin_glob,z,r_granularity):
        if a > ap*z/r_granularity:
            a = max(a - ap*z/r_granularity,0)                    #take a measured step
            ap = np.sqrt(Om_m/(a**3+error_epsi)+Om_l)*a   #apply friedmann equation
        #print(a,ap)
    hz = h#np.sqrt(1/100*(ap/(a+error_epsi))**2)
    ellipsis = random.random()
    epsi = np.sqrt(1-ellipsis**2)
    
    rho0 = hz*M200/(4*np.pi*epsi*r0**3)
    k = 1
    for r in np.linspace(0,rmax,r_granularity):
        pval = rho0*np.exp(-(r/(k*r0))**alpha)
        profile.append(pval)
    
    return profile,M200,z,alpha,r0,epsi,rho0,k,hz


def generate_n_profiles(N,rmax = rmax_glob ,r_granularity = r_gran_max):
    profiles = []
    profile_params = []
    for i in range(N):        
        sample_profile,M,z,alpha,r0,epsi,rho0,k,hz = generate_random_einasto_profile(rmax_glob, r_granularity)
        profile_params.append([M,z,alpha,r0,epsi,rho0,k,hz])
        profiles.append(sample_profile)
    return profiles, profile_params

def print_params(profile_parameters):
    M = profile_parameters[0]
    z = profile_parameters[1]
    alpha = profile_parameters[2]
    r0 = profile_parameters[3]
    epsi = profile_parameters[4]
    rho0 = profile_parameters[5]
    k = profile_parameters[6]
    hz = profile_parameters[7]
    param_string = "Profile parameters: \n \t M = {} Ms,\n \t z = {},\n \t alpha = {},\n \t r0 = {} MPc,\n \t ellipsicity = {},\n \t rho0 = {} Ms/MPc^3,\n \t k = {},\n \t hz = {}".format(M,z,alpha,r0,epsi,rho0,k,hz)
    print(param_string)
    return param_string

if __name__ == "__main__":
    sample_profile,M,z,alpha,r0,epsi,rho0,k,hz = generate_random_einasto_profile(rmax_glob)
    print("Profile parameters: M = {} Ms, z = {},alpha = {}, r0 = {} Mpc, ellipsicity = {}, rho0 = {} Ms/MPc^3,k = {},hz = {}".format(M,z,alpha,r0,epsi,rho0,k,hz))
    plt.figure()
    plt.plot(np.linspace(0,rmax_glob,r_gran_max),sample_profile)
    k_profs,k_params = generate_n_profiles(1000)
