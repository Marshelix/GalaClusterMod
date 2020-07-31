# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:53:31 2020

@author: Martin Sanner
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
import math
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp


Msol = 1#1.9984*(10)**(30) #kg
Lyr2MPc = 3.066013938E-7 #
km2Lyr = 9.461e+12
km2Mpc = km2Lyr*Lyr2MPc
h = 0.7 #cosmological parameter
rmax_glob = 3#52850*Lyr2MPc
Mpc = 3.087*10**(16)
rmin_glob = 1.5e-1
r_gran_max = int(100)

a0g = 1
a0pg = np.sqrt(100*h)

Om_m = 0.3
Om_l = 0.7

error_epsi  = 1e-10
G = 6.67430*10**(-11)   #m^3 kg^-1 s^-2
Gnew = G*km2Mpc**3/Msol


lowest_alpha_val = 1e-1
    


def einasto_maggie(r,Mdelta,cdelta,delta,alpha,zl,h = h,Om = Om_m, Ol = Om_l):
    '''
        Function based on code provided by Dr. Maggie Lieu. 
        Ported to python to compare viability of own code.
        
        
        r: radius in Mpc
        Mdelta: Mass at overdensity delta
        cdelta: Concentration at overdensity position r_delta
        alpha: 1/N, parameter for profile
        zl: Lense redshift
        h: hubble parameter
        Om: mass density
        Ol: Dark energey density
        Assume Ok = 0
        
        
    '''
    H0 = 100*h
    Ez = np.sqrt(Om*(1+zl)**3+Ol)   #evolution parameter a'/a
    rhocrit = (3*Ez**2)/(8*np.pi*Gnew)
    rhodelta = delta*rhocrit
    rdelta = ((3*Mdelta)/(4*np.pi*rhodelta))**(1.0/3.0)
    rs = rdelta/cdelta
    n = 1.0/alpha
    #central density
    # ps=200.0*cdelta^3.0*((2.0*n)^(n))^3.0*alpha/(3.0*gamma(3.0*n)*(1.0-pgamma(q=(cdelta*(2.0*n)^n)^alpha, shape=3.0*n, lower = FALSE)))         #core density (dc)
    first_mult =200.0*cdelta**3
    second_mult  = ((2.0*n)**n)**3
    third_mult = alpha/(error_epsi + 3*math.gamma(3.0*n))
    fourth_mult = 1/(error_epsi + (1.0-stats.gamma.cdf((cdelta*(2.0*n)**n)**alpha,3.0*n))) 
    rho_cd  = first_mult*second_mult*third_mult*fourth_mult
    
    rhomult = rho_cd*rhocrit
    return np.abs(rhomult*np.exp(-(r/rs*(2*n)**n)**alpha)) + error_epsi

def generate_random_einasto_profile_maggie(rmax,r_granularity = r_gran_max):
    profile = []
    #normalized parameters
    Mdelta = 10**(13)*(1+100*random.random())*Msol 
    cdelta = np.abs(random.normalvariate(1,1)) #random.random()
    delta = 200#random.random()/2
    alpha = np.clip(random.normalvariate(0.5,0.5),lowest_alpha_val,1)
    zl = 0.6+(0.9*random.random())
    #print("{}".format([Mdelta, cdelta,delta,alpha,zl]))
    r = np.linspace(rmin_glob,rmax,int(r_granularity)) 
    profile = einasto_maggie(r,Mdelta,cdelta,delta,alpha,zl)
    return profile, [Mdelta/(10**14*Msol), cdelta,alpha,zl/0.9],r
def generate_n_random_einasto_profile_maggie(num_profiles,rmax = rmax_glob,r_granularity = r_gran_max):
    profiles = []
    profile_params = []
    radii = []
    for i in range(num_profiles):
        profile,params,r = generate_random_einasto_profile_maggie(rmax,r_granularity)
        #infinities = np.sum([int(np.isinf(p)) for p in profile])
        #nans = np.sum([int(np.isnan(p)) for p in profile])
        #zeros = np.sum([p == 0 for p in profile]) 
        #print("Profile {} generated, length {}, Infinities: {}, NaNs: {},zeros: {}".format(i,profile.size,infinities,nans,zeros))
        profiles.append(profile)
        profile_params.append(params)
        radii.append(r)
    return profiles, profile_params,radii

def print_params_maggie(parameters):
    assert len(parameters) == 4, "Wrong amount of parameters inserted"
    M = parameters[0]*(10**14*Msol) 
    cdelta = parameters[1]
    delta = 200#parameters[2]
    alpha = parameters[2]
    zl = parameters[3]*0.9
    profile_string = "Parameters: \n \t M = {} Ms \n\t cdelta =  {} \n \t delta = {} \n \t alpha = {} \n \t z = {}".format(M,cdelta,delta,alpha,zl)
    print(profile_string)
    return profile_string    

