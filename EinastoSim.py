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
'''
Task:
        Generate M200, z, r, alpha randomly
        Generate Einasto Profile and derivative corresponding to this

'''

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
    
    #print("{}".format([H0,Ez,rhocrit,rhodelta,rdelta,rs,n]))
    
    #central density
    # ps=200.0*cdelta^3.0*((2.0*n)^(n))^3.0*alpha/(3.0*gamma(3.0*n)*(1.0-pgamma(q=(cdelta*(2.0*n)^n)^alpha, shape=3.0*n, lower = FALSE)))         #core density (dc)
    #print("Before central density")
    first_mult =200.0*cdelta**3
    second_mult  = ((2.0*n)**n)**3
    #print(second_mult)
    third_mult = alpha/(error_epsi + 3*math.gamma(3.0*n))
    fourth_mult = 1/(error_epsi + (1.0-stats.gamma.cdf((cdelta*(2.0*n)**n)**alpha,3.0*n))) 
    rho_cd  = first_mult*second_mult*third_mult*fourth_mult
    #print("After central density")
    rhomult = rho_cd*rhocrit
    return np.abs(rhomult*np.exp(-(r/rs*(2*n)**n)**alpha)) + error_epsi
def generate_set_einasto_profile(rmax, r_granularity, M200,z,alpha,r0,epsi,rho0,k,hz):
    profile = []
    for r in np.linspace(rmin_glob,rmax,r_granularity):
        pval = rho0*np.exp(-(r/(k*r0))**alpha)
        profile.append(pval)
    return profile

def generate_random_einasto_profile_maggie(rmax,r_granularity = r_gran_max):
    profile = []
    #normalized parameters
    Mdelta = 10**(13)*(1+100*random.random())*Msol/(10**14*Msol) 
    cdelta = np.abs(random.normalvariate(1,1)) #random.random()
    delta = 200#random.random()/2
    alpha = np.clip(random.normalvariate(0.5,0.5),lowest_alpha_val,1)
    zl = 0.6+(0.9*random.random())/0.9
    #print("{}".format([Mdelta, cdelta,delta,alpha,zl]))
    r = np.linspace(rmin_glob,rmax,int(r_granularity)) 
    profile = einasto_maggie(r,Mdelta,cdelta,delta,alpha,zl)
    return profile, [Mdelta, cdelta,alpha,zl],r
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
    for zc in np.linspace(9,z,r_granularity):
        if a > ap*z/r_granularity:
            a = max(a - ap*z/r_granularity,0)                    #take a measured step
            ap = np.sqrt(Om_m/(a**3+error_epsi)+Om_l)*a   #apply friedmann equation
        #print(a,ap)
    hz = h#np.sqrt(1/100*(ap/(a+error_epsi))**2)
    ellipsis = random.random()
    epsi = np.sqrt(1-ellipsis**2)
    
    rho0 = hz*M200/(4*np.pi*epsi*r0**3)
    k = 1
    for r in np.linspace(rmin_glob,rmax,r_granularity):
        pval = rho0*np.exp(-(r/(k*r0))**alpha)
        profile.append(pval)
    params = [M200,z,alpha,r0,epsi,rho0,k,hz]
    sel_range = np.linspace(rmin_glob,rmax,r_granularity)
    return profile,params,sel_range


def generate_n_profiles(N,rmax = rmax_glob ,r_granularity = r_gran_max):
    profiles = []
    profile_params = []
    assoc_range = []
    for i in range(N):        
        sample_profile,params,r = generate_random_einasto_profile(rmax_glob, r_granularity)
        profile_params.append(params)
        profiles.append(sample_profile)
        assoc_range.append(r)
    return profiles, profile_params, assoc_range

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
    sample_profile,parameters,r = generate_random_einasto_profile(rmax_glob)
    s_p_m, s_para_m,r = generate_n_random_einasto_profile_maggie(10000,rmax_glob)
    #print("Profile parameters: M = {} Ms, z = {},alpha = {}, r0 = {} Mpc, ellipsicity = {}, rho0 = {} Ms/MPc^3,k = {},hz = {}".format(M,z,alpha,r0,epsi,rho0,k,hz))
    plt.figure()
    plt.plot(np.linspace(0,rmax_glob,r_gran_max),sample_profile)
    k_profs,k_params,k_r = generate_n_profiles(1)
    plt.figure()
    plt.plot(r[0],s_p_m[0])
    