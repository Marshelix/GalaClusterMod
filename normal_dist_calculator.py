# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:52:45 2020

@author: Martin Sánner
normDistGenerator returns a list of tensorflow tensors with the value of the cdf of a normal distribution centered on mu and with variance var. 
The latest values are stored in the class instance
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import scipy as scp
import matplotlib.pyplot as plt
import logging
import scipy.stats
import EinastoSim
from datetime import datetime

import cProfile, pstats, io
import sys
now = datetime.now()
d_string = now.strftime("%d/%m/%Y, %H:%M:%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logfile_{}_{}.log".format(now.day,now.month)),
        logging.StreamHandler(sys.stdout)
    ]
)


class normDistGenerator:
    def __init__(self,mu,var):
        tf.keras.backend.set_floatx("float64")
        self.mu = mu
        self.var = var
    def generate_distribution(self,r_values, mu, var):
        self.mu = mu
        self.var = var
        pred = tf.dtypes.cast(1/tf.sqrt(2*np.pi*var), tf.float64)
        #print(pred)
        #r = r_values[0]
        #print(tf.exp(-(1/(2*self.var))*(r - self.mu)**2))
        dist = [pred*tf.exp(-(1/(2*var))*(r - mu)**2) for r in r_values]
        return dist
    
def check_scipy_consistency(error_significance =  1e-5,test_dists = 10):
    r = np.linspace(0,100,1000)
    mus = np.random.choice(r,test_dists)
    variances = [2.5*np.abs(np.random.normal()) for v in range(test_dists)]
    fits = True
    for i in range(test_dists):
        mu = mus[i]
        var = variances[i]
        dist = normDistGenerator(mu,var).generate_distribution(r,mu,var)
        scipy_dist = scipy.stats.norm(loc = mu,scale = var).pdf(r)
        fits = fits and np.allclose(dist, scipy_dist,atol = error_significance)
        plt.figure()
        plt.plot(r,dist, label = "Calculated distribution")
        plt.plot(r,scipy_dist, label = "Scipy distribution")
        plt.legend()
        plt.title("Scipy: {}::Mu: {}, var: {}".format(i,mu,var))
        if not np.allclose(dist,scipy_dist,atol = error_significance):
            print("Scipy Distribution {} divergent.".format(i))
    return fits

def check_tensorflow_consistency(error_significance = 1e-5,test_dists = 10):
    r = np.linspace(0,100,1000)
    mus = [np.random.choice(r,1) for m in range(test_dists)]
    variances = [2.5*np.abs(np.random.normal()) for v in range(test_dists)]
    fits = True
    for i in range(test_dists):
        m = mus[i]
        v = variances[i]
        dist = normDistGenerator(m,v).generate_distribution(r,m,v)
        tf_dist = tfp.distributions.normal.Normal(m,v).prob(r)
        plt.figure()
        plt.plot(r,dist,label = "Calculated distribution")
        plt.plot(r,tf_dist, label = "Tensorflow distribution")
        plt.legend()
        plt.title("Tensorflow: {}::Mu: {}, var: {}".format(i,m,v))
        fits = fits and np.allclose(dist,tf_dist,atol = error_significance)
        if not np.allclose(dist,tf_dist,atol = error_significance):
            print("Scipy Distribution {} divergent.".format(i))
    return fits

def check_tensorflow_scipy_consistency(error_significance = 1e-5,test_dists = 10):
    r = np.linspace(0,100,1000)
    mus = [np.random.normal() for m in range(test_dists)]
    variances = [np.abs(np.random.normal()) for v in range(test_dists)]
    fits = True
    for i in range(test_dists):
        m = mus[i]
        v = variances[i]
        sci_dist = scipy.stats.norm(loc = m,scale = v).pdf(r)
        tf_dist = tfp.distributions.normal.Normal(m,v).prob(r)
        plt.figure()
        plt.plot(r,sci_dist)
        plt.plot(r,tf_dist)
        plt.title("Mu: {}, var: {}".format(m,v))
        fits = fits and np.allclose(sci_dist,tf_dist,atol = error_significance)
        if not np.allclose(sci_dist,tf_dist,atol = error_significance):
            print("Tensorflow Distribution {} divergent.".format(i))
    return fits



def profile(fnc):
    
    """A decorator that uses cProfile to profile a function"""
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        logging.info(s.getvalue())
        return retval

    return inner

#@profile
def generate_tensor_mixture_model(r_values, pi_values, mu_values, var_values):
    n,k = pi_values.shape
    n2,k2 = mu_values.shape
    n3,k3 = var_values.shape
    assert n == n2 and n2 == n3 and k == k2 and k2 == k3, "Mixture parameters dont have matching shapes, {},{},{}".format(pi_values.shape,mu_values.shape,var_values.shape)
    mixtures = []
    probabilities = []
    for mix_index in range(n):
        probability_array = []
        for kd in range(k):
            probability_array.append(pi_values[mix_index,kd]*tfp.distributions.normal.Normal(mu_values[mix_index,kd],var_values[mix_index,kd]).prob(r_values[mix_index]))
        mixture = tf.add_n(probability_array)
        mixtures.append(mixture)
        probabilities.append(probability_array)
    
    
    return tf.stack(mixtures),probabilities
if __name__ == "__main__":
    plt.close("all")
    #generator_fits_scipy = check_scipy_consistency(test_dists = 5)
    #generator_fits_tensorflow = check_tensorflow_consistency(test_dists = 5)
    #sample a mixture model
    r = np.linspace(0,20,1000)
    pi_ = np.asarray([[abs(np.random.normal()) for i in range(4)]])
    pi_ = np.asarray([p/sum(p) for p in pi_])
    mu_ = np.asarray([np.random.choice(r,4)])
    var_ = np.asarray([[2.5*abs(np.random.normal()) for i in range(4)]])
    mixture, probs = generate_tensor_mixture_model([r],pi_,mu_,var_)
    
    plt.figure()
    for kd in range(4):
        plt.plot(r,probs[0][kd],label = "Density {}: pi = {}; mu = {}; var = {}".format(kd,pi_[0][kd],mu_[0][kd],var_[0][kd]))
    plt.plot(r,mixture[0],label = "Mixture")
    plt.legend()
    
    
    profiles, parameters, rs = EinastoSim.generate_n_random_einasto_profile_maggie(1)
    r = rs[0]
    profile = np.log(profiles[0])
    profile = (profile-np.min(profile))/(np.max(profile)-np.min(profile))
    plt.figure()
    plt.plot(r,profile, label = "Einasto Profile")
    mus = np.asarray([[0.4,1.25,2.0,2.5]])
    vari = np.asarray([[0.225,0.3,0.029,0.22]])
    pis = np.asarray([[0.5,0.4,0.025,0.075]])
    mixture, probs = generate_tensor_mixture_model(rs, pis,mus,vari)
    mixture = mixture[0]
    plt.plot(r,mixture, label = "Mixture")
    plt.plot(r,probs[0][0],label = "Density {}: pi = {}; mu = {}; var = {}".format(0,pis[0][0],mus[0][0],vari[0][0]))

    plt.plot(r,probs[0][1],label = "Density {}: pi = {}; mu = {}; var = {}".format(1,pis[0][1],mus[0][1],vari[0][1]))

    plt.plot(r,probs[0][2],label = "Density {}: pi = {}; mu = {}; var = {}".format(2,pis[0][2],mus[0][2],vari[0][2]))

    plt.plot(r,probs[0][3],label = "Density {}: pi = {}; mu = {}; var = {}".format(3,pis[0][3],mus[0][3],vari[0][3]))
    plt.legend()    
    
    
    