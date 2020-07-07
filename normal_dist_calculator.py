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

plt.close("all")

class normDistGenerator:
    def __init__(self):
        tf.keras.backend.set_floatx("float64")
        self.mu = 0
        self.var =  1
    def generate_distribution(self,r_values, mu, var):
        self.mu = mu
        self.var = var
        pred = tf.dtypes.cast(1/tf.sqrt(2*np.pi*self.var), tf.float64)
        #print(pred)
        #r = r_values[0]
        #print(tf.exp(-(1/(2*self.var))*(r - self.mu)**2))
        dist = [pred*tf.exp(-(1/(2*var))*(r - mu)**2) for r in r_values]
        return dist
    
def check_scipy_consistency(error_significance =  1e-5,test_dists = 10):
    r = np.linspace(0,100,1000)
    mus = np.random.choice(r,test_dists)
    variances = [np.abs(np.random.normal()) for v in range(test_dists)]
    fits = True
    for i in range(test_dists):
        mu = mus[i]
        var = variances[i]
        dist = normDistGenerator().generate_distribution(r,mu,var)
        scipy_dist = scipy.stats.norm(loc = mu,scale = var).pdf(r)
        fits = fits and np.allclose(dist, scipy_dist,atol = error_significance)
        if not np.allclose(dist,scipy_dist,atol = error_significance):
            print("Distribution {} divergent.".format(i))
    return fits

def check_tensorflow_consistency(error_significance = 1e-5,test_dists = 10):
    r = np.linspace(0,100,1000)
    mus = [np.random.choice(r,1) for m in range(test_dists)]
    variances = [np.abs(np.random.normal()) for v in range(test_dists)]
    fits = True
    for i in range(test_dists):
        m = mus[i]
        v = variances[i]
        dist = normDistGenerator().generate_distribution(r,m,v)
        tf_dist = tfp.distributions.normal.Normal(m,v).prob(r)
        plt.figure()
        plt.plot(r,dist)
        plt.plot(r,tf_dist)
        plt.title("Mu: {}, var: {}".format(m,v))
        fits = fits and np.allclose(dist,tf_dist,atol = error_significance)
        if not np.allclose(dist,tf_dist,atol = error_significance):
            print("Distribution {} divergent.".format(i))
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
            print("Distribution {} divergent.".format(i))
    return fits


def generate_tensor_mixture_model(r_values, pi_values, mu_values, var_values):
    n,k = pi_values.shape
    n2,k2 = mu_values.shape
    n3,k3 = var_values.shape
    assert n == n2 and n2 == n3 and k == k2 and k2 == k3, "Mixture parameters dont have matching shapes, {},{},{}".format(pi_values.shape,mu_values.shape,var_values.shape)
    probability_array = [pi_values[:,kd]*tfp.distributions.normal.Normal(mu_values[:,kd],var_values[:,kd]).prob(r_values) for kd in range(k)]
    mixture = tf.add_n(probability_array)
    return mixture,probability_array