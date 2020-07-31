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
from datetime import datetime

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

onedivsqrttwoPi = 1/np.sqrt(2*np.pi)

class normalDistCalc:
    def __init__(self,mu = 0.0,var = 1.0):
        '''

        Parameters
        ----------
        mu : float
            Mean value of distribution
        var : Variance > 0
            applies absolute value function if necessary. If 0, reset to 1

        Returns
        -------
        None.

        '''
        self.mu = mu
        self.var = tf.abs(var) if var < 0 else var
        if var == 0:
            self.var = 1
         
    def generate_distribution(self,x):
        premult = tf.cast(onedivsqrttwoPi*(1/tf.sqrt(self.var)),dtype = tf.float64)
        diff = tf.math.subtract(x,self.mu)
        diffsqr = diff**2
        expon = -diffsqr/(2*self.var)
        return premult*tf.exp(expon)

class normalMixtureCalculator:
    def __init__(self):
        self.kg = 1
        self.pis = np.asarray([1/self.kg for i in range(self.kg)]) #uniform
        self.mus = np.asarray([0 for i in range(self.kg)])
        self.var = np.asarray([1 for i in range(self.kg)])
    def calculate_mixture_distributions(self,xs,pis,mus,var):
        n,k = np.asarray(pis).shape
        n2,k2 = np.asarray(mus).shape
        n3,k3 = np.asarray(var).shape
        assert n == n2 and n2 == n3 and k == k2 and k2 == k3, "Mixture parameters dont have matching shapes, {},{},{}".format(pis.shape,mus.shape,var.shape)
        self.kg = k
        mixture_sources = []
        mixtures = []
        for mix_index in range(n):
            x = xs[n % len(xs)]
            probability_array = []
            for kd in range(self.kg):
                probability_array.append(pis[mix_index][kd]*normalDistCalc(mus[mix_index][kd], var[mix_index][kd]).generate_distribution(x))
            mixture = tf.add_n(probability_array)
            mixtures.append(mixture)
            mixture_sources.append(np.asarray(probability_array))
        return np.asarray(mixtures), mixture_sources
        
def generate_vector_random_gauss_mixture(r_values, kg):
    n = len(r_values)
    mixtures = []
    mixture_pdfs = []
    parameters = []
    for mix_index in range(n):
        r = r_values[mix_index]
        mus = [np.random.uniform(-1.0,1.0) for k in range(kg)]
        var = [np.random.uniform(0.0,1.0) for k in range(kg)]
        pis = [np.random.rand() for k in range(kg)]
        spi = np.sum(pis)
        pis = [cp/spi for cp in pis] # sum(pi) = 1
        gm = tfp.distributions.MixtureSameFamily(mixture_distribution = tfp.distributions.Categorical(probs = pis),
                                                  components_distribution = tfp.distributions.Normal(loc = mus,scale =var))
        mixtures.append(gm)
        mixture_pdfs.append(gm.prob(r))
        parameters.append(np.asarray(pis+mus+var))
    return np.asarray(mixture_pdfs),np.asarray(parameters),mixtures

def generate_vector_gauss_mixture(r_values, pi_values, mu_values, var_values):
    n,k = np.asarray(pi_values).shape
    n2,k2 = np.asarray(mu_values).shape
    n3,k3 = np.asarray(var_values).shape
    assert n == n2 and n2 == n3 and k == k2 and k2 == k3, "Mixture parameters dont have matching shapes, {},{},{}".format(pi_values.shape,mu_values.shape,var_values.shape)
    mixtures = []
    mixture_pdfs = []
    for mix_index in range(n):
        gm = tfp.distributions.MixtureSameFamily(mixture_distribution = tfp.distributions.Categorical(probs = pi_values[mix_index]),components_distribution = tfp.distributions.Normal(loc = mu_values[mix_index],scale = var_values[mix_index])) 
        mixtures.append(gm)
        mixture_pdfs.append(gm.prob(r_values[mix_index]))
    return tf.stack(mixture_pdfs),mixtures



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
    #num_profiles = 10#args.num_profile
    #kg = 4
    plt.close("all")
    r = np.linspace(-10,10,1001)
    mu = 0
    var = 0.5
    calcul = normalDistCalc(mu,var)
    sample_dist = calcul.generate_distribution(r)
    other_dist = tfp.distributions.Normal(loc = mu, scale = tf.sqrt(var))
    diff = tf.cast(sample_dist,tf.float64) - tf.cast(other_dist.prob(r),tf.float64)
    plt.figure()
    plt.plot(r,sample_dist,label = "Calculator distribution")
    plt.plot(r,other_dist.prob(r),label = "TFP dist")
    plt.legend()
    plt.figure()
    plt.plot(r,diff)
    num_gen = 5
    kg = 4
    rs = [r for i in range(num_gen)]
    pis = [tf.cast(tf.nn.softmax([np.random.rand() for j in range(kg)]),dtype = tf.float64) for i in range(num_gen)]
    mus = [[np.random.choice(r,replace = False) for j in range(kg)] for i in range(num_gen)]
    var = [[np.random.rand() for j in range(kg)] for i in range(num_gen)]
    mix_gen = normalMixtureCalculator()
    mixtures,mixture_sources = mix_gen.calculate_mixture_distributions(rs,pis,mus,var)
    tf_mixtures_pdfs,tf_mixture_sources = generate_vector_gauss_mixture(rs,pis,mus,tf.cast(tf.sqrt(var),tf.float64))
    for i in range(num_gen):
        plt.figure()
        plt.plot(r,mixtures[i],label = "Self generated distribution")
        plt.plot(r,tf_mixtures_pdfs[i],label = "TF generated dist")
        plt.legend()
    #rs = [r for i in range(num_profiles)]
    ##generated_gaussians, parameters, gaussMixtures = generate_vector_random_gauss_mixture(rs,kg)
    #pis = [np.random.rand() for k in range(kg)]
    #spi = np.sum(pis)
    #pis = [pi/spi for pi in pis]
    #cat = tfp.distributions.Categorical(probs = pis)