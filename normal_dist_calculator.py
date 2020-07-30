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
    n,k = pi_values.shape
    n2,k2 = mu_values.shape
    n3,k3 = var_values.shape
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
    '''
    usage of the MixtureSameFamily class seems to have some issues. Trace in here
    WARNING:tensorflow:From F:\Addons\Anaconda\envs\project\lib\site-packages\tensorflow_probability\python\distributions\categorical.py:225: Categorical._logits_deprecated_behavior (from tensorflow_probability.python.distributions.categorical) is deprecated and will be removed after 2019-10-01.
    Instructions for updating:
    
    The `logits` property will return `None` when the distribution is parameterized with `logits=None`. Use `logits_parameter()` instead.

    WARNING:tensorflow:From F:\Addons\Anaconda\envs\project\lib\site-packages\tensorflow\python\ops\math_ops.py:2403: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where

    '''
    num_profiles = 10#args.num_profile
    kg = 4
    r = np.linspace(-10,10,1001)
    rs = [r for i in range(num_profiles)]
    #generated_gaussians, parameters, gaussMixtures = generate_vector_random_gauss_mixture(rs,kg)
    pis = [np.random.rand() for k in range(kg)]
    spi = np.sum(pis)
    pis = [pi/spi for pi in pis]
    cat = tfp.distributions.Categorical(probs = pis)