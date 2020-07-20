# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 18:00:59 2020

@author: marti
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
#import tensorflow_addons as tfa #AdamW
import tensorflow_probability as tfp#normal dist
from copy import deepcopy
import sys
import  logging
from datetime import datetime
import numpy as np
import pandas as pd

from normal_dist_calculator import generate_tensor_mixture_model
from Reparameterizer import reparameterizer, normalize_profiles,renormalize_profiles

import pickle

import argparse
import EinastoSim

np.random.seed(42)
tf.random.set_seed(42)
plt.close("all")

num_profile_train = 1
kg = 1
r = np.linspace(0,1000,101)
rs = np.asarray([r for i in range(num_profile_train)])

gaussians, parameters,constituents_train = EinastoSim.generate_n_k_gaussian_parameters(rs,num_profile_train,kg)
plt.figure()
plt.plot(r,gaussians[0])
param = parameters[0]
plt.title("{},{}".format(param[0],param[1]))
for const in constituents_train[0]:
    plt.plot(r,const)

