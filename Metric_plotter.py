# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 10:28:53 2020

@author: Martin Sanner
Metric plotter file

Takes an input on which type to plot, then loads all testing data etc and calculates corresponding errors
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
from normal_dist_calculator import generate_tensor_mixture_model,generate_vector_gauss_mixture,generate_vector_random_gauss_mixture

import pickle
from datetime import datetime
import os


