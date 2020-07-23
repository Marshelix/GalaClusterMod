# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:58:20 2020

@author: Martin Sanner
Gaussian Predictor: 
    Implementation of the GMM trained on the overall mean and variance of gaussians/gaussian mixtures. Should attain 100% accuracy.
    
    keep pis unnormalized as in gmm example
"""

import numpy as np
import matplotlib.pyplot as plt
import EinastoSim
import h5py

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

import pandas as pd

from normal_dist_calculator import generate_tensor_mixture_model
from Reparameterizer import reparameterizer, normalize_profiles,renormalize_profiles

import pickle

import argparse
np.random.seed(42)
tf.random.set_seed(42)
plt.close("all")
'''
Logging: Taken from https://stackoverflow.com/a/13733863
'''

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


if __name__ == "__main__":
    
    run_file = "./runID_gauss.txt"
    run_id = -1
    if not os.path.isfile(run_file):
        with open(run_file,"w") as f:
            run_id = 1
            f.write(str(run_id))
            
    else:
        with open(run_file,"r") as f:
            run_id = int(f.read())
    
    logging.info("="*20)
    logging.info("Run {}".format(run_id))
    logging.info("="*20)
    
    num_profile_train = 500
    kg = 1
    logging.info("Running on GPU: {}".format(len(tf.config.experimental.list_physical_devices('GPU')) > 0))
    logging.info("Generating {} normals based on {} distribution for training".format(num_profile_train, kg))
    r = np.linspace(-10,10,1001)
    rs = np.asarray([r for i in range(num_profile_train)])
    gaussians, parameters,constituents_train = EinastoSim.generate_n_k_gaussian_parameters(rs,num_profile_train,kg)
    
    logging.info("Defining backend type: TF.float64")
    tf.keras.backend.set_floatx("float64")
    
    #no need to log or normalize,already consists of gaussians
    X_full = parameters
    X_full = np.asarray(X_full).astype(np.float64)
    
    losses = []
    EPOCHS = 250
    
    l = len(X_full[0])
    
    #output dimension
    out_dim = 1
    # Number of gaussians to represent the multimodal distribution
    k = 4
    
    logging.info("Running {} dimensions on {} distributions".format(out_dim,k))
    
    
    
    model = tf.keras.Sequential([tf.keras.Input(shape=(l,)), 
                                  tf.keras.layers.Dense(50,activation = 'tanh',name = 'Intermediate_Layer',dtype = tf.float64),
                                  tf.keras.layers.Dropout(0.4,dtype = tf.float64),
                                  tf.keras.layers.Dense(50,activation = 'tanh',name = 'Intermediate_Layer2',dtype = tf.float64),
                                  tf.keras.layers.Dense(3*k*out_dim,activation = None, name = "End_Layer")])
    
    
    # Define model and optimizer
    lr = 1e-4
    wd = 0#1e-6
    
    optimizer = tf.optimizers.Adam(lr)#tfa.optimizers.AdamW(lr,wd)

    model.summary()
    
    N = np.asarray(X_full).shape[0]
    num_batches = 1
    batchsize = N//num_batches
    dataset = tf.data.Dataset \
    .from_tensor_slices((X_full, gaussians)) \
    .shuffle(N).batch(batchsize)
    
    # Start training
    n_test_profiles = 500
    test_gaussians, test_params,_ = EinastoSim.generate_n_k_gaussian_parameters(rs,n_test_profiles, kg)
    #test_gaussians = np.asarray([np.log(p) for p in test_gaussians])
    X_tt = test_params
    counter_max = 5000
    
    loss_target = 1e-3
    
    best_model = model
    best_loss = tf.cast(np.inf,tf.float64)
    max_diff = 0.0  #differential loss
    start_parameters = {}
    epoch = start_parameters.get("epoch",1)
    training_bool = epoch in range(EPOCHS)
    
    counter = start_parameters.get("counter",0)
    print_every = np.max([1, EPOCHS/100])
    
    counters = start_parameters.get("counters", [])
    
    test_MAEs = start_parameters.get("test_MAEs",[])
    
    MSEs = start_parameters.get("MSEs",[])
    
    minimum_delta = 3e-5
    diff = 0
    loss_break = False
    
    max_loss_divergence = 0.2
    
    avg_train_loss_diff = 0
    avg_test_loss_diff = 0
    
    rolling_mean_length = 10
    
    logging.info("="*10+"Training info"+"="*10)
    logging.debug('Print every {} epochs'.format(print_every))
    logging.info("Learning Parameters: lr = {} \t wd = {}".format(lr,wd))
    logging.info("Patience: {} increases".format(counter_max))
    logging.info("Minimum delta loss to not lose patience: {}".format(minimum_delta))
    logging.info("Target loss: < {}".format(loss_target))
    logging.info("# Samples: {}".format(n_test_profiles))
    logging.info("# Training Profiles: {}".format(num_profile_train))
    logging.info("Printing every {} epochs".format(print_every))
    logging.info("Maximum loss divergence: {}".format(max_loss_divergence))
    logging.info("Maximum length values taken into account: {}".format(rolling_mean_length))
    logging.info("="*(33))
    train_start = datetime.now()
    logging.info("Starting training at: {}".format(train_start))
    time_estimate_per_epoch = np.inf
    loss_divergence = False
    while training_bool:
        for train_x, train_y in dataset:
            with tf.GradientTape() as tape:
                #pi_, mu_, var_ = model(train_x,training = True)
                prediction_vector = model(train_x,training = True)
                pi_un,mu_,var_log = tf.split(prediction_vector,3,1)
                pi_ = pi_un#tf.sigmoid(pi_un)
                var_ = tf.exp(var_log)
                sample, prob_array_training = generate_tensor_mixture_model(rs,pi_,mu_,var_)    
                loss = tf.losses.mean_absolute_error(train_y,sample)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        losses.append(tf.reduce_mean(loss))
        #calculate mse
        pi_tt_base,mu_tt,var_tt_log = tf.split(model.predict(np.asarray(X_tt)),3,1)
        pi_tt = pi_tt_base#tf.sigmoid(pi_tt_base)
        var_tt = tf.exp(var_tt_log)
        sample_preds, sample_probability_array = generate_tensor_mixture_model(rs,pi_tt,mu_tt,var_tt)
        mse_error_profiles = tf.reduce_mean(tf.losses.MSE(test_gaussians,sample_preds))
        MSEs.append(mse_error_profiles)
        
        #mae to compare to train loss
        mae_error_profiles_test = tf.cast(tf.reduce_mean(tf.losses.mean_absolute_error(sample_preds,test_gaussians)),tf.float64)
        test_MAEs.append(mae_error_profiles_test)
        
        if mae_error_profiles_test < best_loss:
                best_loss = tf.reduce_mean(mae_error_profiles_test)
                best_model = tf.keras.models.clone_model(model)
                #best_model.save_weights(".\\models\\weights\\Run_{}\\Run".format(run_id))
                counter = 0
        '''
            Amend counter if: not better than the best loss, delta_loss < minimum_delta, delta_loss < max_diff(best delta so far)
            Reset counter if best loss overcome
            '''
        if mae_error_profiles_test > best_loss:
            counter += 1
        if len(losses) > 1:
            diff = losses[-1] - losses[-2]
            if diff < minimum_delta or diff < max_diff:
                counter += 1
            elif diff > max_diff + minimum_delta:
                max_diff = diff
                counter -= 1 #keep going if differential low enough, even if loss > min
                counter = max([0,counter]) #keep > 0
        counters.append(100*counter/counter_max) #counter percentage
        
        
        if len(test_MAEs) > 1:
            tmae_diffs = np.asarray(test_MAEs[1:])-np.asarray(test_MAEs[:-1])
            max_tmae_idx = min(len(tmae_diffs),rolling_mean_length)
            avg_test_loss_diff = np.mean(tmae_diffs[-max_tmae_idx:])
        if len(losses) > 1:
            mae_diffs = np.asarray(losses[1:])-np.asarray(losses[:-1])
            max_mae_idx = min(len(mae_diffs),rolling_mean_length)
            avg_train_loss_diff = np.mean(mae_diffs[-max_mae_idx:])
        
        training_bool = epoch in range(EPOCHS)
        
        loss_break = (best_loss.numpy() < loss_target)
        loss_break = loss_break or (diff < 0) 
        
        loss_divergence = abs(tf.reduce_mean(loss)-tf.cast(mae_error_profiles_test,dtype = tf.float64)) > max_loss_divergence if epoch > 1 else False
        '''
        Continue training if epochs left or if current best loss is worse than the target
        Stop training if training/test losses diverge or if patience lost
        '''
        training_bool = (epoch <= EPOCHS or not loss_break) if ((counter//counter_max < 1) and not loss_divergence) else False
        
        time_estimate_per_epoch = (datetime.now()-train_start)/epoch
        if epoch % print_every == 0:
            logging.info('Epoch {}/{}: Elapsed Time: {};Remaining Time estimate: {}; loss = {}, test loss = {};loss delta: {};test loss delta: {}; Patience: {} %; MSE: {};'.format(epoch, EPOCHS,datetime.now() - train_start,time_estimate_per_epoch*(EPOCHS-epoch), losses[-1],mae_error_profiles_test,avg_train_loss_diff,avg_test_loss_diff,100*counter/counter_max,mse_error_profiles))       
        epoch = epoch+1
    
    
    logging.info("Training completed after {}/{} epochs. Patience: {} %:: Best Loss: {}".format(epoch, EPOCHS, 100*counter/counter_max, best_loss))
    logging.info("Reason for exiting: loss_break: {}, diff < 0: {}, loss divergence: {}".format(loss_break,diff<0,loss_divergence))
    
    data_folder = ".//data//gauss_{}//".format(run_id)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    logging.info("Dumping data to {}".format(data_folder))
    now = datetime.now()
    with open(data_folder+"MAE_Losses.dat","wb") as f:
        pickle.dump(losses,f)
    with open(data_folder+"MSE_Losses.dat","wb") as f:
        pickle.dump(MSEs,f)
    with open(data_folder+"Patience.dat","wb") as f:
        pickle.dump(counters,f)
    with open(data_folder+"mae_test_losses.dat","wb") as f:
        pickle.dump(test_MAEs,f)
    
    
    
    n_test_profiles = 10
    test_gauss,test_params,generators = EinastoSim.generate_n_k_gaussian_parameters(rs,n_test_profiles,kg)
    #test_gauss = np.asarray([np.log(p) for p in test_gauss])
    X_test = test_params
    pi_test_base, mu_test,var_test_log = tf.split(best_model.predict(np.asarray(X_test)),3,1)
    pi_test = pi_test_base#tf.sigmoid(pi_test_base)
    var_test = tf.exp(var_test_log)
    sample_preds, sample_probability_array = generate_tensor_mixture_model(rs, pi_test,mu_test, var_test)
    test_data = {"Profiles":test_gauss, "STDParams":{"Pi":pi_test,"Mu":mu_test,"Var":var_test},"Xtest":X_test, "r":rs}
    with open(data_folder+"test_data.dat","wb") as f:
        pickle.dump(test_data,f)
    with open(run_file,"w") as f:
        f.write(str(run_id +1))