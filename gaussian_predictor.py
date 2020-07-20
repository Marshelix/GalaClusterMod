# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:58:20 2020

@author: Martin Sanner
Gaussian Predictor: 
    Implementation of the GMM trained on the overall mean and variance of gaussians/gaussian mixtures. Should attain 100% accuracy.

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
    num_profile_train = 100
    kg = 1
    logging.info("Running on GPU: {}".format(len(tf.config.experimental.list_physical_devices('GPU')) > 0))
    logging.info("Generating {} Profiles with {} distribution for training".format(num_profile_train, kg))
    r = np.linspace(-10,10,1001)
    rs = np.asarray([r for i in range(num_profile_train)])
    gaussians, parameters,constituents_train = EinastoSim.generate_n_k_gaussian_parameters(rs,num_profile_train,kg)
    #gaussians = np.asarray([np.log(p) for p in gaussians])
    logging.info("Defining backend type: TF.float64")
    tf.keras.backend.set_floatx("float64")
    logging.info("Running Logged renormalized Profiles.")
    
    #no need to log or normalize,already consists of gaussians
    X_full = parameters#create_input_vectors(profile_params, associated_r) 
    X_full = np.asarray(X_full).astype(np.float64)
    
    losses = []
    EPOCHS = 100
    
    l = len(X_full[0])
    
    #output dimension
    out_dim = 1
    # Number of gaussians to represent the multimodal distribution
    k = 4
    
    logging.info("Running {} dimensions on {} distributions".format(out_dim,k))
    # Define initial Model
    input = tf.keras.Input(shape=(l,))
    input_transfer_layer = tf.keras.layers.Dense(1,activation = None,dtype = tf.float64)
    layer = tf.keras.layers.Dense(50, activation='tanh', name='baselayer',dtype = tf.float64)(input)
    mu = tf.keras.layers.Dense((k*out_dim), activation=None, name='mean_layer',dtype = tf.float64)(layer)
    # variance (should be greater than 0 so we exponentiate it)
    var_layer = tf.keras.layers.Dense((k*out_dim), activation=None, name='dense_var_layer')(layer)
    var = tf.keras.layers.Lambda(lambda x: tf.math.exp(x), output_shape=(k,), name='variance_layer',dtype = tf.float64)(var_layer)
    # mixing coefficient should sum to 1.0
    pi = tf.keras.layers.Dense(k*out_dim, activation=None, name='pi_layer',dtype = tf.float64)(layer)
    model = tf.keras.models.Model(input, [pi, mu, var])
    
    
    model2 = tf.keras.Sequential([input, 
                                  tf.keras.layers.Dense(50,activation = 'tanh',name = 'Intermediate_Layer',dtype = tf.float64),
                                  tf.keras.layers.Dense(50,activation = 'tanh',name = 'Intermediate_Layer2',dtype = tf.float64),
                                  tf.keras.layers.Dense(3*k*out_dim,activation = None, name = "End_Layer")])
    
    
    
    #define secondary model: similar to above, one output vector, split into subvectors, running functions on pi, var
    
    
    
    # Define model and optimizer
    lr = 1e-3
    wd = 0#1e-6
    
    optimizer = tf.optimizers.Adam(lr)#tfa.optimizers.AdamW(lr,wd)
    model.summary()
    model2.summary()
    
    N = np.asarray(X_full).shape[0]
    num_batches = 1
    batchsize = N//num_batches
    dataset = tf.data.Dataset \
    .from_tensor_slices((X_full, gaussians)) \
    .shuffle(N).batch(batchsize)
    
    # Start training
    n_test_profiles = 1000
    test_gaussians, test_params,_ = EinastoSim.generate_n_k_gaussian_parameters(rs,n_test_profiles, kg)
    #test_gaussians = np.asarray([np.log(p) for p in test_gaussians])
    X_tt = test_params
    counter_max = 5000
    
    loss_target = 1e-3
    
    best_model = model
    best_loss = np.inf
    max_diff = 0.0  #differential loss
    start_parameters = {}
    epoch = start_parameters.get("epoch",1)
    training_bool = epoch in range(EPOCHS)
    
    counter = start_parameters.get("counter",0)
    print_every = np.max([1, EPOCHS/100])
    
    counters = start_parameters.get("counters", [])
    
    test_MAEs = start_parameters.get("test_MAEs",[])
    
    MSEs = start_parameters.get("MSEs",[])
    
    minimum_delta = 5e-7
    diff = 0
    loss_break = False
    
    max_loss_divergence = 2
    
    avg_train_loss_diff = 0
    avg_test_loss_diff = 0
    
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
    logging.info("="*(33))
    train_start = datetime.now()
    logging.info("Starting training at: {}".format(train_start))
    time_estimate_per_epoch = np.inf
    loss_divergence = False
    while training_bool:
        for train_x, train_y in dataset:
            with tf.GradientTape() as tape:
                #pi_, mu_, var_ = model(train_x,training = True)
                prediction_vector = model2(train_x,training = True)
                pi_un,mu_,var_log = tf.split(prediction_vector,3,1)
                pi_ = pi_un#tf.sigmoid(pi_un)
                var_ = tf.exp(var_log)
                sample, prob_array_training = generate_tensor_mixture_model(rs,pi_,mu_,var_)    
                loss = tf.losses.mean_absolute_error(train_y,sample)
            gradients = tape.gradient(loss, model2.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model2.trainable_variables))
            '''
            Amend counter if: not better than the best loss, delta_loss < minimum_delta, delta_loss < max_diff(best delta so far)
            Reset counter if best loss overcome
            '''
            if tf.reduce_mean(loss) > best_loss:
                counter += 1/num_batches
            if len(losses) > 1:
                diff = losses[-1] - losses[-2]
                if diff < minimum_delta or diff < max_diff:
                    counter += 1/num_batches
                elif diff > max_diff + minimum_delta:
                    max_diff = diff
                    counter -= 1/num_batches #keep going if differential low enough, even if loss > min
                    counter = max([0,counter]) #keep > 0
                    
            if tf.reduce_mean(loss) < best_loss:
                best_loss = tf.reduce_mean(loss)
                best_model = tf.keras.models.clone_model(model2)
                #best_model.save_weights(".\\models\\weights\\Run_{}\\Run".format(run_id))
                counter = 0
        
        losses.append(tf.reduce_mean(loss))
        #calculate mse
        pi_tt_base,mu_tt,var_tt_log = tf.split(best_model.predict(np.asarray(X_tt)),3,1)
        pi_tt = pi_tt_base#tf.sigmoid(pi_tt_base)
        var_tt = tf.exp(var_tt_log)
        sample_preds, sample_probability_array = generate_tensor_mixture_model(rs,pi_tt,mu_tt,var_tt)
        mse_error_profiles = tf.reduce_mean(tf.losses.MSE(test_gaussians,sample_preds))
        MSEs.append(mse_error_profiles)
        
        #mae to compare to train loss
        mae_error_profiles_test = tf.reduce_mean(tf.losses.mean_absolute_error(sample_preds,test_gaussians))
        test_MAEs.append(mae_error_profiles_test)
        
        
        counters.append(100*counter/counter_max) #counter percentage
        
        
        if len(test_MAEs) > 1:
            tmae_diffs = np.asarray(test_MAEs[1:])-np.asarray(test_MAEs[:-1])
            avg_test_loss_diff = np.mean(tmae_diffs)
        if len(losses) > 1:
            mae_diffs = np.asarray(losses[1:])-np.asarray(losses[:-1])
            avg_train_loss_diff = np.mean(mae_diffs)
        
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
    
    data_folder = ".\\data\\gauss\\"
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
    
    
    '''
    plt.figure()
    plt.plot(losses, label = "Training Loss")
    plt.plot(test_MAEs, label = "Test  Loss")
    plt.legend()
    plt.title("Losses")
    plt.ylabel("MAE")
    plt.xlabel("Epoch")
    '''
    
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
    '''    
    plt_gen = False
    
    for i in range(n_test_profiles):
        profile_sample = sample_preds[i,:]
        test_prof = test_gauss[i]
        plt.figure()
        plt.plot(rs[i],test_prof,label = "True profile - mu {} var {}".format(test_params[i][0],test_params[i][1]))
        
        mng = plt.get_current_fig_manager()
        
        plt.plot(r,profile_sample, label = "Sample")
        for kd in range(k):
            plt.plot(r,sample_probability_array[i][kd],label = "Sampled Constituent {}: pi: {}; mu: {}; var {}".format(kd,pi_test[i][kd],mu_test[i][kd],var_test[i][kd])) #plotting probabilities found in the method
        if plt_gen:
            for gen in range(kg):
                plt.plot(r,generators[i][gen],label = "Generator {}".format(gen))
        plt.legend()
        plt.title("Loss: {}".format(tf.losses.mean_absolute_error(test_prof, profile_sample)))
        plt.xlabel("Radius")
        plt.ylabel("Density {} []".format(u"\u03C1"))
        mng.full_screen_toggle()
        plt.show()
        plt.pause(1e-1)
    '''