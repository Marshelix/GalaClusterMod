# -*- coding: utf-8 -*-
"""
Created on Sun May 24 10:07:00 2020

@author: modified from https://www.katnoria.com/mdn/ , a tutorial on tf2 gdns

Made to fit the Einasto profile data generated by Martin Sanner

Recent changes:
    Remove plots from code, save raw data via pickle

List of outstanding issues:
    - figure out a speedup
    
    - Test convergence of train/test errors simultenaity <- failed
    - Figure out crosschecks for model work
    - Calculate those crosschecks
    
    - Implement argument parsing for multitest
    
"""

import numpy as np
import matplotlib.pyplot as plt
import EinastoSim
import h5py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow_addons as tfa #AdamW
import tensorflow_probability as tfp#normal dist
from copy import deepcopy
import sys
import  logging
from datetime import datetime

import pandas as pd

from normal_dist_calculator import generate_tensor_mixture_model
from Reparameterizer import reparameterizer, normalize_profiles,renormalize_profiles

import pickle



'''
Profiling from https://osf.io/upav8/ by Sebastian Maathöt https://www.youtube.com/watch?v=8qEnExGLZfY
'''
import cProfile, pstats, io

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
'''

'''

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
def train_model(model,optimizer,dataset,associated_r,EPOCHS,max_patience,target_loss,test_parameters,test_profiles,t_a_r,start_parameters = {}):
    best_model = model
    best_loss = np.inf
    max_diff = 0.0  #differential loss
    epoch = start_parameters.get("epoch",1)
    training_bool = epoch in range(EPOCHS)
    counter = start_parameters.get("counter",0)
    print_every = np.max([1, EPOCHS/100])
    
    counters = start_parameters.get("counters", [])
    
    test_MAEs = start_parameters.get("test_MAEs",[])
    
    MSEs = start_parameters.get("MSEs",[])
    
    
    overlap_ratios = start_parameters.get("overlaps",[])
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
    
    while training_bool:
        for train_x, train_y in dataset:
            with tf.GradientTape() as tape:
                pi_, mu_, var_ = model(train_x,training = True)
                sample, prob_array_training = generate_tensor_mixture_model(associated_r,pi_,mu_,var_)    
                loss = tf.losses.mean_absolute_error(train_y,sample)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
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
                best_model = tf.keras.models.clone_model(model)
                best_model.save_weights(".\\models\\weights\\Run_{}\\Run".format(run_id))
                counter = 0
        
        losses.append(tf.reduce_mean(loss))
        #calculate mse
        pi_tt,mu_tt,var_tt = best_model.predict(np.asarray(test_parameters))
        sample_preds, sample_probability_array = generate_tensor_mixture_model(t_a_r,pi_tt,mu_tt,var_tt)
        mse_error_profiles = tf.reduce_mean(tf.losses.MSE(ttp_renormed,sample_preds))
        MSEs.append(mse_error_profiles)
        
        #mae to compare to train loss
        mae_error_profiles_test = tf.reduce_mean(tf.losses.mean_absolute_error(sample_preds,ttp_renormed))
        test_MAEs.append(mae_error_profiles_test)
        
        
        #calculate overlap
        s_overlaps = []
        sample_overlaps = []
        for overlap_counter in range(n_test_profiles):    
            s_overlaps.append(np.dot(np.transpose(ttp_renormed[overlap_counter]),ttp_renormed[overlap_counter])) #ignore constant multiplier
            sample_overlaps.append(np.dot(np.transpose(sample_preds[overlap_counter]),sample_preds[overlap_counter]))
        overlap_ratio = tf.reduce_mean([s_overlaps[current_overlap]/sample_overlaps[current_overlap] for current_overlap in range(len(s_overlaps))])
        overlap_ratios.append(overlap_ratio)
        
        counters.append(100*counter/counter_max) #counter percentage
        
        
        if len(test_MAEs) > 1:
            tmae_diffs = test_MAEs[1:]-test_MAEs[:-1]
            avg_test_loss_diff = np.mean(tmae_diffs)
        if len(losses) > 1:
            mae_diffs = losses[1:]-losses[:-1]
            avg_train_loss_diff = np.mean(mae_diffs)
        
        training_bool = epoch in range(EPOCHS)
        
        loss_break = (best_loss.numpy() < loss_target)
        loss_break = loss_break or (diff < 0) 
        
        loss_divergence = abs(tf.reduce_mean(loss)-mae_error_profiles_test) > max_loss_divergence if epoch > 1 else False
        '''
        Continue training if epochs left or if current best loss is worse than the target
        Stop training if training/test losses diverge or if patience lost
        '''
        training_bool = (epoch <= EPOCHS or not loss_break) if ((counter//counter_max < 1) and not loss_divergence) else False
        
        time_estimate_per_epoch = (datetime.now()-train_start)/epoch
        if epoch % print_every == 0:
            logging.info('Epoch {}/{}: Elapsed Time: {};Remaining Time estimate: {}; loss = {}, test loss = {};loss delta: {};test loss delta: {}; Patience: {} %; MSE: {}; overlap: {}'.format(epoch, EPOCHS,datetime.now() - train_start,time_estimate_per_epoch*(EPOCHS-epoch), losses[-1],mae_error_profiles_test,avg_train_loss_diff,avg_test_loss_diff,100*counter/counter_max,mse_error_profiles, overlap_ratio))       
        epoch = epoch+1
    
    
    logging.info("Training completed after {}/{} epochs. Patience: {} %:: Best Loss: {}".format(epoch, EPOCHS, 100*counter/counter_max, best_loss))
    logging.info("Reason for exiting: loss_break: {}, diff < 0: {}, loss divergence: {}".format(loss_break,diff<0,loss_divergence))
    score_file = "./scores.csv"
    logging.info("Saving best score {} to {}".format(best_loss,score_file))
    
    score_df = pd.read_csv(score_file)
    score_df.at[run_id,"MAE"] = best_loss
    score_df.to_csv(score_file)
    
    return best_model, losses,MSEs,counters,overlap_ratios,test_MAEs


    
if __name__ == "__main__":
    run_file = "./runID.txt"
    run_id = -1
    if not os.path.isfile(run_file):
        with open(run_file,"w") as f:
            run_id = 1
            f.write(str(run_id))
            
    else:
        with open(run_file,"r") as f:
            run_id = int(f.read())
    logging.info("="*20)
    logging.info("Starting new run #{} at {}".format(run_id,d_string))
    logging.info("="*20)
    num_profile_train = 10000
    

    logging.info("Running on GPU: {}".format(len(tf.config.experimental.list_physical_devices('GPU')) > 0))
    logging.info("Generating {} Profiles for training".format(num_profile_train))
    
    sample_profiles,profile_params,associated_r = EinastoSim.generate_n_random_einasto_profile_maggie(num_profile_train)
    
    logging.info("Defining backend type: TF.float64")
    tf.keras.backend.set_floatx("float64")
    logging.info("Running Logged renormalized Profiles.")
    
    # remove log as test
    sample_profiles_logged = np.asarray([np.log(p) for p in sample_profiles]).astype(np.float64)
    sample_reparam = reparameterizer(sample_profiles_logged)
    sample_profiles_renormed = normalize_profiles(sample_profiles_logged).astype(np.float64)#np.asarray(calculate_renorm_profiles(sample_profiles_logged)).astype(np.float64)
    
    X_full = profile_params#create_input_vectors(profile_params, associated_r) 
    X_full = np.asarray(X_full).astype(np.float64)
    l = len(profile_params[0])#+1    #current r and all params
    
    #output dimension
    out_dim = 1 #just r
    # Number of gaussians to represent the multimodal distribution
    k = 4
    logging.info("Running {} dimensions on {} distributions".format(out_dim,k))
    # Network
    input = tf.keras.Input(shape=(l,))
    input_transfer_layer = tf.keras.layers.Dense(1,activation = None,dtype = tf.float64)
    layer = tf.keras.layers.Dense(100, activation='tanh', name='baselayer',dtype = tf.float64)(input)
    mu = tf.keras.layers.Dense((k*out_dim), activation='exponential', name='mean_layer',dtype = tf.float64)(layer)
    # variance (should be greater than 0 so we exponentiate it)
    var_layer = tf.keras.layers.Dense((k*out_dim), activation=None, name='dense_var_layer')(layer)
    var = tf.keras.layers.Lambda(lambda x: tf.math.exp(x), output_shape=(k,), name='variance_layer',dtype = tf.float64)(var_layer)
    # mixing coefficient should sum to 1.0
    pi = tf.keras.layers.Dense(k*out_dim, activation=None, name='pi_layer',dtype = tf.float64)(layer)

    
    losses = []
    EPOCHS = 1000
    
    
    # Define model and optimizer
    model = tf.keras.models.Model(input, [pi, mu, var])
    lr = 1e-3
    wd = 1e-6
    
    optimizer = tfa.optimizers.AdamW(lr,wd)
    model.summary()
    
    N = np.asarray(X_full).shape[0]
    num_batches = 1
    batchsize = N//num_batches
    dataset = tf.data.Dataset \
    .from_tensor_slices((X_full, sample_profiles_renormed)) \
    .shuffle(N).batch(batchsize)
    
    # Start training
    
    
    n_test_profiles = 10000
    train_testing_profile, tt_p_para,t_a_r = EinastoSim.generate_n_random_einasto_profile_maggie(n_test_profiles)
    ttp_logged = np.asarray([np.log(p) for p in train_testing_profile]).astype(np.float64)
    ttp_reparam = reparameterizer(ttp_logged)
    ttp_renormed = normalize_profiles(ttp_logged).astype(np.float64)#np.asarray(calculate_renorm_profiles(ttp_logged)).astype(np.float64)
    X_tt = tt_p_para#create_input_vectors(tt_p_para,t_a_r)
    
    counter_max = 5000
    
    loss_target = 1e-3
    
    best_model,losses,MSEs,counters,overlap_ratios,test_MAEs = train_model(model,optimizer,dataset,associated_r,EPOCHS,counter_max,loss_target,X_tt,ttp_renormed,t_a_r)
    
    
    
    plot_folder = ".//plots//Run_{}//".format(run_id)
    save_folder = ".//models//Run_{}//best_model".format(run_id)
    data_folder = ".//data//Run_{}//".format(run_id)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        
    logging.info("Saving best model to {}".format(save_folder))
    best_model.save_weights(save_folder)
                
    logging.info("Dumping data to {}".format(data_folder))
    now = datetime.now()
    with open(data_folder+"MAE_Losses.dat","wb") as f:
        pickle.dump(losses,f)
    with open(data_folder+"MSE_Losses.dat","wb") as f:
        pickle.dump(MSEs,f)
    with open(data_folder+"Patience.dat","wb") as f:
        pickle.dump(counters,f)
    with open(data_folder+"overlap.dat","wb") as f:
        pickle.dump(overlap_ratios,f)
    with open(data_folder+"mae_test_losses.dat","wb") as f:
        pickle.dump(test_MAEs,f)
    
    n_test_profiles = 10
    test_profiles,t_profile_params,t_associated_r = EinastoSim.generate_n_random_einasto_profile_maggie(n_test_profiles)
    t_sample_profiles_logged = np.asarray([np.log(p) for p in test_profiles]).astype(np.float64)
    t_s_reparam = reparameterizer(t_sample_profiles_logged)
    t_s_renorm = normalize_profiles(t_sample_profiles_logged).astype(np.float64)
    X_test = t_profile_params#create_input_vectors(t_profile_params,t_associated_r)
    
    pi_test, mu_test,var_test = best_model.predict(np.asarray(X_test))
    
    test_data = {"Profiles":t_s_renorm, "STDParams":{"Pi":pi_test,"Mu":mu_test,"Var":var_test},"Xtest":X_test, "r":t_associated_r}
    with open(data_folder+"test_data.dat","wb") as f:
        pickle.dump(test_data,f)
    
    with open(run_file,"w") as f:
        f.write(str(run_id +1))