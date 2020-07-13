# -*- coding: utf-8 -*-
"""
Created on Sun May 24 10:07:00 2020

@author: modified from https://www.katnoria.com/mdn/ , a tutorial on tf2 gdns

Made to fit the Einasto profile data generated by Martin Sanner

Recent changes:
    Remove plots from code, save raw data via pickle

List of outstanding issues:
    - Test convergence of train/test errors simultenaity
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
#import tensorflow_addons as tfa #AdamW
import tensorflow_probability as tfp#normal dist
from copy import deepcopy
import sys
import  logging
from datetime import datetime

import pandas as pd

from normal_dist_calculator import generate_tensor_mixture_model
from Reparameterizer import reparameterizer

import pickle

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
    num_profile_train = 400
    

    logging.info("Running on GPU: {}".format(tf.test.is_gpu_available()))
    logging.info("Generating {} Profiles for training".format(num_profile_train))
    
    sample_profiles,profile_params,associated_r = EinastoSim.generate_n_random_einasto_profile_maggie(num_profile_train)
    
    logging.info("Defining backend type: TF.float64")
    tf.keras.backend.set_floatx("float64")
    logging.info("Running Logged renormalized Profiles.")
    
    # remove log as test
    sample_profiles_logged = np.asarray([np.log(p) for p in sample_profiles]).astype(np.float64)
    sample_reparam = reparameterizer(sample_profiles_logged)
    sample_profiles_renormed = sample_reparam.calculate_parameterization().astype(np.float64)#np.asarray(calculate_renorm_profiles(sample_profiles_logged)).astype(np.float64)
    
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
    layer = tf.keras.layers.Dense(190, activation='tanh', name='baselayer',dtype = tf.float64)(input)
    mu = tf.keras.layers.Dense((k*out_dim), activation=None, name='mean_layer',dtype = tf.float64)(layer)
    # variance (should be greater than 0 so we exponentiate it)
    var_layer = tf.keras.layers.Dense((k*out_dim), activation=None, name='dense_var_layer')(layer)
    var = tf.keras.layers.Lambda(lambda x: tf.math.exp(x), output_shape=(k,), name='variance_layer',dtype = tf.float64)(var_layer)
    # mixing coefficient should sum to 1.0
    pi = tf.keras.layers.Dense(k*out_dim, activation='softmax', name='pi_layer',dtype = tf.float64)(layer)

    
    losses = []
    EPOCHS = 250
    print_every = int(EPOCHS/100)
    
    # Define model and optimizer
    model = tf.keras.models.Model(input, [pi, mu, var])
    lr = 1e-3
    wd = 0#1e-6
    
    optimizer = tf.keras.optimizers.Adam(lr)
    model.summary()
    
    N = np.asarray(X_full).shape[0]
    num_batches = 10
    batchsize = N//num_batches
    dataset = tf.data.Dataset \
    .from_tensor_slices((X_full, sample_profiles_renormed)) \
    .shuffle(N).batch(batchsize)
    
    # Start training
    best_model = model
    best_loss = np.inf
    max_diff = 0.0  #differential loss
    epoch = 1
    training_bool = epoch in range(EPOCHS)
    counter = 0
    counter_max = 5000
    counters = []
    
    minimum_delta = 5e-7
    
    test_MAEs = []
    
    MSEs = []
    train_testing_profile, tt_p_para,t_a_r = EinastoSim.generate_n_random_einasto_profile_maggie(1)
    ttp_logged = np.asarray([np.log(p) for p in train_testing_profile]).astype(np.float64)
    ttp_reparam = reparameterizer(ttp_logged)
    ttp_renormed = ttp_reparam.calculate_parameterization().astype(np.float64)#np.asarray(calculate_renorm_profiles(ttp_logged)).astype(np.float64)
    X_tt = tt_p_para#create_input_vectors(tt_p_para,t_a_r)
    
    overlap_ratios = []
    num_samples = 10
    likelihood_minimum = 0.9
    loss_target = 1e-3#-np.log(likelihood_minimum)
    diff = 0
    logging.info("="*10+"Training info"+"="*10)
    logging.debug('Print every {} epochs'.format(print_every))
    logging.info("Learning Parameters: lr = {} \t wd = {}".format(lr,wd))
    logging.info("Patience: {} increases".format(counter_max))
    logging.info("Minimum delta loss to not lose patience: {}".format(minimum_delta))
    logging.info("Target loss: < {}".format(loss_target))
    logging.info("# Samples: {}".format(num_samples))
    logging.info("# Training Profiles: {}".format(num_profile_train))
    logging.info("Printing every {} epochs".format(print_every))
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
                
            # compute and apply gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            if tf.reduce_mean(loss) > best_loss:
                counter += 1
            
            if len(losses) > 1:
                diff = losses[-1] - losses[-2]
                if diff < max_diff:
                    counter += 1/num_batches
                if diff < minimum_delta:
                    counter += 1/num_batches
                elif diff > max_diff + minimum_delta:
                    max_diff = diff
                    counter -= 1/num_batches #keep going if differential low enough, even if loss > min
                    counter = max([0,counter]) #keep > 0
                    
            if tf.reduce_mean(loss) < best_loss:
                logging.info("Epoch {}/{}: Elapsed Time: {};Remaining Time estimate: {}; new best loss: {}; Patience: {} %".format(epoch,EPOCHS,datetime.now()-train_start,time_estimate_per_epoch*(EPOCHS-epoch),tf.reduce_mean(loss), 100*counter/counter_max))
                best_loss = tf.reduce_mean(loss)
                best_model = tf.keras.models.clone_model(model)
                #best_model.save(".\\models\\Run_{}\\best_model".format(run_id))
                best_model.save_weights(".\\models\\weights\\Run_{}\\Run".format(run_id))
                counter = 0
        #append epoch loss
        losses.append(tf.reduce_mean(loss))
        #calculate mse
        pi_tt,mu_tt,var_tt = best_model.predict(np.asarray(X_tt))
        sample_preds, sample_probability_array = generate_tensor_mixture_model(t_a_r,pi_tt,mu_tt,var_tt)
        #sample_preds,sample_probability_array,_ = sample_predictions_tf_r(t_a_r,pi_tt,mu_tt,var_tt)
        profile_sample = sample_preds[0]
        mse_error_profiles = tf.losses.MSE(ttp_renormed[0],profile_sample)
        MSEs.append(mse_error_profiles)
        
        
        mae_error_profiles_test = tf.losses.mean_absolute_error(profile_sample,ttp_renormed[0])
        test_MAEs.append(mae_error_profiles_test)
        
        
        #calculate overlap
        source_overlap = np.dot(np.transpose(ttp_renormed[0]),ttp_renormed[0]) #ignore constant multiplier
        
        overlap_ratio = source_overlap/tf.reduce_sum(profile_sample**2)
        overlap_ratios.append(overlap_ratio)
        
        
        counters.append(counter)
        
        training_bool = epoch in range(EPOCHS)
        
        loss_break = (best_loss.numpy() < loss_target)# and (np.exp(-best_loss.numpy()) > likelihood_minimum) #equivalent frankly, just redundant
        loss_break = loss_break or (diff < 0) 
        training_bool = (epoch <= EPOCHS or not loss_break) if (counter//counter_max < 1) else False
        time_estimate_per_epoch = (datetime.now()-train_start)/epoch
        if epoch % print_every == 0:
            logging.info('Epoch {}/{}: Elapsed Time: {};Remaining Time estimate: {}; loss {}, Patience: {} %; MSE: {}; overlap: {}'.format(epoch, EPOCHS,datetime.now() - train_start,time_estimate_per_epoch*(EPOCHS-epoch), losses[-1],100*counter/counter_max,mse_error_profiles, overlap_ratio))       
        epoch = epoch+1
    
    logging.info("Training completed after {}/{} epochs. Patience: {} %:: Best Loss: {}".format(epoch, EPOCHS, 100*counter/counter_max, best_loss))
    logging.info("Reason for exiting: loss_break: {}, diff < 0: {}".format(loss_break,diff<0))
    score_file = "./scores.csv"
    logging.info("Saving best score {} to {}".format(best_loss,score_file))
    
    score_df = pd.read_csv(score_file)
    score_df["MAE"][run_id] = best_loss
    score_df.to_csv(score_file)
    
    plot_folder = ".\\plots\\Run_{}\\".format(run_id)
    save_folder = ".\\models\\Run_{}\\best_model".format(run_id)
    data_folder = ".\\data\\Run_{}\\".format(run_id)
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
    
    
    
    '''
    
    logging.info("Saving to plot folder: {}".format(plot_folder))
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    
    now = datetime.now()
    #plt.figure()
    plt.plot(losses)
    plt.title("MAE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("NAE Loss")
    plt.savefig(plot_folder+"Losses_{}_{}_{}.png".format(now.hour,now.day,now.month))
    #plt.close("all")
    plt.cla()
    
    #plt.figure()
    plt.plot(counters)
    plt.title("Absolute Patience")
    plt.xlabel("Epoch")
    plt.ylabel("Patience")
    plt.savefig(plot_folder+"Counter_{}_{}_{}.png".format(now.hour,now.day,now.month))
    #plt.close("all")
    plt.cla()
    
    #plt.figure()
    plt.plot(MSEs)
    plt.title("MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Pseudo MSE")
    plt.savefig(plot_folder+"MSE_{}_{}_{}.png".format(now.hour,now.day,now.month))
    plt.cla()
    #plt.close("all")
    
    #plt.figure()
    plt.plot(overlap_ratios)
    plt.title("Profile Overlap Ratios true/generated")
    plt.xlabel("Epoch")
    plt.ylabel("Overlap")
    plt.savefig(plot_folder+"Overlap_{}_{}_{}.png".format(now.hour,now.day,now.month))
    #plt.close("all")
    plt.cla()
    '''
    n_test_profiles = 10
    test_profiles,t_profile_params,t_associated_r = EinastoSim.generate_n_random_einasto_profile_maggie(n_test_profiles)
    t_sample_profiles_logged = np.asarray([np.log(p) for p in test_profiles]).astype(np.float64)
    t_s_reparam = reparameterizer(t_sample_profiles_logged)
    t_s_renorm = t_s_reparam.calculate_parameterization().astype(np.float64)
    X_test = t_profile_params#create_input_vectors(t_profile_params,t_associated_r)
    
    pi_test, mu_test,var_test = best_model.predict(np.asarray(X_test))
    
    test_data = {"Profiles":t_s_renorm, "STDParams":{"Pi":pi_test,"Mu":mu_test,"Var":var_test},"Xtest":X_test, "r":t_associated_r}
    with open(data_folder+"test_data.png","wb") as f:
        pickle.dump(test_data,f)
    
    
    sample_preds, sample_probability_array = generate_tensor_mixture_model(t_associated_r, pi_test,mu_test, var_test)
    #sample_preds, sample_probability_array,sample_stat_inputs = sample_predictions_tf_r(t_associated_r,pi_test,mu_test,var_test)
    '''
    for i in range(n_test_profiles):
        profile_sample = sample_preds[i,:]
        test_prof = t_s_renorm[i]
        #plt.figure()
        plt.plot(t_associated_r[i],test_prof,label = "True profile")
        
        logging.debug("Parameters for {}: {}".format(i,EinastoSim.print_params_maggie(t_profile_params[i])))
        #mng = plt.get_current_fig_manager()
        
        plt.plot(t_associated_r[i],profile_sample, label = "Sample")
        #probability_arr = [pi_test[i][kd]/(tf.sqrt(2*np.pi*var_test[i][kd]))*tf.exp(-(1/(2*var_test[i][kd]))*((t_associated_r[i]-mu_test[i][kd])**2)) for kd in range(k)]
        #constituent_probabilities = tf.stack(probability_arr)
        for kd in range(k):
            plt.plot(t_associated_r[i],sample_probability_array[i][kd],label = "Sampled Constituent {}".format(kd)) #plotting probabilities found in the method
            #plt.plot(t_associated_r[i],constituent_probabilities[kd], label = "Constituent {}".format(kd))
        #plt.plot(t_associated_r[i],tf.add_n(probability_arr),label = "Profile Addition")
        plt.legend()
        plt.title(EinastoSim.print_params_maggie(t_profile_params[i]).replace("\t",""))
        plt.xlabel("Radius [Mpc]")
        plt.ylabel("log({}) []".format(u"\u03C1"))
        #mng.full_screen_toggle()
        #plt.show()
        #plt.pause(1e-3)
        plt.savefig(plot_folder+"Sample_profiles_{}_{}_{}_{}_{}.png".format(run_id,i,now.hour,now.day,now.month))
        plt.cla()
        #plt.close("all")
    '''
    with open(run_file,"w") as f:
        f.write(str(run_id +1))