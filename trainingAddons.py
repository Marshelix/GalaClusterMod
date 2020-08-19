# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:00:18 2020

@author: Martin Sanner
Training functions file
exports general training behaviour here, eg def train()
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys

import logging
from normal_dist_calculator import normalMixtureCalculator,generate_vector_gauss_mixture,generate_vector_random_gauss_mixture,generate_tensor_mixture_model
from datetime import datetime

np.random.seed(42)
tf.random.set_seed(42)
#plt.close("all")

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


def create_initial_nodes(l,n_hid_1,n_hid_2,k,out_dim):
    # first layer
    w1 = tf.Variable(tf.random.normal(shape = (n_hid_1,l),dtype = tf.float64),dtype = tf.float64)
    b1 = tf.Variable(tf.random.normal(shape = [n_hid_1,1],dtype = tf.float64),dtype = tf.float64)
    # second layer
    w2 = tf.Variable(tf.random.normal(shape = [n_hid_2,n_hid_1],dtype = tf.float64))
    b2 = tf.Variable(tf.random.normal(shape = [n_hid_2,1],dtype = tf.float64))
    #output
    out_w = tf.Variable(tf.random.normal(shape = [k*out_dim*3,n_hid_2],dtype = tf.float64))
    out_b = tf.Variable(tf.random.normal(shape = [k*out_dim*3,1],dtype = tf.float64))
    num_parameters = n_hid_1*(l+1)+n_hid_2*(n_hid_1+1)+k*out_dim*3*(n_hid_2+1)
    logging.info("Number of trainable parameters in model: {}".format(num_parameters))
    
    
    best_w1 = tf.identity(w1)
    best_w2 = tf.identity(w2)
    best_outw = tf.identity(out_w)
    best_b1 = tf.identity(b1)
    best_b2 = tf.identity(b2)
    best_outb = tf.identity(out_b)
    best_nodes = (best_w1,best_b1,best_w2,best_b2,best_outw,best_outb)
    initial_nodes = (w1,b1,w2,b2,out_w,out_b)
    return initial_nodes,best_nodes

def run_network(in_batch,network_nodes):
    '''
    Runs a batch of data through the nodes specified in network nodes. 3 layer network.
    '''
    best_w1,best_b1,best_w2,best_b2,best_outw,best_outb = network_nodes
    l = best_w1.shape[1]
    net_input = tf.constant(in_batch, shape = [l, len(in_batch)])#input layer
    l1_output = tf.nn.relu(tf.linalg.matmul(best_w1,net_input))+best_b1
    l2_output = tf.nn.tanh(tf.linalg.matmul(best_w2,l1_output))+best_b2
    network_out_vector = tf.linalg.matmul(best_outw,l2_output)+best_outb
    return network_out_vector

def model_norm(in_batch,network_nodes):
    '''
    Calculates the pi, mu, var parameters for a given network and batch. 3 layer network
    '''
    return_vector = tf.transpose(run_network(in_batch,network_nodes))
    pi_max_base,mu_,var_log = tf.split(return_vector,3,1)
        
    pi_un = pi_max_base - tf.reduce_max(pi_max_base,1,True)
    pi_ = tf.nn.softmax(pi_un)
    var_ = tf.exp(var_log)
    return pi_,mu_,var_

def model_unnorm(in_batch,network_nodes):
    '''
    Calculates the pi, mu, var parameters for a given network and batch. 3 layer network
    Does not normalize pi.
    '''
    return_vector = tf.transpose(run_network(in_batch,network_nodes))
    pi_max_base,mu_,var_log = tf.split(return_vector,3,1)
    #try multiplying pi_max_base and var together?
    #pi_un = pi_max_base - tf.reduce_max(pi_max_base,1,True)
    #pi_ = tf.nn.softmax(pi_un)
    #square mean to get positive radii?
    mu_ = tf.cast(tf.square(mu_),tf.float64)
    var_ = tf.exp(var_log)
    #pi_max_base = tf.math.multiply(pi_max_base, tf.sqrt(var_))
    return pi_max_base,mu_,var_

def model(in_batch,network_nodes,normalized):
    '''
    Runs the model_norm or model_unnorm functions with its inputs based on the normalization
    Parameters
    ----------
    in_batch: vector of inputs
    network_nodes: Definition of model parameters, 3 layers assumed
    normalized: boolean, no default
    '''
    return model_norm(in_batch,network_nodes) if normalized else model_unnorm(in_batch,network_nodes)

def train_model(model_nodes, optimizer, dataset,initial_r,
                num_epochs,
                X_tt,test_gaussians,t_a_r,
                min_train_epochs = 100, start_parameters = {}, print_every_n_epoch = 5,lr = 1e-4,
                norm = True,max_divergence = 0.4,patience_disabled = False,l2_weight_decay = 1e-2):
    '''
    

    Parameters
    ----------
    model_nodes : TYPE
        DESCRIPTION.
    optimizer : TYPE
        DESCRIPTION.
    dataset : TYPE
        DESCRIPTION.
    initial_r : TYPE
        DESCRIPTION.
    num_epochs : TYPE
        DESCRIPTION.
    target_loss : TYPE
        DESCRIPTION.
    X_tt : TYPE
        DESCRIPTION.
    test_profiles : TYPE
        DESCRIPTION.
    t_a_r : TYPE
        DESCRIPTION.
    min_train_epochs : TYPE, optional
        DESCRIPTION. The default is 100.
    start_parameters : TYPE, optional
        DESCRIPTION. The default is {}.
    roll_mean_length : TYPE, optional
        DESCRIPTION. The default is 5.
    norm : Boolean, optional
        Description. Decides on whether pi should be normalized or not. Optional

    Returns
    -------
    None.

    '''
    best_loss = tf.cast(np.inf,tf.float64)
    
    w1,b1,w2,b2,out_w,out_b = model_nodes
    
    best_w1 = tf.identity(w1)
    best_w2 = tf.identity(w2)
    best_outw = tf.identity(out_w)
    best_b1 = tf.identity(b1)
    best_b2 = tf.identity(b2)
    best_outb = tf.identity(out_b)
    best_nodes = (best_w1,best_b1,best_w2,best_b2,best_outw,best_outb)
    
    counter_max = start_parameters.get("Counter_max",5000)
    
    loss_target = start_parameters.get("loss_target",1e-3)
    
    max_diff = start_parameters.get("max_diff",0.0)  #differential loss
    epoch = start_parameters.get("epoch",1)
    training_bool = epoch in range(num_epochs)
    
    counter = start_parameters.get("counter",0)
    print_every = print_every_n_epoch
    
    counters = start_parameters.get("counters", [])
    
    test_MAEs = start_parameters.get("test_MAEs",[])
    
    MSEs = start_parameters.get("MSEs",[])
    
    minimum_delta = start_parameters.get("minDelta",1e-6)
    diff = 0
    loss_break = False
    losses = start_parameters.get("losses",[])
    max_loss_divergence = max_divergence
    
    avg_train_loss_diff = 0
    avg_test_loss_diff = 0
    min_train_epochs_pre_div = min_train_epochs
    
    # test data consistent
    assert len(t_a_r) >= len(test_gaussians) and len(test_gaussians) == len(X_tt)
    train_length = len(initial_r) if len(initial_r) == len(X_tt) else len(initial_r)-len(X_tt)
    logging.info("="*10+"Training info"+"="*10)
    logging.debug('Print every {} epochs'.format(print_every))
    logging.info("Learning Parameters: lr = {} \t wd = {}".format(lr,l2_weight_decay))
    logging.info("Patience: {} increases".format(counter_max))
    logging.info("Minimum delta loss to not lose patience: {}".format(minimum_delta))
    logging.info("Target loss: < {}".format(loss_target))
    logging.info("# Samples: {}".format(len(X_tt)))
    logging.info("# Training Profiles: {}".format(train_length))
    logging.info("Maximum loss divergence: {}".format(max_loss_divergence))
    logging.info("Printing every {} epochs".format(print_every))
    logging.info("Minimum epochs before divergence is taken into account: {}".format(min_train_epochs_pre_div))
    logging.info("Training for {} epochs".format(num_epochs))
    logging.info("="*(33))
    train_start = datetime.now()
    logging.info("Starting training at: {}".format(train_start))
    time_estimate_per_epoch = np.inf
    loss_divergence = False
    with tf.GradientTape() as tape:
        tape.reset()
        
    while training_bool:
        for train_x, train_y in dataset:
            with tf.GradientTape() as tape:
                nodes = (w1,b1,w2,b2,out_w,out_b)
                pi_,mu_,var_ = model(train_x,nodes,norm)
                
                sample, mixtures = generate_tensor_mixture_model(initial_r,pi_,mu_,var_)
                #generate_vector_gauss_mixture(initial_r,pi_,mu_,var_)#generate_tensor_mixture_model(initial_r,pi_,mu_,var_)    
                loss = tf.losses.MSE(train_y,sample) + l2_weight_decay*(tf.nn.l2_loss(w1)+tf.nn.l2_loss(w2)+tf.nn.l2_loss(out_w))#tf.nn.l2_loss(sample-tf.cast(train_y,dtype = tf.float64))#tf.losses.mean_absolute_error(train_y,sample)
            gradients = tape.gradient(loss, [w1,b1,w2,b2,out_w,out_b])
            optimizer.apply_gradients(zip(gradients, [w1,b1,w2,b2,out_w,out_b]))
        
        losses.append(tf.reduce_mean(loss))
        #calculate mse
        nodes = (w1,b1,w2,b2,out_w,out_b)
        pi_tt,mu_tt,var_tt = model(np.asarray(X_tt),nodes,norm)
        sample_preds, sample_mixtures = generate_tensor_mixture_model(t_a_r,pi_tt,mu_tt,var_tt)#generate_vector_gauss_mixture(rs,pi_tt,mu_tt,var_tt)
        mse_error_profiles = tf.reduce_mean(tf.losses.MSE(test_gaussians,sample_preds))+l2_weight_decay*(tf.nn.l2_loss(w1)+tf.nn.l2_loss(w2)+tf.nn.l2_loss(out_w))
        MSEs.append(mse_error_profiles)
        
        #mae to compare to train loss
        mae_error_profiles_test = tf.cast(tf.reduce_mean(tf.losses.mean_absolute_error(sample_preds,test_gaussians)),tf.float64)
        test_MAEs.append(mae_error_profiles_test)
        
        if mse_error_profiles < best_loss:
                best_loss = tf.reduce_mean(mse_error_profiles)
                #best_model = tf.keras.models.clone_model(model)
                best_w1 = tf.identity(w1)
                best_w2 = tf.identity(w2)
                best_outw = tf.identity(out_w)
                best_b1 = tf.identity(b1)
                best_b2 = tf.identity(b2)
                best_outb = tf.identity(out_b)
                best_nodes = (best_w1,best_b1,best_w2,best_b2,best_outw,best_outb)
                #best_model.save_weights(".\\models\\weights\\Run_{}\\Run".format(run_id))
                counter = 0
        '''
            Amend counter if: not better than the best loss, delta_loss < minimum_delta, delta_loss < max_diff(best delta so far)
            Reset counter if best loss overcome
            '''
        if not patience_disabled:
            if mse_error_profiles > best_loss:
                counter += 1
            if len(losses) > 1:
                diff = np.asarray(MSEs)[-1] - np.asarray(MSEs)[-2]
                if abs(diff) < minimum_delta or diff < max_diff:
                    counter += 1
                elif diff > max_diff + minimum_delta:
                    max_diff = diff
                    counter -= 1 #keep going if differential low enough, even if loss > min
                    counter = max([0,counter]) #keep > 0
        counters.append(100*counter/counter_max) #counter percentage
        
        
        if len(MSEs) > 1:
            tmse_diffs = np.asarray(MSEs[1:])-np.asarray(MSEs[:-1])
            max_tmae_idx = min(len(tmse_diffs),print_every)
            avg_test_loss_diff = np.mean(tmse_diffs[-max_tmae_idx:])
        if len(losses) > 1:
            mae_diffs = np.asarray(losses[1:])-np.asarray(losses[:-1])
            max_mae_idx = min(len(mae_diffs),print_every)
            avg_train_loss_diff = np.mean(mae_diffs[-max_mae_idx:])
        
        training_bool = epoch in range(num_epochs)
        
        loss_break = (best_loss.numpy() < loss_target)
        loss_break = loss_break or (diff < 0) 
        
        loss_divergence = abs(tf.reduce_mean(loss)-tf.cast(mse_error_profiles,dtype = tf.float64)) > max_loss_divergence if epoch > min_train_epochs_pre_div else False
        '''
        Continue training if epochs left or if current best loss is worse than the target
        Stop training if training/test losses diverge or if patience lost
        '''
        training_bool = (epoch <= num_epochs or not loss_break) if ((counter//counter_max < 1) and not loss_divergence) else False
        
        time_estimate_per_epoch = (datetime.now()-train_start)/epoch
        if epoch % print_every == 0:
            logging.info('Epoch {}/{}: Elapsed Time: {};Remaining Time estimate: {}; loss = {}, test MAE = {};loss delta: {};test loss delta: {}; Patience: {} %; MSE: {};'.format(epoch, num_epochs,datetime.now() - train_start,time_estimate_per_epoch*(num_epochs-epoch), losses[-1],mae_error_profiles_test,avg_train_loss_diff,avg_test_loss_diff,100*counter/counter_max,mse_error_profiles))       
        epoch = epoch+1
    
    logging.info("Training completed after {}/{} epochs. Patience: {} %:: Best Loss: {}".format(epoch, num_epochs, 100*counter/counter_max, best_loss))
    logging.info("Reason for exiting: loss_break: {}, diff < 0: {}, loss divergence: {}".format(loss_break,diff<0,loss_divergence))
    
    return best_nodes,losses,MSEs,counters,test_MAEs
    
    