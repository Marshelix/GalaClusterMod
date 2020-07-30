# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:01:49 2020

@author: Martin Sanner
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys

import logging
from normal_dist_calculator import generate_vector_gauss_mixture,generate_vector_random_gauss_mixture
from datetime import datetime

import pickle
from sklearn.model_selection import train_test_split
import argparse
np.random.seed(42)
tf.random.set_seed(42)



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
    '''
    Restricted float from https://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-between-0-0-1-0-usin
    '''
    def restricted_float(x):
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))
        return x
    '''
    Argument defaults
    '''
    def_num_profiles = 5000
    def_train_ratio = 0.5
    def_lr  = 1e-4
    def_k = 8
    def_kg = 4
    def_epochs = 2000
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_profile",type = int,default = def_num_profiles, help = "Number of profiles - default {}".format(def_num_profiles))
    parser.add_argument("--train_ratio",type = restricted_float, default = def_train_ratio, help = "Ratio of training to test samples - default {}".format(def_train_ratio))
    parser.add_argument("--lr",type = restricted_float,default = def_lr,help = "Learning rate - default {}".format(def_lr))
    parser.add_argument("--k",type = int, default = def_k, help = "k - default {}".format(def_k))
    parser.add_argument("--kg",type = int, default = def_kg, help = "k-generator - default {}".format(def_kg))
    parser.add_argument("--epochs",type = int, default = def_epochs, help = "Epochs - default {}".format(def_epochs))
    args = parser.parse_args()
    
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
    
    num_profiles = args.num_profile
    kg = args.kg
    r = np.linspace(-10,10,1001)
    rs = [r for i in range(num_profiles)]
    logging.info("Generating {} normals based on {} distribution for training".format(num_profiles, kg))
    generated_gaussians, parameters, gaussMixtures = generate_vector_random_gauss_mixture(rs,kg)
    gaussians,test_gaussians,X_full,X_tt = train_test_split(generated_gaussians,parameters,test_size = float(args.train_ratio))
    logging.info("Defining backend type: TF.float64")
    tf.keras.backend.set_floatx("float64")
    X_full = np.asarray(X_full).astype(np.float64)
    
    losses = []
    EPOCHS = args.epochs
    l = len(X_full[0])
    
    #output dimension
    out_dim = 1
    # Number of gaussians to represent the multimodal distribution
    k = args.k
    
    logging.info("Running {} dimensions on {} distributions".format(out_dim,k))
    '''
    Define model manually
    
    '''
    lr = args.lr
    n_hid_1 = 30
    n_hid_2 = 30
    # first layer
    w1 = tf.Variable(tf.random.normal(shape = (n_hid_1,l),dtype = tf.float64),dtype = tf.float64)
    b1 = tf.Variable(tf.random.normal(shape = [n_hid_1,1],dtype = tf.float64),dtype = tf.float64)
    # second layer
    w2 = tf.Variable(tf.random.normal(shape = [n_hid_2,n_hid_1],dtype = tf.float64))
    b2 = tf.Variable(tf.random.normal(shape = [n_hid_2,1],dtype = tf.float64))
    #output
    out_w = tf.Variable(tf.random.normal(shape = [k*out_dim*3,n_hid_2],dtype = tf.float64))
    out_b = tf.Variable(tf.random.normal(shape = [k*out_dim*3,1],dtype = tf.float64))
    
    
    best_loss = tf.cast(np.inf,tf.float64)
    best_w1 = tf.identity(w1)
    best_w2 = tf.identity(w2)
    best_outw = tf.identity(out_w)
    best_b1 = tf.identity(b1)
    best_b2 = tf.identity(b2)
    best_outb = tf.identity(out_b)
    
    def run_network(in_batch,network_nodes = (best_w1,best_b1,best_w2,best_b2,best_outw,best_outb)):
        best_w1,best_b1,best_w2,best_b2,best_outw,best_outb = network_nodes
        net_input = tf.constant(in_batch, shape = [l, len(in_batch)])#input layer
        l1_output = tf.nn.tanh(tf.linalg.matmul(best_w1,net_input))+best_b1
        l2_output = tf.nn.tanh(tf.linalg.matmul(best_w2,l1_output))+best_b2
        network_out_vector = tf.linalg.matmul(best_outw,l2_output)+best_outb
        return network_out_vector
    
    def model(in_batch,network_nodes = (best_w1,best_b1,best_w2,best_b2,best_outw,best_outb)):
        return_vector = tf.transpose(run_network(in_batch,network_nodes))
        pi_max_base,mu_,var_log = tf.split(return_vector,3,1)
    
        pi_un = pi_max_base - tf.reduce_max(pi_max_base,1,True)
        pi_ = tf.nn.softmax(pi_un)
        var_ = tf.exp(var_log)
        return pi_,mu_,var_
        
    optimizer = tf.optimizers.Adam(lr)
    '''
    Create Dataset
    '''
    N = np.asarray(X_full).shape[0]
    num_batches = 1
    batchsize = N//num_batches
    dataset = tf.data.Dataset \
    .from_tensor_slices((X_full, gaussians)) \
    .shuffle(N).batch(batchsize)
    '''
    Create training parameters
    '''
    
    
    n_test_profiles = args.train_ratio*num_profiles
    start_parameters = {}
    counter_max = start_parameters.get("Counter_max",500)
    
    loss_target = start_parameters.get("loss_target",1e-3)
    
    max_diff = start_parameters.get("max_diff",0.0)  #differential loss
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
    min_train_epochs_pre_div = 100
    rolling_mean_length = 5
    
    logging.info("="*10+"Training info"+"="*10)
    logging.debug('Print every {} epochs'.format(print_every))
    logging.info("Learning Parameters: lr = {}".format(lr))
    logging.info("Patience: {} increases".format(counter_max))
    logging.info("Minimum delta loss to not lose patience: {}".format(minimum_delta))
    logging.info("Target loss: < {}".format(loss_target))
    logging.info("# Samples: {}".format(n_test_profiles))
    logging.info("# Training Profiles: {}".format(num_profiles*(1-args.train_ratio)))
    logging.info("Printing every {} epochs".format(print_every))
    logging.info("Maximum loss divergence: {}".format(max_loss_divergence))
    logging.info("Maximum length values taken into account of mean calculation: {}".format(rolling_mean_length))
    logging.info("Minimum epochs before divergence is taken into account: {}".format(min_train_epochs_pre_div))
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
                #pi_, mu_, var_ = model(train_x,training = True)
                #prediction_vector = model(train_x,training = True)
                #pi_un,mu_,var_log = tf.split(prediction_vector,3,1)
                #pi_ = tf.nn.softmax(pi_un)
                #var_ = tf.exp(var_log)
                nodes = (w1,b1,w2,b2,out_w,out_b)
                pi_,mu_,var_ = model(train_x,nodes)
                sample, mixtures = generate_vector_gauss_mixture(rs,pi_,mu_,var_)#generate_tensor_mixture_model(rs,pi_,mu_,var_)    
                loss = tf.losses.MSE(train_y,sample)#tf.nn.l2_loss(sample-tf.cast(train_y,dtype = tf.float64))#tf.losses.mean_absolute_error(train_y,sample)
            gradients = tape.gradient(loss, [w1,b1,w2,b2,out_w,out_b])
            optimizer.apply_gradients(zip(gradients, [w1,b1,w2,b2,out_w,out_b]))
        
        losses.append(tf.reduce_mean(loss))
        #calculate mse
        nodes = (w1,b1,w2,b2,out_w,out_b)
        pi_tt,mu_tt,var_tt = model(np.asarray(X_tt),nodes)
        sample_preds, sample_mixtures = generate_vector_gauss_mixture(rs,pi_tt,mu_tt,var_tt)#generate_tensor_mixture_model(rs,pi_tt,mu_tt,var_tt)
        mse_error_profiles = tf.reduce_mean(tf.losses.MSE(test_gaussians,sample_preds))
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
                #best_model.save_weights(".\\models\\weights\\Run_{}\\Run".format(run_id))
                counter = 0
        '''
            Amend counter if: not better than the best loss, delta_loss < minimum_delta, delta_loss < max_diff(best delta so far)
            Reset counter if best loss overcome
            '''
        if mse_error_profiles > best_loss:
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
        
        
        if len(MSEs) > 1:
            tmae_diffs = np.asarray(MSEs[1:])-np.asarray(MSEs[:-1])
            max_tmae_idx = min(len(MSEs),rolling_mean_length)
            avg_test_loss_diff = np.mean(MSEs[-max_tmae_idx:])
        if len(losses) > 1:
            mae_diffs = np.asarray(losses[1:])-np.asarray(losses[:-1])
            max_mae_idx = min(len(mae_diffs),rolling_mean_length)
            avg_train_loss_diff = np.mean(mae_diffs[-max_mae_idx:])
        
        training_bool = epoch in range(EPOCHS)
        
        loss_break = (best_loss.numpy() < loss_target)
        loss_break = loss_break or (diff < 0) 
        
        loss_divergence = abs(tf.reduce_mean(loss)-tf.cast(mse_error_profiles,dtype = tf.float64)) > max_loss_divergence if epoch > min_train_epochs_pre_div else False
        '''
        Continue training if epochs left or if current best loss is worse than the target
        Stop training if training/test losses diverge or if patience lost
        '''
        training_bool = (epoch <= EPOCHS or not loss_break) if ((counter//counter_max < 1) and not loss_divergence) else False
        
        time_estimate_per_epoch = (datetime.now()-train_start)/epoch
        if epoch % print_every == 0:
            logging.info('Epoch {}/{}: Elapsed Time: {};Remaining Time estimate: {}; loss = {}, test MAE = {};loss delta: {};test loss delta: {}; Patience: {} %; MSE: {};'.format(epoch, EPOCHS,datetime.now() - train_start,time_estimate_per_epoch*(EPOCHS-epoch), losses[-1],mae_error_profiles_test,avg_train_loss_diff,avg_test_loss_diff,100*counter/counter_max,mse_error_profiles))       
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
    rs = [r for i in range(n_test_profiles)]
    test_gauss,test_params,generators = generate_vector_random_gauss_mixture(rs,kg)#EinastoSim.generate_n_k_gaussian_parameters(rs,n_test_profiles,kg)
    #test_gauss = np.asarray([np.log(p) for p in test_gauss])
    X_test = test_params
    best_nodes = (best_w1,best_b1,best_w2,best_b2,best_outw,best_outb)
    pi_test,mu_test,var_test = model(np.asarray(X_test),best_nodes)
    #pi_test_base, mu_test,var_test_log = tf.split(best_model.predict(np.asarray(X_test)),3,1)
    #pi_test = tf.nn.softmax(pi_test_base)#tf.sigmoid(pi_test_base)
    #var_test = tf.exp(var_test_log)
    sample_preds, sample_mixtures = generate_vector_gauss_mixture(rs,pi_test,mu_test,var_test)#generate_tensor_mixture_model(rs, pi_test,mu_test, var_test)
    test_data = {"Profiles":test_gauss, "STDParams":{"Pi":pi_test,"Mu":mu_test,"Var":var_test},"Xtest":X_test, "r":rs}
    with open(data_folder+"test_data.dat","wb") as f:
        pickle.dump(test_data,f)
    with open(run_file,"w") as f:
        f.write(str(run_id +1))