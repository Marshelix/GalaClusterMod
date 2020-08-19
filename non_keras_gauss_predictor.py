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

import trainingAddons as trad
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
    def_num_profiles = 2000
    def_train_ratio = 0.5
    def_lr  = 1e-3
    def_k = 4
    def_kg = 1
    def_epochs = 100
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
    logging.info("Gauss Run {}".format(run_id))
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
    n_hid_1 = 20
    n_hid_2 = 20
    
    initial_nodes,best_nodes = trad.create_initial_nodes(l,n_hid_1,n_hid_2,k,out_dim)
      
    optimizer = tf.optimizers.Adam(lr)#tf.optimizers.Adadelta(lr)#tfa.optimizers.AdamW(lr,wd)
    logging.info("Training with optimizer: {}".format(optimizer.__class__.__name__))
    '''
    Create Dataset
    '''
    N = np.asarray(X_full).shape[0]
    num_batches = 10
    batchsize = N//num_batches
    logging.info("Employing {} batches with size {}".format(num_batches,batchsize))
    dataset = tf.data.Dataset \
    .from_tensor_slices((X_full, gaussians)) \
    .shuffle(N).batch(batchsize)
    
    '''
    Initialize training
    '''
    
    start_parameters = {"minDelta":1e-5}
    min_epoch_train_pre_div = 100
    normalize = True
    max_loss_divergence = 10
    patience_disabled = False
    wd = 0
    best_nodes,losses,MSEs,counters,test_MAEs = trad.train_model(initial_nodes,
                                                                 optimizer,
                                                                 dataset,
                                                                 rs,
                                                                 EPOCHS,
                                                                 X_tt,
                                                                 test_gaussians,
                                                                 rs,
                                                                 min_epoch_train_pre_div,
                                                                 start_parameters,
                                                                 5,
                                                                 lr,
                                                                 normalize,
                                                                 max_loss_divergence,
                                                                 patience_disabled,
                                                                 wd)
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
    X_test = test_params
    pi_test,mu_test,var_test = trad.model(np.asarray(X_test),best_nodes,True)
    sample_preds, sample_mixtures = generate_vector_gauss_mixture(rs,pi_test,mu_test,var_test)#generate_tensor_mixture_model(rs, pi_test,mu_test, var_test)
    test_data = {"Profiles":test_gauss, "STDParams":{"Pi":pi_test,"Mu":mu_test,"Var":var_test},"Xtest":X_test, "r":rs}
    with open(data_folder+"test_data.dat","wb") as f:
        pickle.dump(test_data,f)
    with open(run_file,"w") as f:
        f.write(str(run_id +1))