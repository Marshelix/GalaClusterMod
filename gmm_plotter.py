# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 09:36:12 2020

@author: Martin Sanner
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from normal_dist_calculator import generate_tensor_mixture_model
import pickle
from datetime import datetime
import os
import EinastoSim

if __name__ == "__main__":
    
    mode = ""
    while mode not in ["density","normal","d","n"]:
        mode = input("Input one of {}:".format(["density","normal","d","n"]))
    
    run_file = "./runID.txt"
    run_id = -1
    if not os.path.isfile(run_file):
        with open(run_file,"w") as f:
            run_id = 1
            f.write(str(run_id))
            
    else:
        with open(run_file,"r") as f:
            run_id = int(f.read()) -1 #do this for the latest available files
    if mode in ["density","d"]:
        data_folder = ".//data//Run_{}//".format(run_id)
    else:
        data_folder = ".//data//gauss//"
    
    
    with open(data_folder+"MAE_Losses.dat","rb") as f:
        losses = pickle.load(f)
    with open(data_folder+"MSE_Losses.dat","rb") as f:
        MSEs = pickle.load(f)
    with open(data_folder+"Patience.dat","rb") as f:
        counters = pickle.load(f)
    #with open(data_folder+"overlap.dat","rb") as f:
    #    overlap_ratios = pickle.load(f)
    
    with open(data_folder+"mae_test_losses.dat","rb") as f:
        test_MAEs = pickle.load(f)
    
    plot_folder = ".//plots//Run_{}//".format(run_id) if mode in ["d","density"] else ".//plots//gauss//"
    
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    
    
    plt.figure()
    plt.plot(losses, label = "Training Error")
    plt.plot(test_MAEs, label = "Testing Error")
    plt.title("MAE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MAE Loss")
    plt.legend()
    plt.savefig(plot_folder+"MAE_Losses.png")
    plt.close("all")
    
    plt.figure()
    plt.plot(counters)
    plt.title("Patience")
    plt.xlabel("Epoch")
    plt.ylabel("Patience [%]")
    plt.savefig(plot_folder+"Patience_Counter.png")
    plt.close("all")
    
    plt.figure()
    plt.plot(MSEs)
    plt.title("MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Pseudo MSE")
    plt.savefig(plot_folder+"MSE_Losses.png")
    plt.close("all")
    
    #plt.figure()
    #plt.plot(overlap_ratios)
    #plt.title("Profile Overlap Ratios true/generated")
    #plt.xlabel("Epoch")
    #plt.ylabel("Overlap")
    #plt.savefig(plot_folder+"Overlap_Ratios.png")
    #plt.close("all")
    
    with open(data_folder + "test_data.dat","rb") as f:
        test_data = pickle.load(f)
    t_s_renorm = test_data["Profiles"]
    pi_test = test_data["STDParams"]["Pi"]
    mu_test = test_data["STDParams"]["Mu"]
    var_test = test_data["STDParams"]["Var"]
    X_test = test_data["Xtest"]
    n_test_profiles,k = pi_test.shape
    t_associated_r = test_data["r"]
    
    sample_preds, sample_probability_array = generate_tensor_mixture_model(t_associated_r, pi_test,mu_test, var_test)
    
    for i in range(n_test_profiles):
        profile_sample = sample_preds[i,:]
        test_prof = t_s_renorm[i]
        plt.figure()
        plt.plot(t_associated_r[i],test_prof,label = "True profile")
        
        mng = plt.get_current_fig_manager()
        
        plt.plot(t_associated_r[i],profile_sample, label = "Sample")
        for kd in range(k):
            plt.plot(t_associated_r[i],sample_probability_array[i][kd],label = "Sampled Constituent {}: pi: {}; mu: {}; var {}".format(kd,pi_test[i][kd],mu_test[i][kd],var_test[i][kd])) #plotting probabilities found in the method
            
        plt.legend()
        #plt.title(EinastoSim.print_params_maggie(X_test[i]).replace("\t",""))
        plt.xlabel("Radius [Mpc]")
        plt.ylabel("log({}) []".format(u"\u03C1"))
        mng.full_screen_toggle()
        plt.show()
        plt.pause(1e-1)
        plt.savefig(plot_folder+"Sample_profiles_{}.png".format(i))
        plt.cla()
        plt.close("all")