# -*- coding: utf-8 -*-
"""
Created on Sun May 24 10:07:00 2020

@author: Martin Sanner

following https://github.com/llSourcell/Gaussian_Mixture_Models/blob/master/intro_to_gmm_%26_em.ipynb for initial design
"""

import numpy as np
import matplotlib.pyplot as plt
import EinastoSim
import h5py


sample_profiles,profile_params = EinastoSim.generate_n_profiles(1000)
EinastoSim.print_params(profile_params[0])

class gmm:
    def __init__(self, num_statistics = 4, num_epochs = 10, lr = 1e-3,r_granularity = EinastoSim.r_gran_max,rmax = EinastoSim.rmax_glob,rmin = EinastoSim.rmin_glob):
        self.num_statistics =  num_statistics
        self.num_epochs = num_epochs
        self.lr = lr
        self.num_input_dims = 1 #only input: radius 
        #for each input, create ns statistics, with 3 param sets each, mu, std^2, w
        self.group_dims = 2*self.num_input_dims+1
        self.num_output_dims = int(self.num_statistics*(self.group_dims)) 
        self.parameters = [EinastoSim.error_epsi for i in range(self.num_output_dims)]
        '''
            Parameters are grouped as follows: [weight_i, mu_bar_i, stdsqr_bar_i] 
            where i refers to the ith group, and _bar signifies a vector
        '''
        self.best_log_likelihood = -np.inf
        self.c_log_like = 0.0
        
        self.radii = np.linspace(rmin,rmax,r_granularity)
        #set mu, sigma correctly
        for i in range(self.num_statistics):
            ind = i*self.group_dims
            mu = np.mean(self.radii)+EinastoSim.error_epsi*(2*np.random.random()-0.5)
            std = np.cov(self.radii)+EinastoSim.error_epsi*(2*np.random.random()-0.5)#fluctuate around mean
            self.parameters[ind+1] = mu
            self.parameters[ind+2] = std
            self.parameters[ind] = 1/self.num_statistics
        self.best_params = self.parameters
    def get_grouped_params(self):
        return np.asarray(self.parameters).reshape((self.num_statistics, int(self.group_dims)))
    def get_grouped_param_vecs(self):
        '''
        return normalized weights, vector of vector of means and vector of vector of stds
        '''
        grouped_params = self.get_grouped_params()
        weights = []
        mu_vecs = []
        std_vecs = []
        for i in range(self.num_statistics):
            group = grouped_params[i]
            weights.append(group[0])
            mu = group[1:(2+self.num_input_dims)]
            std = group[2+self.num_input_dims:]
            mu_vecs.append(mu)
            std_vecs.append(std)
        return np.asarray(weights)/np.sum(weights), mu_vecs, std_vecs
            
    def calculate_covariance_matrices(self):
        _,_,stds = self.get_grouped_param_vecs()
        cov_matrices = []
        for std_group in stds:
            cov_matrices.append(np.diag(std_group))
        return cov_matrices
    def pdf(self,input_sample):
        '''
        return the sum of the weighted Normals
        '''
        weights, mu_vecs,_ = self.get_grouped_param_vecs()
        cov_mats = self.calculate_covariance_matrices()
        pdf_sum = 0
        for i in range(self.num_statistics):
            cov_current = cov_mats[i]
            mu = mu_vecs[i]
            w = weights[i]
            cov_inv = np.linalg.inv(cov_current)
            det_cov = np.linalg.det(cov_current)
            prefactor = 1/((2*np.pi)**(self.num_input_dims/2)*np.sqrt(det_cov))
            pdf_sum += w*prefactor*np.exp(-(1/2)*np.dot(np.transpose(input_sample-mu),np.dot(cov_inv,(input_sample-mu))))
        return pdf_sum
    def create_pdf(self):
        pdf = []
        for r in self.radii:
            pdf.append(self.pdf(r))
        return pdf
    
    
    def estimation_step(self,data):
        self.c_log_like = 0.0
        weights, mu_vecs,_ = self.get_grouped_param_vecs()
        cov_mats = self.calculate_covariance_matrices()
        for d in data:
            denominator = 0.0
            weight_stats = []
            for current_stat in range(self.num_statistics):
                w = weights[current_stat]
                cov = cov_mats[current_stat]
                mu = mu_vecs[current_stat]
                cov_inv = np.linalg.inv(cov)
                det_cov = np.linalg.det(cov)
                prefactor = 1/((2*np.pi)**(self.num_input_dims/2)*np.sqrt(det_cov))
                c_pdf = w*prefactor*np.exp(-(1/2)*np.dot(np.transpose(d-mu),np.dot(cov_inv,(d-mu))))
                denominator += c_pdf
                weight_stats.append(c_pdf)
            weight_stats = np.asarray(weight_stats)*np.linalg.inv(denominator)
            self.c_log_like += np.log(np.sum(weight_stats))
        return weight_stats
    def maximization_step(self,data):
        #have weighted stats: 1d array detailling how relevant each statistic is
        den = np.sum(self.weighted_stats)
        for i in range(self.num_statistics):
            #mu = <wN,f>/den
            mu = np.dot(self.weighted_stats,data)/den
            diff = data - mu
            self.parameters[i+1] = mu
            self.parameters[i+2] = np.dot(np.multiply(self.weighted_stats,diff),diff)/den
            self.parameters[i] = den
        
        
    def fit_data(self,dataset):
        n,m = dataset.shape        
        current_epoch = 0
        current_best_epoch = 0
        #train until either convergence, arrival on target or out of epochs
        training_requirement = ((current_epoch < self.num_epochs) and (current_epoch - current_best_epoch <= self.num_epochs//5)) and self.best_log_likelihood > 1e-5
        while training_requirement:
            current_epoch += 1
            
            self.c_log_like = np.sum(np.log(self.create_pdf()))
            for data in dataset:
                self.weighted_stats = self.estimation_step(np.multiply(self.radii,data))
                self.parameters = self.maximization_step(np.multiply(self.radii,data))
            if self.c_log_like <  self.best_log_likelihood:
                self.best_log_likelihood = self.c_log_like
                current_best_epoch = current_epoch

g = gmm()
