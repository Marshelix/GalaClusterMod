# -*- coding: utf-8 -*-
"""
Created on Sun May 24 10:07:00 2020

@author: modified from https://www.katnoria.com/mdn/ , a tutorial on tf2 gdns

Made to fit the Einasto profile data generated by Martin Sanner


List of outstanding issues:
    - Check sample predictions, appears to just return the last probability multiplied by its pi value
    - Implement plotting of constituent distributions
"""

import numpy as np
import matplotlib.pyplot as plt
import EinastoSim
import h5py




import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
#import tensorflow_addons as tfa #AdamW
import tensorflow_probability as tfp#normal dist
from copy import deepcopy
import sys
import  logging
from datetime import datetime

import pandas as pd

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


logging.info("Running on GPU: {}".format(tf.test.is_gpu_available()))
def calc_pdf(y, mu, var):
    """Calculate component density"""
    value = tf.subtract(y, mu)**2
    value = (1/tf.math.sqrt(2 * np.pi * var)) * tf.math.exp((-1/(2*var)) * value)
    return value

def pdf_np(y, mu, var):
    n = np.exp((-(y-mu)**2)/(2*var))
    d = np.sqrt(2 * np.pi * var)
    return n/d

def mdn_loss(y_true, pi, mu, var):
    """MDN Loss Function
    The eager mode in tensorflow 2.0 makes is extremely easy to write 
    functions like these. It feels a lot more pythonic to me.
    """
    
    #throw away first few y values
    out = calc_pdf(y_true, mu, var)
    # multiply with each pi and sum it
    out = tf.multiply(out, pi)
    out = tf.reduce_sum(out, 1, keepdims=True)
    out = -tf.math.log(out + 1e-10)
   # logging.debug(tf.reduce_mean(out))
    return tf.reduce_mean(out)


@tf.function
def train_step(model, optimizer, train_x, train_y):
    # GradientTape: Trace operations to compute gradients
    with tf.GradientTape() as tape:
        pi_, mu_, var_ = model(train_x, training=True)
        print(pi_,mu_,var_)
        # calculate loss
        sample = sample_predictions(pi_,mu_,var_,1)[:,0,0]
        loss = tf.losses.mean_absolute_error(train_y,sample)#mdn_loss(train_y, pi_, mu_, var_)
    # compute and apply gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def sample_predictions_tensorflow(pi_,mu_,var_,samples = 10):
    '''
    Implement the generation of samples
    '''
    n,k = pi_.shape
    out_dimensions = 1
    out = tf.zeros((n,samples,out_dimensions),dtype = tf.float64)
    out_l = tf.unstack(out)
    for i in range(n):
        for j in range(samples):
            for li in range(out_dimensions):
                for kdist in range(k):
                    print(mu_[i,kdist*(li+out_dimensions)],var_[i,kdist],pi[i][kdist])
                    loc = mu_[i,kdist*(li+out_dimensions)]
                    scale = tf.sqrt(var_[i,kdist])
                    dist = tfp.distributions.normal.Normal(loc,scale)
                    out_l[i][j][li] += pi_[i][kdist]*dist.sample(1)
    return tf.stack(out_l)

def sample_predictions(pi_vals, mu_vals, var_vals, samples=10):
    #print("Inputs: {},{},{}".format(pi_vals,mu_vals,var_vals)
    n, k = pi_vals.shape
    l_out = 1
    
    #pi_vals = pi_vals.numpy()
    #mu_vals = mu_vals.numpy()
    #var_vals = var_vals.numpy()
    # place holder to store the y value for each sample of each row
    out = np.zeros((n, samples, l_out))
    
    for i in range(n):
        for j in range(samples):
            # for each sample, use pi/probs to sample the index
            # that will be used to pick up the mu and var values
            #idx = np.random.choice(range(k), p=probs)
            for li in range(l_out):
                for kdist in range(k):
                    #print(mu_[i,kdist*(li+l_out)],var_[i][kdist],pi[i][kdist])
                    out[i,j,li] += pi_vals[i][kdist]*np.random.normal(mu_vals[i,kdist*(li+l_out)],np.sqrt(var_vals[i][kdist]))
                # Draw random sample from gaussian distribution
                #out[i,j,li] = np.random.normal(mu_vals[i, idx*(li+l_out)], np.sqrt(var_vals[i, idx]))
    return out

def sample_predictions_tf_r(r_values, pi_vals, mu_vals, var_vals):
    n,k = pi_vals.shape
    assert n == len(r_values), "R value length does not match statistic values"
    final_array = []
    probability_array = [[] for pc in range(n)]
    for i in range(n):
        prob = [pi_vals[i,kd]/(tf.sqrt(2*np.pi*var_vals[i][kd]))*tf.exp(-(1/2*var_vals[i][kd])*((r_values[i]-mu_vals[i][kd])**2)) for kd in range(k)]
        probability_array[i] = prob
        # 4xn distribution list
        final_dist = tf.add_n(prob)
        final_array.append(final_dist)
    return tf.stack(final_array),probability_array




def mse_error(y_true, y_sample):
    '''
    Generates an average MSE fit between a true distribution and a sequence of samples
    '''
    if np.asarray(y_true.shape).size > 1:
        n,l = y_true.shape
    else:
        n = len(y_true)
    n2, num_samples = y_sample.shape
    assert n2 == n, "Sample has different granularity from original sequence"
    error = np.sum([np.dot(np.transpose(y_true-y_sample[:,k]),y_true-y_sample[:,k]) for k in range(num_samples)])
    error /= num_samples
    return error




#fixed sigma activation
# taken from https://github.com/cpmpercussion/keras-mdn-layer/blob/master/mdn/__init__.py
def elu_plus(x):
    return tf.keras.activations.elu(x)+1

def calculate_profile_minmax_param(profiles):
    params = [[min(p),max(p)] for p in (profiles)]
    return params

def calculate_renorm_profiles(profiles, profile_params = None):
    if profile_params == None:
        profile_params = calculate_profile_minmax_param(profiles)
    assert len(profile_params) == len(profiles)
    out = [(profiles[i] - profile_params[i][0])/(profile_params[i][1]-profile_params[i][0]) for i in range(len(profiles))]
    return out

def plot_constituent_profiles(associated_r,pi_val,mu_val,var_val):
    n,k = pi_val.shape
    sample,probability_array = sample_predictions_tf_r(associated_r,pi_val,mu_val,var_val)[0,:]
    i = 0
    prob = tf.stack([pi_val[i,kd]/(tf.sqrt(2*np.pi*var_val[i][kd]))*tf.exp(-(1/2*var_val[i][kd])*((associated_r[i]-mu_val[i][kd])**2)) for kd in range(k)])
    plt.figure()
    plt.plot(associated_r[i],sample,label = "Sample")
    for kd in range(k):
        plt.plot(associated_r[i],prob[kd], label = "Constituent {}".format(kd))
    plt.legend()
    
    
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
    num_profile_train = 2000
    logging.info("Generating {} Profiles for training".format(num_profile_train))
    
    sample_profiles,profile_params,associated_r = EinastoSim.generate_n_random_einasto_profile_maggie(num_profile_train)
    
    logging.info("Defining backend type: TF.float64")
    tf.keras.backend.set_floatx("float64")
    logging.info("Running Logged renormalized Profiles.")
    
    # remove log as test
    sample_profiles_logged = np.asarray([np.log(p) for p in sample_profiles]).astype(np.float64)
    sample_profiles_renormed = np.asarray(calculate_renorm_profiles(sample_profiles_logged)).astype(np.float64)
    #EinastoSim.print_params(profile_params[0])
    def create_input_vectors(profile_params, assoc_r):
        assert len(assoc_r) == len(profile_params), "mismatch between parameter and r lengths"
        N = len(assoc_r)
        logging.info("Generating {} elements from {} pairs".format(len(assoc_r[0])*N,N))
        input_vecs = []
        for i in range(N):
            base_vec = profile_params[i].copy()
            current_vec = []
            for j in range(len(assoc_r[i])):
                current_vec = base_vec.copy()
                r = assoc_r[i][j]
                current_vec.append(r)
                assert len(current_vec) == (len(base_vec) +1), "Appended more than one element"
                input_vecs.append(current_vec)
        return input_vecs
    
    X_full = profile_params#create_input_vectors(profile_params, associated_r) 
    X_full = np.asarray(X_full).astype(np.float64)
    l = len(profile_params[0])#+1    #current r and all params
    
    #output dimension
    out_dim = 1 #just r
    # Number of gaussians to represent the multimodal distribution
    k = 8#4
    logging.info("Running {} dimensions on {} distributions".format(out_dim,k))
    # Network
    input = tf.keras.Input(shape=(l,))
    input_transfer_layer = tf.keras.layers.Dense(1,activation = None,dtype = tf.float64)
    layer = tf.keras.layers.Dense(50, activation='tanh', name='baselayer',dtype = tf.float64)(input)
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
    
    class gdnn(tf.keras.Model):
        def __init__(self,input_dims,output_dimensions, num_mixtures):
            super(gdnn,self).__init__()
            self.in_dim = input_dims
            self.out_dim = output_dimensions
            self.k = num_mixtures
            #create a network of:
            '''
                Input layer
                  |-Mu
                  |
                ->|-Sigma
                  |
                  |-Pi
            '''
            self.mu = tf.keras.layers.Dense(self.k*self.out_dim)
            self.sigma = tf.keras.layers.Dense(self.k*self.out_dim,activation = elu_plus)
            self.pi = tf.keras.layers.Dense(self.k)
            self.built = False
        def build(self,input_shape):
            self.mu.build(input_shape)
            self.sigma.build(input_shape)
            self.pi.build(input_shape)
            self.built = True
        
        def call(self,x):
            xs = x.shape
            assert xs[1] == self.in_dim,"Input to model not of correct size"
            if not self.built:
                self.build(xs)
            return tf.concat([self.mu(x),self.sigma(x), self.pi(x)])

        
            
            
    
    
    model = tf.keras.models.Model(input, [pi, mu, var])
    lr = 5e-4
    wd = 1e-6
    
    optimizer = tf.keras.optimizers.Adam(lr)
    model.summary()
    #model.compile(optimizer, mdn_loss)
    N = np.asarray(X_full).shape[0]
    
    dataset = tf.data.Dataset \
    .from_tensor_slices((X_full, sample_profiles_renormed)) \
    .shuffle(N).batch(N)
    
    # Start training
    best_model = model
    best_loss = np.inf
    max_diff = 0.0  #differential loss
    i = 0
    training_bool = i in range(EPOCHS)
    counter = 0
    counter_max = 100
    counters = []
    
    minimum_delta = 1e-5
    
    MSEs = []
    train_testing_profile, tt_p_para,t_a_r = EinastoSim.generate_n_random_einasto_profile_maggie(1)
    ttp_logged = np.asarray([np.log(p) for p in train_testing_profile]).astype(np.float64)
    ttp_renormed = np.asarray(calculate_renorm_profiles(ttp_logged)).astype(np.float64)
    X_tt = tt_p_para#create_input_vectors(tt_p_para,t_a_r)
    
    overlap_ratios = []
    num_samples = 10
    likelihood_minimum = 0.9
    loss_target = 1e-6#-np.log(likelihood_minimum)
    diff = 0
    logging.info("="*10+"Training info"+"="*10)
    logging.debug('Print every {} epochs'.format(print_every))
    logging.info("Learning Parameters: lr = {} \t wd = {}".format(lr,wd))
    logging.info("Patience: {} increases".format(counter_max))
    logging.info("Minimum delta loss to not lose patience: {}".format(minimum_delta))
    logging.info("Target loss: < {}".format(loss_target))
    logging.info("# Samples: {}".format(num_samples))
    logging.info("="*(33))
    
    train_start = datetime.now()
    logging.info("Starting training at: {}".format(train_start))
    while training_bool:
        for train_x, train_y in dataset:
            with tf.GradientTape() as tape:
                pi_, mu_, var_ = model(train_x,training = True)
                #print(pi_,mu_,var_)
                # calculate loss
                sample, prob_array_training = sample_predictions_tf_r(associated_r,pi_,mu_,var_)
                loss = tf.losses.mean_absolute_error(train_y,sample)
                #mdn_loss(train_y, pi_, mu_, var_)
            # compute and apply gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            #loss = train_step(model, optimizer,train_x,train_y)
            losses.append(tf.reduce_mean(loss))
            #likelihood = np.exp(-tf.reduce_mean(loss).numpy())
            if tf.reduce_mean(loss) > best_loss:
                counter += 1
            
            if len(losses) > 1:
                diff = losses[-1] - losses[-2]
                if diff < max_diff:
                    counter += 1
                if diff < minimum_delta:
                    counter += 1
                elif diff > max_diff + minimum_delta:
                    max_diff = diff
                    counter -= 1 #keep going if differential low enough, even if loss > min
                    counter = max([0,counter]) #keep > 0
            if tf.reduce_mean(loss) < best_loss:
                logging.info("Epoch {}/{}: Elapsed Time: {};Remaining Time estimate: {}; new best loss: {}; Patience: {} %".format(i,EPOCHS,datetime.now()-train_start,(datetime.now() - train_start)*(1-i/EPOCHS),losses[-1], 100*counter/counter_max))
                best_loss = tf.reduce_mean(loss)
                best_model = tf.keras.models.clone_model(model)
                best_model.save(".\\models\\Run_{}\\best_model".format(run_id))
                counter = 0
        #calculate mse
        pi_tt,mu_tt,var_tt = best_model.predict(np.asarray(X_tt))
        sample_preds,sample_probability_array = sample_predictions_tf_r(t_a_r,pi_tt,mu_tt,var_tt)
        profile_sample = sample_preds[0]
        mse_error_profiles = tf.losses.MSE(ttp_renormed[0],profile_sample)
        MSEs.append(mse_error_profiles)
        
        #calculate overlap
        source_overlap = np.dot(np.transpose(ttp_renormed[0]),ttp_renormed[0]) #ignore constant multiplier
        
        overlap_ratio = source_overlap/tf.reduce_sum(profile_sample**2)
        overlap_ratios.append(overlap_ratio)
        
        
        counters.append(counter)
        
        training_bool = i in range(EPOCHS)
        
        loss_break = (best_loss.numpy() < loss_target)# and (np.exp(-best_loss.numpy()) > likelihood_minimum) #equivalent frankly, just redundant
        loss_break = loss_break or (diff < 0) 
        training_bool = (training_bool or (loss_break)) or (counter//counter_max < 1)
        if i % print_every == 0:
            logging.info('Epoch {}/{}: Elapsed Time: {};Remaining Time estimate: {}; loss {}, Patience: {} %; MSE: {}; overlap: {}'.format(i, EPOCHS,datetime.now() - train_start,(datetime.now() - train_start)*(1-i/EPOCHS), losses[-1],100*counter/counter_max,mse_error_profiles, overlap_ratio))       
        i = i+1
    
    logging.info("Training completed after {}/{} epochs. Patience: {} %:: Best Loss: {}".format(i, EPOCHS, 100*counter/counter_max, best_loss))
    logging.info("Reason for exiting: loss_break: {}, diff < 0: {}".format(loss_break,diff<0))
    score_file = "./scores.csv"
    logging.info("Saving best score {} to {}".format(best_loss,score_file))
    
    score_df = pd.read_csv(score_file)
    score_df["MAE"][run_id] = best_loss
    score_df.to_csv(score_file)
    
    plot_folder = ".\\plots\\Run_{}\\".format(run_id)
    save_folder = ".\\models\\Run_{}\\best_model".format(run_id)
    logging.info("Saving best model to {}".format(save_folder))
    best_model.save(save_folder)
    
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    logging.info("Saving to plot folder: {}".format(plot_folder))
    
    now = datetime.now()
    plt.figure()
    plt.plot(losses)
    plt.title("MAE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("NAE Loss")
    plt.savefig(plot_folder+"Losses_{}_{}_{}.png".format(now.hour,now.day,now.month))
    
    plt.figure()
    plt.plot(counters)
    plt.title("Counter values")
    plt.xlabel("Epoch")
    plt.ylabel("Counter")
    plt.savefig(plot_folder+"Counter_{}_{}_{}.png".format(now.hour,now.day,now.month))
    
    plt.figure()
    plt.plot(MSEs)
    plt.title("MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Pseudo MSE")
    plt.savefig(plot_folder+"MSE_{}_{}_{}.png".format(now.hour,now.day,now.month))
    
    plt.figure()
    plt.plot(overlap_ratios)
    plt.title("Profile Overlap Ratios true/generated")
    plt.xlabel("Epoch")
    plt.ylabel("Overlap")
    plt.savefig(plot_folder+"Overlap_{}_{}_{}.png".format(now.hour,now.day,now.month))
    
    n_test_profiles = 10
    test_profiles,t_profile_params,t_associated_r = EinastoSim.generate_n_random_einasto_profile_maggie(n_test_profiles)
    t_sample_profiles_logged = np.asarray([np.log(p) for p in test_profiles]).astype(np.float64)
    t_s_renorm = np.asarray(calculate_renorm_profiles(t_sample_profiles_logged)).astype(np.float64)
    X_test = t_profile_params#create_input_vectors(t_profile_params,t_associated_r)
    
    pi_test, mu_test,var_test = best_model.predict(np.asarray(X_test))
    sample_preds, sample_probability_array = sample_predictions_tf_r(t_associated_r,pi_test,mu_test,var_test)
    
    for i in range(n_test_profiles):
        profile_sample = sample_preds[i,:]
        test_prof = t_s_renorm[i]
        plt.figure()
        plt.plot(t_associated_r[i],test_prof,label = "True profile")
        
        logging.debug("Parameters for {}: {}".format(i,EinastoSim.print_params_maggie(t_profile_params[i])))
        mng = plt.get_current_fig_manager()
        
        plt.plot(t_associated_r[i],profile_sample, label = "Sample")
        probability_arr = [pi_test[i][kd]/(tf.sqrt(2*np.pi*var_test[i][kd]))*tf.exp(-(1/(2*var_test[i][kd]))*((t_associated_r[i]-mu_test[i][kd])**2)) for kd in range(k)]
        constituent_probabilities = tf.stack(probability_arr)
        for kd in range(k):
            plt.plot(t_associated_r[i],constituent_probabilities[kd], label = "Constituent {}".format(kd))
        plt.plot(t_associated_r[i],tf.add_n(probability_arr),label = "Profile Addition")
        plt.legend()
        plt.title(EinastoSim.print_params_maggie(t_profile_params[i]).replace("\t",""))
        plt.xlabel("Radius [Mpc]")
        plt.ylabel("log({}) []".format(u"\u03C1"))
        mng.full_screen_toggle()
        plt.show()
        plt.pause(1e-3)
        plt.savefig(plot_folder+"Sample_profiles_{}_{}_{}_{}_{}.png".format(run_id,i,now.hour,now.day,now.month))
    
    #plot_constituent_profiles(t_associated_r,pi_test,mu_test,var_test)
    
    
    #profile_sample = sample_preds[0,:]
    #pi_s_t = pi_test[0]
    #mu_s_t = mu_test[0]
    #var_s_t = var_test[0]
    '''
    def sample_predictions_tf_r(r_values, pi_vals, mu_vals, var_vals):
    n,k = pi_vals.shape
    assert n == len(r_values), "R value length does not match statistic values"
    final_array = []
    for i in range(n):
        prob = tf.stack([1/(tf.sqrt(2*np.pi*var_[i][kd]))*tf.exp(-(1/2*var_[i][kd])*((associated_r[i]-mu_[i][kd])**2)) for kd in range(k)])
        p2 = tf.unstack(prob)
        p3 = tf.stack([pi_[i,j]*p2[j] for j in range(k)])
        final_array.append(tf.math.reduce_sum(p3,0))
    return tf.stack(final_array)

    '''
    
    #test_prob = tf.stack([pi_s_t[kd]/(tf.sqrt(2*np.pi*var_s_t[kd]))*tf.exp(-(1/2*var_s_t[kd])*((associated_r[i]-mu_s_t[kd])**2)) for kd in range(k)])
    
    with open(run_file,"w") as f:
        f.write(str(run_id +1))