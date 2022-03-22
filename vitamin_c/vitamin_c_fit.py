import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
tfd = tfp.distributions
import time
from lal import GreenwichMeanSiderealTime
from astropy.time import Time
from astropy import coordinates as coord
import corner
import os
import shutil
import h5py
import json
import sys
from sys import exit
from universal_divergence import estimate
import natsort
import plotting
from tensorflow.keras import regularizers
from scipy.spatial.distance import jensenshannon
import scipy.stats as st
import wandb
from wandb.keras import WandbCallback

from vitamin_c_model_fit import CVAE

from callbacks import  PlotCallback, TrainCallback, TestCallback, TimeCallback
from load_data_fit import load_data, load_samples, convert_ra_to_hour_angle, convert_hour_angle_to_ra, DataLoader, psiphi_to_psiX, psiX_to_psiphi, m1m2_to_chirpmassq, chirpmassq_to_m1m2
#from keras_adamw import AdamW

def get_param_index(all_pars,pars,sky_extra=None):
    """ 
    Get the list index of requested source parameter types
    """
    # identify the indices of wrapped and non-wrapped parameters - clunky code
    mask = []
    idx = []

    # loop over inference params
    for i,p in enumerate(all_pars):

        # loop over wrapped params
        flag = False
        for q in pars:
            if p==q:
                flag = True    # if inf params is a wrapped param set flag

        # record the true/false value for this inference param
        if flag==True:
            mask.append(True)
            idx.append(i)
        elif flag==False:
            mask.append(False)

    if sky_extra is not None:
        if sky_extra:
            mask.append(True)
            idx.append(len(all_pars))
        else:
            mask.append(False)

    return mask, idx, np.sum(mask)




def get_params(params, bounds, fixed_vals, params_dir = "./params_files", print_masks=False):
    """
    params_file = os.path.join(params_dir,'params.json')
    bounds_file = os.path.join(params_dir,'bounds.json')
    fixed_vals_file = os.path.join(params_dir,'fixed_vals.json')

    EPS = 1e-3

    # Load parameters files
    with open(params_file, 'r') as fp:
        params = json.load(fp)
    with open(bounds_file, 'r') as fp:
        bounds = json.load(fp)
    with open(fixed_vals_file, 'r') as fp:
        fixed_vals = json.load(fp)
    """
    EPS = 1e-3
    # if doing hour angle, use hour angle bounds on RA                                                                                                                     
    bounds['ra_min'] = convert_ra_to_hour_angle(bounds['ra_min'],params,None,single=True)
    bounds['ra_max'] = convert_ra_to_hour_angle(bounds['ra_max'],params,None,single=True)
    print('... converted RA bounds to hour angle')
    masks = {}
    masks["inf_ol_mask"], masks["inf_ol_idx"], masks["inf_ol_len"] = get_param_index(params['inf_pars'],params['bilby_pars'])
    masks["bilby_ol_mask"], masks["bilby_ol_idx"], masks["bilby_ol_len"] = get_param_index(params['bilby_pars'],params['inf_pars'])
    
    
    # identify the indices of different sets of physical parameters                                                                                                          
    masks["vonmise_mask"], masks["vonmise_idx_mask"], masks["vonmise_len"] = get_param_index(params['inf_pars'],params['vonmise_pars'])
    masks["gauss_mask"], masks["gauss_idx_mask"], masks["gauss_len"] = get_param_index(params['inf_pars'],params['gauss_pars'])
    masks["sky_mask"], masks["sky_idx_mask"], masks["sky_len"] = get_param_index(params['inf_pars'],params['sky_pars'])
    masks["ra_mask"], masks["ra_idx_mask"], masks["ra_len"] = get_param_index(params['inf_pars'],['ra'])
    masks["dec_mask"], masks["dec_idx_mask"], masks["dec_len"] = get_param_index(params['inf_pars'],['dec'])
    masks["m1_mask"], masks["m1_idx_mask"], masks["m1_len"] = get_param_index(params['inf_pars'],['mass_1'])
    masks["m2_mask"], masks["m2_idx_mask"], masks["m2_len"] = get_param_index(params['inf_pars'],['mass_2'])
    #masks["q_mask"], masks["q_idx_mask"], masks["q_len"] = get_param_index(params['inf_pars'],['mass_ratio'])
    #masks["M_mask"], masks["M_idx_mask"], masks["M_len"] = get_param_index(params['inf_pars'],['chirp_mass'])

    #idx_mask = np.argsort(gauss_idx_mask + vonmise_idx_mask + m1_idx_mask + m2_idx_mask + sky_idx_mask) # + dist_idx_mask)                                                  
    masks["idx_mask"] = np.argsort(masks["m1_idx_mask"] + masks["m2_idx_mask"] + masks["gauss_idx_mask"] + masks["vonmise_idx_mask"]) # + sky_idx_mask)                 
    #masks["idx_mask"] = np.argsort(masks["q_idx_mask"] + masks["M_idx_mask"] + masks["gauss_idx_mask"] + masks["vonmise_idx_mask"]) # + sky_idx_mask)                      
    masks["dist_mask"], masks["dist_idx_mask"], masks["dis_len"] = get_param_index(params['inf_pars'],['luminosity_distance'])
    masks["not_dist_mask"], masks["not_dist_idx_mask"], masks["not_dist_len"] = get_param_index(params['inf_pars'],['mass_1','mass_2','psi','phase','geocent_time','theta_jn','ra','dec','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl'])

    masks["phase_mask"], masks["phase_idx_mask"], masks["phase_len"] = get_param_index(params['inf_pars'],['phase'])
    masks["not_phase_mask"], masks["not_phase_idx_mask"], masks["not_phase_len"] = get_param_index(params['inf_pars'],['mass_1','mass_2','luminosity_distance','psi','geocent_time','theta_jn','ra','dec','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl'])

    masks["geocent_mask"], masks["geocent_idx_mask"], masks["geocent_len"] = get_param_index(params['inf_pars'],['geocent_time'])
    masks["not_geocent_mask"], masks["not_geocent_idx_mask"], masks["not_geocent_len"] = get_param_index(params['inf_pars'],['mass_1','mass_2','luminosity_distance','psi','phase','theta_jn','ra','dec','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl'])

    masks["xyz_mask"], masks["xyz_idx_mask"], masks["xyz_len"] = get_param_index(params['inf_pars'],['luminosity_distance','ra','dec'])
    masks["not_xyz_mask"], masks["not_xyz_idx_mask"], masks["not_xyz_len"] = get_param_index(params['inf_pars'],['mass_1','mass_2','psi','phase','geocent_time','theta_jn','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl'])

    masks["periodic_mask"], masks["periodic_idx_mask"], masks["periodic_len"] = get_param_index(params['inf_pars'],['phase','psi','phi_12','phi_jl'])
    masks["nonperiodic_mask"], masks["nonperiodic_idx_mask"], masks["nonperiodic_len"] = get_param_index(params['inf_pars'],['mass_1','mass_2','luminosity_distance','geocent_time','theta_jn','a_1','a_2','tilt_1','tilt_2'])
    masks["nonperiodic_nonm1m2_mask"], masks["nonperiodic_nonm1m2_idx_mask"], masks["nonperiodic_nonm1m2_len"] = get_param_index(params['inf_pars'],['luminosity_distance','geocent_time','theta_jn','a_1','a_2','tilt_1','tilt_2'])
    masks["nonperiodic_m1m2_mask"], masks["nonperiodic_m1m2_idx_mask"], masks["nonperiodic_m1m2_len"] = get_param_index(params['inf_pars'],['mass_1','mass_2'])

    masks["idx_xyz_mask"] = np.argsort(masks["xyz_idx_mask"] + masks["not_xyz_idx_mask"])
    masks["idx_dist_mask"] = np.argsort(masks["not_dist_idx_mask"] + masks["dist_idx_mask"])
    masks["idx_phase_mask"] = np.argsort(masks["not_phase_idx_mask"] + masks["phase_idx_mask"])
    masks["idx_geocent_mask"] = np.argsort(masks["not_geocent_idx_mask"] + masks["geocent_idx_mask"])
    masks["idx_periodic_mask"] = np.argsort(masks["nonperiodic_idx_mask"] + masks["periodic_idx_mask"] + masks["ra_idx_mask"] + masks["dec_idx_mask"])
    if print_masks:
        print(masks["xyz_mask"])
        print(masks["not_xyz_mask"])
        print(masks["idx_xyz_mask"])
        #masses_len = m1_len + m2_len                                                                                                                                            
        print(params['inf_pars'])
        print(masks["vonmise_mask"],masks["vonmise_idx_mask"])
        print(masks["gauss_mask"],masks["gauss_idx_mask"])
        print(masks["m1_mask"],masks["m1_idx_mask"])
        print(masks["m2_mask"],masks["m2_idx_mask"])
        print(masks["sky_mask"],masks["sky_idx_mask"])
        print(masks["idx_mask"])
        
    return params, bounds, masks, fixed_vals



def ramp_func(epoch,start,ramp_length, n_cycles):
    i = (epoch-start)/(2.0*ramp_length)
    print("ramp",epoch,i)
    if i<0:
        return 0.0
    if i>=n_cycles:
        return 1.0

    return min(1.0,2.0*np.remainder(i,1.0))

def paper_plots(test_dataset, y_data_test, x_data_test, model, params, plot_dir, run, bilby_samples):
    """ Make publication plots
    """
    epoch = 'pub_plot'; ramp = 1
    plotter = plotting.make_plots(params, None, None, x_data_test) 

    for step, (x_batch_test, y_batch_test) in test_dataset.enumerate():
        mu_r1, z_r1, mu_q, z_q = model.gen_z_samples(x_batch_test, y_batch_test, nsamples=1000)
        plot_latent(mu_r1,z_r1,mu_q,z_q,epoch,step,run=plot_dir)
        start_time_test = time.time()
        samples = model.gen_samples(y_batch_test, ramp=ramp, nsamples=params['n_samples'])
        end_time_test = time.time()
        if np.any(np.isnan(samples)):
            print('Found nans in samples. Not making plots')
            for k,s in enumerate(samples):
                if np.any(np.isnan(s)):
                    print(k,s)
            KL_est = [-1,-1,-1]
        else:
            print('Run {} Testing time elapsed for {} samples: {}'.format(run,params['n_samples'],end_time_test - start_time_test))
            KL_est = plot_posterior(samples,x_batch_test[0,:],epoch,step,all_other_samples=bilby_samples[:,step,:],run=plot_dir, params=params, bounds=bounds, masks=masks)
            _ = plot_posterior(samples,x_batch_test[0,:],epoch,step,run=plot_dir, params=params, bounds=bounds, masks=masks)
    print('... Finished making publication plots! Congrats fam.')

    # Make p-p plots
    plotter.plot_pp(model, y_data_test, x_data_test, params, bounds, inf_ol_idx, bilby_ol_idx)
    print('... Finished making p-p plots!')

    # Make KL plots
    plotter.gen_kl_plots(model,y_data_test, x_data_test, params, bounds, inf_ol_idx, bilby_ol_idx)
    print('... Finished making KL plots!')    

    return


def run_vitc(params, x_data_train, y_data_train, x_data_val, y_data_val, x_data_test, y_data_test, y_data_test_noisefree, save_dir, truth_test, bounds, fixed_vals, bilby_samples, snrs_test=None, params_dir = "./params_files"):

    #params, bounds, masks, fixed_vals = get_params(params_dir = params_dir)
    params, bounds, masks, fixed_vals = get_params(params, bounds, fixed_vals, params_dir = params_dir)
    run = time.strftime('%y-%m-%d-%X-%Z')

    # define which gpu to use during training
    gpu_num = str(params['gpu_num'])   
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num
    #print('... running on GPU {}'.format(gpu_num))
    print("CUDA DEV: ",os.environ["CUDA_VISIBLE_DEVICES"])
    # Let GPU consumption grow as needed
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    print('... letting GPU consumption grow as needed')
    
    train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_log_dir = params['plot_dir'] + '/logs'

    epochs = params['num_iterations']
    train_size = params['load_chunk_size']
    batch_size = params['batch_size']
    val_size = params['val_dataset_size']
    test_size = params['r']
    plot_dir = params['plot_dir']
    plot_cadence = int(0.5*params['plot_interval'])
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = os.path.join(plot_dir,"checkpoint","model.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    comp_post_dir = os.path.join(plot_dir, "comp_posterior")
    full_post_dir = os.path.join(plot_dir, "full_posterior")
    post_samp_dir = os.path.join(plot_dir, "posterior_samples")
    latent_dir = os.path.join(plot_dir, "latent_plot")
    dirs = [checkpoint_dir, comp_post_dir, full_post_dir, post_samp_dir, latent_dir]
    for direc in dirs:
        if not os.path.isdir(direc):
            os.makedirs(direc)

    make_paper_plots = params['make_paper_plots']
    hyper_par_tune = False

    # if doing hour angle, use hour angle bounds on RA
    bounds['ra_min'] = convert_ra_to_hour_angle(bounds['ra_min'],params,None,single=True)
    bounds['ra_max'] = convert_ra_to_hour_angle(bounds['ra_max'],params,None,single=True)
    print('... converted RA bounds to hour angle')

    # load the training data
    if not make_paper_plots:
        train_dataset = DataLoader(params["train_set_dir"],params = params,bounds = bounds, masks = masks,fixed_vals = fixed_vals, chunk_batch = 40, num_epoch_load = 10, batch_size = params["batch_size"]) 
        validation_dataset = DataLoader(params["val_set_dir"],params = params,bounds = bounds, masks = masks,fixed_vals = fixed_vals, chunk_batch = 2, val_set = True)

    test_dataset = DataLoader(params["test_set_dir"],params = params,bounds = bounds, masks = masks,fixed_vals = fixed_vals, chunk_batch = test_size, test_set = True)

    print("Loading intitial data...")
    train_dataset.load_next_chunk()
    validation_dataset.load_next_chunk()
    test_dataset.load_next_chunk()

    # load precomputed samples
    bilby_samples = []
    for sampler in params['samplers'][1:]:
        bilby_samples.append(load_samples(params,sampler, bounds = bounds))
    bilby_samples = np.array(bilby_samples)
    #bilby_samples = np.array([load_samples(params,'dynesty'),load_samples(params,'ptemcee'),load_samples(params,'cpnest')])

    train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    start_epoch = 0
    if params['resume_training']:
        model = CVAE(params, bounds, masks)
        # Load the previously saved weights
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest)
        print('... loading in previous model %s' % checkpoint_path)
        with open(os.path.join(params['plot_dir'], "loss.txt"),"r") as f:
            start_epoch = len(np.loadtxt(f))
    else:
        model = CVAE(params, bounds, masks)

    # Make publication plots
    if make_paper_plots:
        print('... Making plots for publication.')
        # Load the previously saved weights
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest)
        print('... loading in previous model %s' % checkpoint_path)
        paper_plots(test_dataset, y_data_test, x_data_test, model, params, plot_dir, run, bilby_samples)
        return

    # start the training loop
    train_loss = np.zeros((epochs,3))
    val_loss = np.zeros((epochs,3))
    ramp_start = params['ramp_start']
    ramp_length = params['ramp_end']
    ramp_cycles = 1
    KL_samples = []

    #tf.keras.mixed_precision.set_global_policy('float32')
    #optimizer = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay = 1e-8)
    optimizer = tf.keras.optimizers.Adam(params["initial_training_rate"], clipvalue = 0.5)
    #optimizer = AdamW(lr=1e-4, model=model,
    #                  use_cosine_annealing=True, total_iterations=40)
    #optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    # Keras hyperparameter optimization
    if hyper_par_tune:
        import keras_hyper_optim
        del model
        keras_hyper_optim.main(train_dataset, val_dataset)
        exit()

    # log params used for this run
    path = params['plot_dir']
    root_dir = params['root_dir']
    shutil.copy(os.path.join(root_dir,'vitamin_c_fit.py'),path)
    if model_fit_type == "basic":
        shutil.copy(os.path.join(root_dir,'vitamin_c_model_fit.py'),path)
    elif model_fit_type == "multidet":
        shutil.copy(os.path.join(root_dir,'vitamin_c_model_fit_multidet.py'),path)
    elif model_fit_type == "multiscale":
        shutil.copy(os.path.join(root_dir,'vitamin_c_model_fit_multiscale.py'),path)
    elif model_fit_type == "kaggle":
        shutil.copy(os.path.join(root_dir,'vitamin_c_model_fit_kaggle1.py'),path)
    elif model_fit_type == "resnet":
        shutil.copy(os.path.join(root_dir,'vitamin_c_model_fit_resnet.py'),path)
    elif model_fit_type == "freq":
        shutil.copy(os.path.join(root_dir,'vitamin_c_model_fit_freq.py'),path)

    shutil.copy(os.path.join(root_dir,'load_data_fit.py'),path)
    shutil.copy(os.path.join(params_dir,'params.json'),path)
    shutil.copy(os.path.join(params_dir,'bounds.json'),path)

    # compile and build the model (hardcoded values will change soon)
    model.compile(run_eagerly = False, optimizer = optimizer, loss = model.compute_loss)
    #test_data = tf.zeros((batch_size, 1024, 2))
    #test_pars = tf.zeros((batch_size, 15))
    #model([test_data, test_pars])
    #model.build([(None, 1024,2), (None, 15)])
    #print(model.summary())
    with open(os.path.join(path, "model_summary.txt"),"w") as f:
        #model.summary(print_fn=lambda x: f.write(x + '\n'))
        model.encoder_r1.summary(print_fn=lambda x: f.write(x + '\n'))
        model.encoder_q.summary(print_fn=lambda x: f.write(x + '\n'))
        model.decoder_r2.summary(print_fn=lambda x: f.write(x + '\n'))
    
    

    #wandb.init(project="Vitamin", entity="jgl")
    #wandb.config = params

    callbacks = [PlotCallback(plot_dir, epoch_plot=100,start_epoch=start_epoch), TrainCallback(checkpoint_path, optimizer, plot_dir, train_dataset, model), TestCallback(test_dataset,comp_post_dir,full_post_dir, latent_dir, bilby_samples, test_epoch = 1000), TimeCallback(save_dir=plot_dir, save_interval = 100)]#, WandbCallback(save_model = False)]

    model.fit(train_dataset, use_multiprocessing = False, workers = 6,epochs = 30000, callbacks = callbacks, shuffle = False, validation_data = validation_dataset, max_queue_size = 100, initial_epoch = start_epoch)

    # not happy with this re-wrapping of the dataset
    #data_gen_wrap = tf.data.Dataset.from_generator(lambda : train_dataset,(tf.float32,tf.float32))
    #valdata_gen_wrap = tf.data.Dataset.from_generator(lambda : validation_dataset,(tf.float32,tf.float32))

    #model.fit_generator(data_gen_wrap, use_multiprocessing = False, workers = 6,epochs = 10000, callbacks = callbacks, shuffle = False, validation_data = valdata_gen_wrap, max_queue_size = 100)


