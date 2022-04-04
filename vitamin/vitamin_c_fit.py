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
#import plotting
from tensorflow.keras import regularizers
from scipy.spatial.distance import jensenshannon
import scipy.stats as st
#import wandb
#from wandb.keras import WandbCallback

from .model import CVAE

from .callbacks import  PlotCallback, TrainCallback, TestCallback, TimeCallback
from .load_data import DataLoader, load_samples, convert_ra_to_hour_angle, convert_hour_angle_to_ra, psiphi_to_psiX, psiX_to_psiphi, m1m2_to_chirpmassq, chirpmassq_to_m1m2
#from keras_adamw import AdamW


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
            KL_est = plot_posterior(samples,x_batch_test[0,:],epoch,step,all_other_samples=bilby_samples[:,step,:],run=plot_dir, config=config)
            _ = plot_posterior(samples,x_batch_test[0,:],epoch,step,run=plot_dir, config=config)
    print('... Finished making publication plots! Congrats fam.')

    # Make p-p plots
    plotter.plot_pp(model, y_data_test, x_data_test, params, bounds, inf_ol_idx, bilby_ol_idx)
    print('... Finished making p-p plots!')

    # Make KL plots
    plotter.gen_kl_plots(model,y_data_test, x_data_test, params, bounds, inf_ol_idx, bilby_ol_idx)
    print('... Finished making KL plots!')    

    return


def train_vitamin(config, save_dir, truth_test, bounds, fixed_vals, bilby_samples, snrs_test=None, params_dir = "./params_files"):

    #params, bounds, masks, fixed_vals = get_params(params_dir = params_dir)
    run = time.strftime('%y-%m-%d-%X-%Z')

    # define which gpu to use during training
    gpu_num = str(config['gpu_num'])   
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num

    print("CUDA DEV: ",os.environ["CUDA_VISIBLE_DEVICES"])

    # Let GPU consumption grow as needed
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    print('... letting GPU consumption grow as needed')
    
    train_log_dir = config["output"]['plot_dir'] + '/logs'

    epochs = config["training"]['num_iterations']
    train_size = config["training"]['load_chunk_size']
    batch_size = config["training"]['batch_size']
    val_size = config["training"]['val_dataset_size']
    test_size = config["data"]['n_test_data']
    plot_dir = config["output"]['plot_dir']
    plot_cadence = int(0.5*init.args['plot_interval'])
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = os.path.join(plot_dir,"checkpoint","model.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    dirs = [checkpoint_dir]
    for direc in dirs:
        if not os.path.isdir(direc):
            os.makedirs(direc)

    make_paper_plots = config["testing"]['make_paper_plots']
    hyper_par_tune = False

    # if doing hour angle, use hour angle bounds on RA
    #bounds['ra_min'] = convert_ra_to_hour_angle(bounds['ra_min'],init.args,None,single=True)
    #bounds['ra_max'] = convert_ra_to_hour_angle(bounds['ra_max'],init.args,None,single=True)
    #print('... converted RA bounds to hour angle')

    # load the training data
    if not make_paper_plots:
        train_dataset = DataLoader(config["data"]["training_set_directory"],config = config,chunk_batch = config["training"]["load_chunk"], num_epoch_load = config["training"]["num_epoch_load"], batch_size = config["training"]["batch_size"]) 
        validation_dataset = DataLoader(config["data"]["validation_set_directory"],config=config, chunk_batch = 2, val_set = True)

    test_dataset = DataLoader(config["data"]["test_set_directory"],config=config, chunk_batch = test_size, test_set = True)

    print("Loading intitial data...")
    train_dataset.load_next_chunk()
    validation_dataset.load_next_chunk()
    test_dataset.load_next_chunk()

    # load precomputed samples
    bilby_samples = []
    for sampler in config["data"]["samplers"][1:]:
        bilby_samples.append(load_samples(config,sampler))
    bilby_samples = np.array(bilby_samples)
    #bilby_samples = np.array([load_samples(init.args,'dynesty'),load_samples(init.args,'ptemcee'),load_samples(init.args,'cpnest')])

    train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    start_epoch = 0
    
    model = CVAE(init)
    if init.args['resume_training']:
        # Load the previously saved weights
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest)
        print('... loading in previous model %s' % checkpoint_path)
        with open(os.path.join(init.args['plot_dir'], "loss.txt"),"r") as f:
            start_epoch = len(np.loadtxt(f))

    # start the training loop
    train_loss = np.zeros((config["training"]['num_iterations'],3))
    val_loss = np.zeros((config["training"]['num_iterations'],3))
    ramp_cycles = 1
    KL_samples = []

    optimizer = tfa.optimizers.AdamW(learning_rate=config["training"]["initial_learning_rate"], weight_decay = 1e-8)
    #optimizer = tf.keras.optimizers.Adam(config["training"]["initial_learning_rate"])
    #optimizer = AdamW(lr=1e-4, model=model,
    #                  use_cosine_annealing=True, total_iterations=40)
    # Keras hyperparameter optimization
    if hyper_par_tune:
        import keras_hyper_optim
        del model
        keras_hyper_optim.main(train_dataset, val_dataset)
        exit()

    # compile and build the model (hardcoded values will change soon)
    model.compile(run_eagerly = False, optimizer = optimizer, loss = model.compute_loss)

    #model([test_data, test_pars])
    #model.build([(None, 1024,2), (None, 15)])

    with open(os.path.join(config["output"]["output_directory"], "model_summary.txt"),"w") as f:
        model.encoder_r1.summary(print_fn=lambda x: f.write(x + '\n'))
        model.encoder_q.summary(print_fn=lambda x: f.write(x + '\n'))
        model.decoder_r2.summary(print_fn=lambda x: f.write(x + '\n'))
    
    
    callbacks = [PlotCallback(config["output"]["output_directory"], epoch_plot=100,start_epoch=start_epoch), TrainCallback(config, checkpoint_path, optimizer, plot_dir, train_dataset, model), TestCallback(config, test_dataset,comp_post_dir,full_post_dir, latent_dir, bilby_samples, test_epoch = 1000), TimeCallback(init.args)]

    model.fit(train_dataset, use_multiprocessing = False, workers = 6,epochs = config["training"]["num_iterations"], callbacks = callbacks, shuffle = False, validation_data = validation_dataset, max_queue_size = 100, initial_epoch = start_epoch)

    #model.fit_generator(data_gen_wrap, use_multiprocessing = False, workers = 6,epochs = 10000, callbacks = callbacks, shuffle = False, validation_data = valdata_gen_wrap, max_queue_size = 100)


