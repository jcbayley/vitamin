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
import argparse
from .model import CVAE
from .vitamin_parser import InputParser
from .callbacks import  PlotCallback, TrainCallback, TestCallback, TimeCallback
from .load_data import DataLoader, convert_ra_to_hour_angle, convert_hour_angle_to_ra, psiphi_to_psiX, psiX_to_psiphi, m1m2_to_chirpmassq, chirpmassq_to_m1m2
#from keras_adamw import AdamW


def train(config):

    #params, bounds, masks, fixed_vals = get_params(params_dir = params_dir)
    run = time.strftime('%y-%m-%d-%X-%Z')

    # define which gpu to use during training
    gpu_num = str(config["training"]['gpu_num'])   
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num

    print("CUDA DEV: ",os.environ["CUDA_VISIBLE_DEVICES"])

    # Let GPU consumption grow as needed
    config_gpu = tf.compat.v1.ConfigProto()
    config_gpu.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config_gpu)
    print('... letting GPU consumption grow as needed')

    train_log_dir = os.path.join(config["output"]['output_directory'],'logs')

    training_directory = os.path.join(config["data"]["data_directory"], "training")
    validation_directory = os.path.join(config["data"]["data_directory"], "validation")
    test_directory = os.path.join(config["data"]["data_directory"], "test", "waveforms")

    epochs = config["training"]['num_iterations']
    plot_cadence = int(0.5*config["training"]["plot_interval"])
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = os.path.join(config["output"]["output_directory"],"checkpoint","model.ckpt")
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
        train_dataset = DataLoader(training_directory,config = config) 
        validation_dataset = DataLoader(validation_directory,config=config,val_set = True)

    test_dataset = DataLoader(test_directory,config=config, test_set = True)

    print("Loading intitial data...")
    train_dataset.load_next_chunk()
    validation_dataset.load_next_chunk()
    test_dataset.load_next_chunk()
    test_dataset.load_bilby_samples()

    # load precomputed samples
    bilby_samples = []
    for sampler in config["testing"]["samplers"][1:]:
        bilby_samples.append(test_dataset.sampler_outputs[sampler])
    bilby_samples = np.array(bilby_samples)

    start_epoch = 0
    
    model = CVAE(config)
    if config["training"]['resume_training']:
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
    
    
    callbacks = [PlotCallback(config["output"]["output_directory"], epoch_plot=100,start_epoch=start_epoch), TrainCallback(config, optimizer, train_dataset, model), TestCallback(config, test_dataset, bilby_samples, test_epoch = 1000), TimeCallback(config)]

    model.fit(train_dataset, use_multiprocessing = False, workers = 6, epochs = config["training"]["num_iterations"], callbacks = callbacks, shuffle = False, validation_data = validation_dataset, max_queue_size = 100, initial_epoch = start_epoch)

    #model.fit_generator(data_gen_wrap, use_multiprocessing = False, workers = 6,epochs = 10000, callbacks = callbacks, shuffle = False, validation_data = valdata_gen_wrap, max_queue_size = 100)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Input files and options')
    parser.add_argument('--ini-file', metavar='i', type=str, help='path to ini file')

    args = parser.parse_args()
    vitamin_config = InputParser(args.ini_file)

    train(vitamin_config)

