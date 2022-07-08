
def test(config):

    #params, bounds, masks, fixed_vals = get_params(params_dir = params_dir)
    run = time.strftime('%y-%m-%d-%X-%Z')

    # define which gpu to use during training
    try:
        #gpu_num = str(vitamin_config["training"]['gpu_num'])   
        #os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num
        print("CUDA DEV: ",os.environ["CUDA_VISIBLE_DEVICES"])
    except:
        print("No CUDA devices")


    from lal import GreenwichMeanSiderealTime
    from astropy.time import Time
    from astropy import coordinates as coord
    import corner
    from universal_divergence import estimate
    import natsort
    from scipy.spatial.distance import jensenshannon
    import scipy.stats as st
    #from keras_adamw import AdamW
    import tensorflow as tf
    import tensorflow_addons as tfa
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    from tensorflow.keras import regularizers
    from ..vitamin_model import CVAE
    from ..callbacks import  PlotCallback, TrainCallback, TestCallback, TimeCallback
    from .load_data import DataLoader, convert_ra_to_hour_angle, convert_hour_angle_to_ra, psiphi_to_psiX, psiX_to_psiphi, m1m2_to_chirpmassq, chirpmassq_to_m1m2
    from keras_adabound import AdaBound
        
    # Let GPU consumption grow as needed
    config_gpu = tf.compat.v1.ConfigProto()
    config_gpu.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config_gpu)
    print('... letting GPU consumption grow as needed')

    train_log_dir = os.path.join(config["output"]['output_directory'],'logs')

    test_directory = os.path.join(config["data"]["data_directory"], "test", "waveforms")

    epochs = config["training"]['num_iterations']
    plot_cadence = int(0.5*config["training"]["plot_interval"])
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = os.path.join(config["output"]["output_directory"],"checkpoint","model")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    dirs = [checkpoint_dir]
    for direc in dirs:
        if not os.path.isdir(direc):
            os.makedirs(direc)


    test_dataset = DataLoader(test_directory,config=config, test_set = True)
    test_dataset.load_next_chunk()
    test_dataset.load_bilby_samples()
    
    # load precomputed samples
    bilby_samples = []
    for sampler in config["testing"]["samplers"][1:]:
        bilby_samples.append(test_dataset.sampler_outputs[sampler])
    bilby_samples = np.array(bilby_samples)

    start_epoch = 0
    
    model = CVAE(config)
    if config["training"]["transfer_model_checkpoint"]:
        model.load_weights(config["training"]["transfer_model_checkpoint"])
        print('... loading in previous model %s' % config["training"]["transfer_model_checkpoint"])

    elif config["training"]['resume_training']:
        # Load the previously saved weights
        #latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(checkpoint_path)
        print('... loading in previous model %s' % checkpoint_path)
        with open(os.path.join(config["output"]['output_directory'], "loss.txt"),"r") as f:
            start_epoch = len(np.loadtxt(f))

    # compile and build the model (hardcoded values will change soon)
    model.compile(run_eagerly = None, optimizer = optimizer, loss = model.compute_loss)

    test_call = TestCallback(config, test_dataset, bilby_samples)


if __name__ == "__main__":

    import os
    import shutil
    import h5py
    import json
    import sys
    from sys import exit
    import time
    import argparse
    import matplotlib.pyplot as plt
    import numpy as np
    
    parser = argparse.ArgumentParser(description='Input files and options')
    parser.add_argument('--ini-file', metavar='i', type=str, help='path to ini file')
    #parser.add_argument('--gpu', metavar='i', type=int, help='path to ini file', default = None)
    args = parser.parse_args()

    """
    if args.gpu is not None:
        # define which gpu to use during training
        try:
            os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
            print("SET CUDA DEV: ",os.environ["CUDA_VISIBLE_DEVICES"])
        except:
            print("No CUDA devices")
    """

    from .gw_parser import GWInputParser

    vitamin_config = GWInputParser(args.ini_file)

    test(vitamin_config)

