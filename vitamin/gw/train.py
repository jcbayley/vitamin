import torch
from ..tools import make_ppplot, loss_plot, latent_corner_plot, latent_samp_fig
from .test import test_model
from collections import OrderedDict
from ..train import train_loop
from .callbacks import PosteriorComparisonCallback, LoadDataCallback
from ..callbacks import SaveModelCallback, LossPlotCallback, AnnealCallback, LearningRateCallback
import time

def setup_and_train(config):

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
    import pickle
    import torchsummary
    from ..vitamin_model import CVAE
    #pip infrom ..callbacks import  PlotCallback, TrainCallback, TestCallback, TimeCallback, optimiserSave, LearningRateCallback, LogminRampCallback, AnnealCallback, BatchRampCallback
    from .load_data import DataSet, convert_ra_to_hour_angle, convert_hour_angle_to_ra, psiphi_to_psiX, psiX_to_psiphi, m1m2_to_chirpmassq, chirpmassq_to_m1m2
        
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_log_dir = os.path.join(config["output"]['output_directory'],'logs')

    training_directory = os.path.join(config["data"]["data_directory"], "training")
    validation_directory = os.path.join(config["data"]["data_directory"], "validation")
    test_directory = os.path.join(config["data"]["data_directory"], "test", "waveforms")

    epochs = config["training"]['num_iterations']
    plot_cadence = int(0.5*config["training"]["plot_interval"])
    checkpoint_path = os.path.join(config["output"]["output_directory"],"checkpoint","model.pt")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    dirs = [checkpoint_dir]
    for direc in dirs:
        if not os.path.isdir(direc):
            os.makedirs(direc)

    make_paper_plots = config["testing"]['make_paper_plots']

    # load the training data
    if not make_paper_plots:
        train_dataset = DataSet(training_directory,config = config)
        validation_dataset = DataSet(validation_directory,config=config,val_set = True)
        train_dataset.load_next_chunk()
        validation_dataset.load_next_chunk()

        print("VAL_SIZE: ", len(validation_dataset))
        print("TRAIN_SIZE: ", len(train_dataset))
        
    if config["training"]["test_interval"] != False:
        test_dataset = DataSet(test_directory,config=config, test_set = True)
        test_dataset.load_next_chunk()
        test_dataset.load_bilby_samples()

        # load precomputed samples
        bilby_samples = []
        for sampler in config["testing"]["samplers"][1:]:
            bilby_samples.append(test_dataset.sampler_outputs[sampler])
        bilby_samples = np.array(bilby_samples)

    start_epoch = 0
    
    model = CVAE(config, device=device).to(device)

    model.forward(torch.ones((2, model.n_channels, model.y_dim)).to(device), torch.ones((2, model.x_dim)).to(device))
    with open(os.path.join(config["output"]["output_directory"], "model_summary.txt"),"w") as f:
        summary = torchsummary.summary(model, [(model.n_channels, model.y_dim), (model.x_dim, )], depth = 3)


    if config["training"]["transfer_model_checkpoint"] and not config["training"]["resume_training"]:
        checkpoint = torch.load(config["training"]["transfer_model_checkpoint"])
        #std = checkpoint["model_state_dict"]
        #std_new = OrderedDict((key.replace("shared_conv", "net_shared_conv") if "shared_conv" in key else key, v) for key, v in std.items())
        #model.load_state_dict(std_new)
        model.load_state_dict(checkpoint["model_state_dict"])
        print('... loading in previous model %s' % config["training"]["transfer_model_checkpoint"])

    elif config["training"]['resume_training']:
        if config["training"]["transfer_model_checkpoint"]:
            print(f"Warning: Continuing training from trained weights, not from pretrained model from {config['training']['transfer_model_checkpoint']}")
        # Load the previously saved weights
        #model = torch.load(os.path.join(checkpoint_dir,"model.pt"))
        checkpoint = torch.load(os.path.join(checkpoint_dir,"model.pt"))
        #std = checkpoint["model_state_dict"]
        #std_new = OrderedDict((key.replace("shared_conv", "net_shared_conv") if "shared_conv" in key else key, v) for key, v in std.items())
        #model.load_state_dict(std_new)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        print('... loading in previous model %s' % os.path.join(checkpoint_dir,"model.pt"))

    
    model.forward(torch.ones((2, model.n_channels, model.y_dim)).to(device), torch.ones((2, model.x_dim)).to(device))

    with open(os.path.join(config["output"]["output_directory"], "model_summary.txt"),"w") as f:
        summary = torchsummary.summary(model, [(model.n_channels, model.y_dim), (model.x_dim, )], depth = 3)
        f.write(str(summary))

    if config["training"]["optimiser"] == "adam":
        optimiser = torch.optim.AdamW(model.parameters(), lr=config["training"]["initial_learning_rate"])
        if config["training"]["resume_training"]:
            optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
    else:
        raise Exception(f'Optimiser not implemented: {config["training"]["optimiser"]}')


    callbacks = []
    callbacks.append(SaveModelCallback(
        model, 
        optimiser, 
        checkpoint_dir, 
        save_interval = config["training"]["plot_interval"]))

    callbacks.append(LossPlotCallback(
        config["output"]["output_directory"], 
        checkpoint_dir, 
        save_interval = config["training"]["plot_interval"]))

    callbacks.append(LoadDataCallback(
        train_dataset, 
        config["training"]["num_epoch_load"]))

    if config["training"]["decay_lr"] or config["training"]["cycle_lr"]:
        callbacks.append(LearningRateCallback(
            optimiser, 
            config["training"]["initial_learning_rate"], 
            config["training"]["cycle_lr"], 
            config["training"]["cycle_lr_start"], 
            config["training"]["cycle_lr_length"], 
            config["training"]["cycle_lr_amp"],
            config["training"]["decay_lr"], 
            config["training"]["decay_lr_start"], 
            config["training"]["decay_lr_length"], 
            config["training"]["decay_lr_logend"]))

    if config["training"]["ramp"]:
        callbacks.append(AnnealCallback(
            model, 
            config["training"]["ramp_start"], 
            config["training"]["ramp_start"] + config["training"]["ramp_length"],
            config["training"]["ramp_n_cycles"]))

    if config["training"]["test_interval"]:
        bilby_samples = []
        for sampler in config["testing"]["samplers"][1:]:
            bilby_samples.append(test_dataset.sampler_outputs[sampler])
        bilby_samples = np.array(bilby_samples)
        
        callbacks.append(PosteriorComparisonCallback(
            config["output"]["output_directory"], 
            model, 
            bilby_samples, 
            test_dataset, 
            device = device, 
            n_samples = config["testing"]["n_samples"], 
            config=config,
            save_interval = config["training"]["test_interval"]))


    train_loop(
        model=model, 
        device=device, 
        optimiser=optimiser, 
        n_epochs=config["training"]['num_iterations'], 
        train_iterator=train_dataset, 
        validation_iterator=validation_dataset, 
        save_dir = config["output"]["output_directory"],  
        continue_train = config["training"]['resume_training'],
        start_epoch = start_epoch,
        checkpoint_dir=checkpoint_dir,
        callbacks = callbacks
        )



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
    import bilby

    bilby.core.utils.log.setup_logger(outdir='./', label=None, log_level='warning', print_version=False)
    
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

    setup_and_train(vitamin_config)

