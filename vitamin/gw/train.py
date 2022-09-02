
def train(config):

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
    #from keras_adamw import AdamW
    import tensorflow as tf
    import tensorflow_addons as tfa
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    from tensorflow.keras import regularizers
    from ..vitamin_model import CVAE
    from ..callbacks import  PlotCallback, TrainCallback, TestCallback, TimeCallback, OptimizerSave, LearningRateCallback, LogminRampCallback, AnnealCallback, BatchRampCallback
    from .load_data import DataLoader, convert_ra_to_hour_angle, convert_hour_angle_to_ra, psiphi_to_psiX, psiX_to_psiphi, m1m2_to_chirpmassq, chirpmassq_to_m1m2
    from keras_adabound import AdaBound
        
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
    checkpoint_path = os.path.join(config["output"]["output_directory"],"checkpoint","model")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    dirs = [checkpoint_dir]
    for direc in dirs:
        if not os.path.isdir(direc):
            os.makedirs(direc)

    make_paper_plots = config["testing"]['make_paper_plots']
    hyper_par_tune = False

    # load the training data
    if not make_paper_plots:
        train_dataset = DataLoader(training_directory,config = config) 
        validation_dataset = DataLoader(validation_directory,config=config,val_set = True)
        train_dataset.load_next_chunk()
        validation_dataset.load_next_chunk()

        #enq = tf.keras.utils.OrderedEnqueuer(train_dataset)
        #enq.start(workers = 4)
        
    if config["training"]["test_interval"] != False:
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

    if config["training"]["optimiser"] == "adam":
        optimizer = tfa.optimizers.AdamW(learning_rate=config["training"]["initial_learning_rate"], weight_decay = 1e-8, clipvalue = 5)
    elif config["training"]["optimiser"] == "sgd":
        optimizer = tf.keras.optimizers.SGD(config["training"]["initial_learning_rate"], clipvalue = 5)
    elif config["training"]["optimiser"] == "adabound":
        optimizer = AdaBound(lr=config["training"]["initial_learning_rate"], final_lr=config["training"]["final_learning_rate"], clipvalue = 5)
    elif config["training"]["optimiser"] == "lookahead":
        optimizer = tfa.optimizers.AdamW(learning_rate=config["training"]["initial_learning_rate"], weight_decay = 1e-8, clipvalue = 5)
        optimizer = tfa.optimizers.Lookahead(optimizer)

    #optimizer = tf.keras.optimizers.Adam(config["training"]["initial_learning_rate"])

    # Keras hyperparameter optimization
    if hyper_par_tune:
        import keras_hyper_optim
        del model
        keras_hyper_optim.main(train_dataset, val_dataset)
        exit()

    # compile and build the model (hardcoded values will change soon)
    model.compile(run_eagerly = None, optimizer = optimizer, loss = model.compute_loss)

    if config["training"]["transfer_model_checkpoint"] and not config["training"]["resume_training"]:
        model.load_weights(config["training"]["transfer_model_checkpoint"])
        #model = tf.keras.models.load_model(config["training"]["transfer_model_checkpoint"])
        """
        with open(os.path.join(checkpoint_dir, "optimizer.pkl"),"rb") as f:
            weight_values = pickle.load(f)
        model.optimizer.set_weights(weight_values)
        """
        print('... loading in previous model %s' % config["training"]["transfer_model_checkpoint"])

    elif config["training"]['resume_training']:
        if config["training"]["transfer_model_checkpoint"]:
            print(f"Warning: Continuing training from trained weights, not from pretrained model from {config['training']['transfer_model_checkpoint']}")
        # Load the previously saved weights
        #latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(checkpoint_path)
        #model = tf.keras.models.load_model(checkpoint_path)
        """
        with open(os.path.join(checkpoint_dir, "optimizer.pkl"),"rb") as f:
            weight_values = pickle.load(f)
        for i,w in enumerate(model.optimizer.weights):
            print(i, np.shape(w))
            if i > 10:break
            #if np.shape(w) == (64,2,96):
            #    print(i,np.shape(w))
        for i,w in enumerate(weight_values):
            print(i, np.shape(w))
            if i > 10:break
            #if np.shape(w) == (64,2,96):
            #    print(i,np.shape(w))
                
        model.optimizer.set_weights(np.roll(weight_values[1:], -1))
        """
        print('... loading in previous model %s' % checkpoint_path)
        with open(os.path.join(config["output"]['output_directory'], "loss.txt"),"r") as f:
            start_epoch = len(np.loadtxt(f))


    #model([test_data, test_pars])
    #model.build([(None, 1024,2), (None, 15)])

    with open(os.path.join(config["output"]["output_directory"], "model_summary.txt"),"w") as f:
        model.encoder_r1.summary(print_fn=lambda x: f.write(x + '\n'))
        model.encoder_q.summary(print_fn=lambda x: f.write(x + '\n'))
        model.decoder_r2.summary(print_fn=lambda x: f.write(x + '\n'))
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        save_freq=10*config["training"]["chunk_batch"],
        options=None,
        initial_value_threshold=None,
    )


    callbacks = [checkpoint]
    callbacks.append(PlotCallback(config["output"]["output_directory"], epoch_plot=config["training"]["plot_interval"],start_epoch=start_epoch))
    callbacks.append(TrainCallback(config, optimizer, train_dataset, model))
    callbacks.append(TimeCallback(config))#, OptimizerSave(config, checkpoint_dir, 10)]

    if config["training"]["cycle_lr"] or config["training"]["decay_lr"]:
        lr_call = LearningRateCallback(config["training"]["initial_learning_rate"], cycle_lr = config["training"]["cycle_lr"], cycle_lr_start = config["training"]["cycle_lr_start"], cycle_lr_length=config["training"]["cycle_lr_length"], cycle_lr_amp=config["training"]["cycle_lr_amp"], decay_lr=config["training"]["decay_lr"], decay_lr_start=config["training"]["decay_lr_start"], decay_lr_length=config["training"]["decay_lr_length"], decay_lr_logend = config["training"]["decay_lr_logend"])
        callbacks.append(lr_call)

    if config["training"]["logvarmin_ramp"]:
        lmr_call = LogminRampCallback(logvarmin_ramp_start=config["training"]["logvarmin_ramp_start"], logvarmin_ramp_length=config["training"]["logvarmin_ramp_length"], logvarmin_start=config["training"]["logvarmin_start"], logvarmin_end=config["training"]["logvarmin_end"], model=model)
        callbacks.append(lmr_call)

    if config["training"]["ramp_length"] != 0:
        ann_call = AnnealCallback(ramp_start=config["training"]["ramp_start"], ramp_length=config["training"]["ramp_length"])
        callbacks.append(ann_call)

    if config["training"]["batch_ramp"]:
        batch_call = BatchRampCallback(batch_ramp_start=config["training"]["batch_ramp_start"], batch_ramp_length=config["training"]["batch_ramp_length"], batch_size=config["training"]["batch_size"], batch_size_end=config["training"]["batch_size_end"])
        callbacks.append(batch_call)

    if config["training"]["test_interval"] != False:
        pass
        callbacks.append(TestCallback(config, test_dataset, bilby_samples))
        
    if config["training"]["tensorboard_log"]:
        logdir = os.path.join(config["output"]["output_directory"], "profile")
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir = logdir,histogram_freq = 50,profile_batch = 200,update_freq = 500))
    
    model.fit(train_dataset, use_multiprocessing = False, workers = 1, epochs = config["training"]["num_iterations"], callbacks = callbacks, shuffle = False, validation_data = validation_dataset, max_queue_size = 1, initial_epoch = start_epoch)
        


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

    train(vitamin_config)

