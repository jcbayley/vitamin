#!/usr/bin/env python
######################################################################################################################

# -- Variational Inference for Gravitational wave Parameter Estimation --


#######################################################################################################################

import warnings
warnings.filterwarnings("ignore")
import os
from os import path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import scipy.io as sio
import h5py
import sys
from sys import exit
import shutil
import bilby
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from time import strftime
import corner
import glob
from matplotlib.lines import Line2D
import pandas as pd
import logging.config
from contextlib import contextmanager
import json
from lal import GreenwichMeanSiderealTime
import random

import skopt
from skopt import gp_minimize, forest_minimize, dump
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args

try:
    from .gen_benchmark_pe import run, gen_real_noise
    from . import plotting
    from . import vitamin_c_fit as vitamin_c
    from .plotting import prune_samples
except (ModuleNotFoundError, ImportError):
    from gen_benchmark_pe import run, gen_real_noise
    import plotting
    import vitamin_c_fit as vitamin_c
    from plotting import prune_samples

# Check for optional basemap installation
try:
    from mpl_toolkits.basemap import Basemap
    print("module 'basemap' is installed")
except (ModuleNotFoundError, ImportError):
    print("module 'basemap' is not installed")
    print("Skyplotting functionality is automatically disabled.")
    skyplotting_usage = False
else:
    skyplotting_usage = True
    try:
        from .skyplotting import plot_sky
    except:
        from skyplotting import plot_sky

""" Script has several main functions:
1.) Generate training data
2.) Generate testing data
3.) Train model
4.) Test model
5.) Generate samples only given model and timeseries
6.) Apply importance sampling to VItamin results
"""

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def gen_rnoise(params=None,bounds=None,fixed_vals=None):
    """ Generate real noise over requested time segment

    Parameters
    ----------
    params: dict
        Dictionary containing run parameters
    bounds: dict
        Dictionary containing allowed bounds of GW source parameters
    fixed_vals: dict
        Dictionary containing the fixed values of GW source parameters

    """

    # Check for requried parameters files
    if params == None or bounds == None or fixed_vals == None:
        print('Missing either params file, bounds file or fixed vals file')
        exit()

    # Load parameters files
    with open(params, 'r') as fp:
        params = json.load(fp)
    with open(bounds, 'r') as fp:
        bounds = json.load(fp)
    with open(fixed_vals, 'r') as fp:
        fixed_vals = json.load(fp)

    # Make training set directory
    os.system('mkdir -p %s' % params['train_set_dir'])

    # Make directory for plots
    os.system('mkdir -p %s/latest_%s' % (params['plot_dir'],params['run_label']))

    print()
    print('... Making real noise samples')
    print()

    # continue producing noise samples until requested number has been fullfilled
    stop_flag = False; idx = time_cnt = 0
    start_file_seg = params['real_noise_time_range'][0]
    end_file_seg = None
    # iterate until we get the requested number of real noise samples
    while idx <= params['tot_dataset_size'] or stop_flag:


        file_samp_idx = 0
        # iterate until we get the requested number of real noise samples per tset_split files
        while file_samp_idx <= params['tset_split']:
            real_noise_seg = [start_file_seg+idx+time_cnt, start_file_seg+idx+time_cnt+1]
            real_noise_data = np.zeros((int(params['tset_split']),int( params['ndata']*params['duration'])))        

            try:
                # make the data - shift geocent time to correct reference
                real_noise_data[file_samp_idx, :] = gen_real_noise(params['duration'],params['ndata'],params['det'],
                                        params['ref_geocent_time'],params['psd_files'],
                                        real_noise_seg=real_noise_seg
                                        )
                print('Found segment')
            except ValueError as e:
                print(e)
                time_cnt+=1
                continue
            print(real_noise_data)
            exit()

        print("Generated: %s/data_%d-%d.h5py ..." % (params['train_set_dir'],start_file_seg,end_file_seg))

        # store noise sample information in hdf5 format
        with h5py.File('%s/data_%d-%d.h5py' % (params['train_set_dir'],start_file_seg,end_file_seg), 'w') as hf:
            hf.create_dataset('real_noise_samples', data=real_noise_data)
            hf.close()
        idx+=params['tset_split']

        # stop training
        if idx>params['real_noise_time_range'][1]:
            stop_flat = True  
        exit()
    return 

def gen_train(params=None,bounds=None,fixed_vals=None, num_files = 100, start_ind = 0):
    """ Generate training samples

    Parameters
    ----------
    params: dict
        Dictionary containing run parameters
    bounds: dict
        Dictionary containing allowed bounds of GW source parameters
    fixed_vals: dict
        Dictionary containing the fixed values of GW source parameters
    """

    # Check for requried parameters files
    if params == None or bounds == None or fixed_vals == None:
        print('Missing either params file, bounds file or fixed vals file')
        exit()

    # Load parameters files
    with open(params, 'r') as fp:
        params = json.load(fp)
    with open(bounds, 'r') as fp:
        bounds = json.load(fp)
    with open(fixed_vals, 'r') as fp:
        fixed_vals = json.load(fp)

    # Make training set directory
    os.system('mkdir -p %s' % params['train_set_dir'])

    # Make directory for plots
    os.system('mkdir -p %s/latest_%s' % (params['plot_dir'],params['run_label']))

    print()
    print('... Making training set')
    print()

    # Iterate over number of requested training samples
    for i in range(num_files):

        logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': True,
        })
        with suppress_stdout():
            # generate training sample source parameter, waveform and snr
            signal_train, signal_train_pars,snrs = run(sampling_frequency=params['ndata']/params['duration'],
                                                       duration=params['duration'],
                                                       N_gen=params['tset_split'],
                                                       ref_geocent_time=params['ref_geocent_time'],
                                                       bounds=bounds,
                                                       fixed_vals=fixed_vals,
                                                       rand_pars=params['rand_pars'],
                                                       seed=params['training_data_seed']+start_ind+i,
                                                       label=params['run_label'],
                                                       training=True,det=params['det'],
                                                       psd_files=params['psd_files'],
                                                       use_real_det_noise=params['use_real_det_noise'],
                                                       samp_idx=start_ind + i, params=params,
                                                       return_polarisations = False)

        logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        })
        fname = '%s/data_%d-%d.h5py' % (params['train_set_dir'],((start_ind + i)*params['tset_split']),params['tot_dataset_size'])
        print("Generated: %s ..." % (fname))

        # store training sample information in hdf5 format
        with h5py.File('%s' % (fname), 'w') as hf:
            for k, v in params.items():
                try:
                    hf.create_dataset(k,data=v)
                    hf.create_dataset('rand_pars', data=np.string_(params['rand_pars']))
                except:
                    pass
            hf.create_dataset('x_data', data=signal_train_pars)
            for k, v in bounds.items():
                hf.create_dataset(k,data=v)
            hf.create_dataset('y_data_noisy', data=np.array([]))
            hf.create_dataset('y_data_noisefree', data=signal_train)
            #hf.creat_dataset('y_hplus_hcross', data = signal_train)
            hf.create_dataset('snrs', data=snrs)
            hf.close()
    return

def gen_val(params=None,bounds=None,fixed_vals=None):
    """ Generate validation samples

    Parameters
    ----------
    params: dict
        Dictionary containing run parameters
    bounds: dict
        Dictionary containing allowed bounds of GW source parameters
    fixed_vals: dict
        Dictionary containing the fixed values of GW source parameters
    """

    # Check for requried parameters files
    if params == None or bounds == None or fixed_vals == None:
        print('Missing either params file, bounds file or fixed vals file')
        exit()

    # Load parameters files
    with open(params, 'r') as fp:
        params = json.load(fp)
    with open(bounds, 'r') as fp:
        bounds = json.load(fp)
    with open(fixed_vals, 'r') as fp:
        fixed_vals = json.load(fp)

    # Make training set directory
    os.system('mkdir -p %s' % params['val_set_dir'])

    # Make directory for plots
    os.system('mkdir -p %s/latest_%s' % (params['plot_dir'],params['run_label']))

    print()
    print('... Making validation set')
    print()


    # Iterate over number of requested training samples
    for i in range(0,params['val_dataset_size'],params['tset_split']):

        logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': True,
        })
        with suppress_stdout():
            # generate training sample source parameter, waveform and snr
            signal_train, signal_train_pars,snrs = run(sampling_frequency=params['ndata']/params['duration'],
                                                          duration=params['duration'],
                                                          N_gen=params['val_dataset_size'],
                                                          ref_geocent_time=params['ref_geocent_time'],
                                                          bounds=bounds,
                                                          fixed_vals=fixed_vals,
                                                          rand_pars=params['rand_pars'],
                                                          seed=params['validation_data_seed']+i,
                                                          label=params['run_label'],
                                                          training=True,det=params['det'],
                                                          psd_files=params['psd_files'],
                                                          use_real_det_noise=params['use_real_det_noise'])
        logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        })
        fname = "%s/data_%d-%d.h5py ..." % (params['val_set_dir'],(i + params['tset_split']),params['val_dataset_size'])
        print("Generated: %s" % fname)

        # store training sample information in hdf5 format
        with h5py.File(fname, 'w') as hf:
            for k, v in params.items():
                try:
                    hf.create_dataset(k,data=v)
                    hf.create_dataset('rand_pars', data=np.string_(params['rand_pars']))
                except:
                    pass
            hf.create_dataset('x_data', data=signal_train_pars)
            for k, v in bounds.items():
                hf.create_dataset(k,data=v)
            hf.create_dataset('y_data_noisy', data=np.array([]))
            hf.create_dataset('y_data_noisefree', data=signal_train)
            hf.create_dataset('snrs', data=snrs)
            hf.close()
    return

def gen_test(params=None,bounds=None,fixed_vals=None, sig_ind = 0):
    """ Generate testing sample time series and posteriors using Bayesian inference (bilby)

    Parameters
    ----------
    params: dict
        Dictionary containing run parameters
    bounds: dict
        Dictionary containing allowed bounds of GW source parameters
    fixed_vals: dict
        Dictionary containing the fixed values of GW source parameters
    """

    # Check for requried parameters files
    if params == None or bounds == None or fixed_vals == None:
        print('Missing either params file, bounds file or fixed vals file')
        exit()

    # Load parameters files
    with open(params, 'r') as fp:
        params = json.load(fp)
    with open(bounds, 'r') as fp:
        bounds = json.load(fp)
    with open(fixed_vals, 'r') as fp:
        fixed_vals = json.load(fp)

    # Make testing set directory
    if not os.path.isdir(params["test_set_dir"]):
        os.system('mkdir -p %s' % params['test_set_dir'])

    # Add numerical label to samplers
    for i in range(len(params['samplers'])):
        if i == 0:
            continue
        else:
            params['samplers'][i] = params['samplers'][i]+'2'

    # Make testing samples
    temp_noisy, temp_noisefree, temp_pars, temp_snr = run(sampling_frequency=params['ndata']/params['duration'],
                                                          duration=params['duration'],
                                                          N_gen=1,
                                                          ref_geocent_time=params['ref_geocent_time'],
                                                          bounds=bounds,
                                                          fixed_vals=fixed_vals,
                                                          rand_pars=params['rand_pars'],
                                                          inf_pars=params['inf_pars'],
                                                          label=params['bilby_results_label'] + '_' + str(sig_ind),
                                                          out_dir=params['pe_dir'],
                                                          samplers=params['samplers'],
                                                          training=False,
                                                          seed=params['testing_data_seed']+sig_ind,
                                                          do_pe=params['doPE'],det=params['det'],
                                                          psd_files=params['psd_files'],
                                                          use_real_det_noise=params['use_real_det_noise'],
                                                          use_real_events=params['use_real_events'],
                                                          samp_idx=sig_ind)

    signal_test_noisy = temp_noisy
    signal_test_noisefree = temp_noisefree
    signal_test_pars = temp_pars
    signal_test_snr = temp_snr
    fname = "%s/%s_%s.h5py ..." % (params['test_set_dir'],params['bilby_results_label'],sig_ind)
    print("Generated: %s ..." % (fname))
        
    # Save generated testing samples in h5py format
    with h5py.File(fname,'w') as hf:
        for k, v in params.items():
            try:
                hf.create_dataset(k,data=v) 
                hf.create_dataset('rand_pars', data=np.string_(params['rand_pars']))
            except:
                pass
        hf.create_dataset('x_data', data=signal_test_pars)
        for k, v in bounds.items():
            hf.create_dataset(k,data=v)
        hf.create_dataset('y_data_noisefree', data=signal_test_noisefree)
        hf.create_dataset('y_data_noisy', data=signal_test_noisy)
        hf.create_dataset('snrs', data=signal_test_snr)
        hf.close()
    return

def gen_real_test(params=None,bounds=None,fixed_vals=None, sig_ind = 0):
    """ Generate testing sample time series and posteriors using Bayesian inference (bilby)

    Parameters
    ----------
    params: dict
        Dictionary containing run parameters
    bounds: dict
        Dictionary containing allowed bounds of GW source parameters
    fixed_vals: dict
        Dictionary containing the fixed values of GW source parameters
    """

    # Check for requried parameters files
    if params == None or bounds == None or fixed_vals == None:
        print('Missing either params file, bounds file or fixed vals file')
        exit()

    # Load parameters files
    with open(params, 'r') as fp:
        params = json.load(fp)
    with open(bounds, 'r') as fp:
        bounds = json.load(fp)
    with open(fixed_vals, 'r') as fp:
        fixed_vals = json.load(fp)

    save_directory = "/home/joseph.bayley/public_html/CBC/vitmain_O4MDC/U1_1024_1s_dynesty/test"
    # Make testing set directory
    if not os.path.isdir(save_directory):
        os.system('mkdir -p %s' % save_directory)

    # Add numerical label to samplers
    for i in range(len(params['samplers'])):
        if i == 0:
            continue
        else:
            params['samplers'][i] = params['samplers'][i]+'1'

    # temporary placement
    real_test_data_dir = "/home/joseph.bayley/data/CBC/O4MDC/o4_online/U1/"

    # Make testing samples
    temp_noisy, temp_noisefree, temp_pars, temp_snr = run(sampling_frequency=params['ndata']/params['duration'],
                                                          duration=params['duration'],
                                                          N_gen=1,
                                                          ref_geocent_time=params['ref_geocent_time'],
                                                          bounds=bounds,
                                                          fixed_vals=fixed_vals,
                                                          rand_pars=params['rand_pars'],
                                                          inf_pars=params['inf_pars'],
                                                          label=params['bilby_results_label'] + '_' + str(sig_ind),
                                                          out_dir=save_directory,
                                                          samplers=params['samplers'],
                                                          training=False,
                                                          seed=params['testing_data_seed']+sig_ind,
                                                          do_pe=params['doPE'],det=params['det'],
                                                          psd_files=params['psd_files'],
                                                          use_real_det_noise=params['use_real_det_noise'],
                                                          use_real_events=params['use_real_events'],
                                                          samp_idx=sig_ind,
                                                          real_test_data = real_test_data_dir)

    signal_test_noisy = temp_noisy
    signal_test_noisefree = temp_noisefree
    signal_test_pars = temp_pars
    signal_test_snr = temp_snr

    fname = "%s/%s_%s.h5py ..." % (save_directory,params['bilby_results_label'],sig_ind)
    print("Generated: %s ..." % (fname))
        
    # Save generated testing samples in h5py format
    with h5py.File(fname,'w') as hf:
        for k, v in params.items():
            try:
                hf.create_dataset(k,data=v) 
                hf.create_dataset('rand_pars', data=np.string_(params['rand_pars']))
            except:
                pass
        hf.create_dataset('x_data', data=signal_test_pars)
        for k, v in bounds.items():
            hf.create_dataset(k,data=v)
        hf.create_dataset('y_data_noisefree', data=signal_test_noisefree)
        hf.create_dataset('y_data_noisy', data=signal_test_noisy)
        hf.create_dataset('snrs', data=signal_test_snr)
        hf.close()
    return


def create_dirs(dirs):
    for i in dirs:
        if not os.path.isdir(i):
            try: 
                os.makedirs(i)
            except:
                print >> sys.stderr, "Could not create directory {}".format(i)
                sys.exit(1)
    print("All directories exist")

def write_subfile(sub_filename,p,comment):
    print(sub_filename)
    with open(sub_filename,'w') as f:
        f.write('# filename: {}\n'.format(sub_filename))
        f.write('universe = vanilla\n')
        f.write('executable = {}\n'.format(p["exec"]))
        #f.write('enviroment = ""\n')
        f.write('getenv  = True\n')
        #f.write('RequestMemory = {} \n'.format(p["memory"]))
        f.write('log = {}/{}_$(cluster).log\n'.format(p["log_dir"],comment))
        f.write('error = {}/{}_$(cluster).err\n'.format(p["err_dir"],comment))
        f.write('output = {}/{}_$(cluster).out\n'.format(p["output_dir"],comment))
        args = ""
        if p["train"]:
            args += "--gen_train True "
        if p["val"]:
            args += "--gen_val True "
        if p["test"]:
            args += "--gen_test True "
        if p["real_test"]:
            args += "--gen_real_test True "
        args += "--start_ind $(start_ind) --num_files {} --params_dir {}".format( p["files_per_job"], p["params_dir"])
        f.write('arguments = {}\n'.format(args))
        f.write('accounting_group = ligo.dev.o4.cbc.explore.test\n')
        f.write('queue\n')


def make_train_dag(params, bounds, fixed_vals, params_dir, run_type = "train"):

    # Load parameters files
    with open(params, 'r') as fp:
        params = json.load(fp)
    with open(bounds, 'r') as fp:
        bounds = json.load(fp)
    with open(fixed_vals, 'r') as fp:
        fixed_vals = json.load(fp)

    p = {}
    p["root_dir"] = os.getcwd()
    p["condor_dir"] = "{}/condor".format(os.path.join(p["root_dir"],params_dir))
    p["log_dir"] = os.path.join(p["condor_dir"], "log")
    p["err_dir"] = os.path.join(p["condor_dir"], "err")
    p["output_dir"] = os.path.join(p["condor_dir"], "output")
    p["params_dir"] = os.path.join(p["root_dir"],params_dir)
    p["exec"] = os.path.join(p["root_dir"], "run_vitamin_condor.py")

    p["train"] = False
    p["val"] = False
    p["test"] = False
    p["real_test"] = False
    p[run_type] = True
    p["files_per_job"] = 20

    for direc in [p["condor_dir"], p["log_dir"], p["err_dir"], p["output_dir"]]:
        if not os.path.exists(direc):
            os.makedirs(direc)

    comment = "{}_run".format(run_type)
    run_sub_filename = os.path.join(p["condor_dir"], "{}.sub".format(comment))
    write_subfile(run_sub_filename,p,comment)


    dag_filename = "{}/{}.dag".format(p["condor_dir"],comment)
    if run_type == "train":
        num_files = int(params["tot_dataset_size"]/params["tset_split"])
        num_jobs = int(num_files/p["files_per_job"])
    elif run_type == "val":
        num_files = int(params["val_dataset_size"]/params["tset_split"])
        num_jobs = 1
    elif run_type == "test":
        num_jobs = params["r"]
    elif run_type == "real_test":
        num_jobs = 1

    with open(dag_filename,'w') as f:
        seeds = []
        for i in range(num_jobs):
            seeds.append(random.randint(1,1e9))
        for i in range(num_jobs):
            comment = "File_{}".format(i)
            uid = seeds[i]
            jobid = "{}_{}_{}".format(comment,i,uid)
            job_string = "JOB {} {}\n".format(jobid,run_sub_filename)
            retry_string = "RETRY {} 1\n".format(jobid)
            if run_type == "train":
                vars_string = 'VARS {} start_ind="{}"\n'.format(jobid,int(i*p["files_per_job"]))
            else:
                vars_string = 'VARS {} start_ind="{}"\n'.format(jobid,int(i))
            f.write(job_string)
            f.write(retry_string)
            f.write(vars_string)



if __name__ == "__main__":
    # If running module from command line

    parser = argparse.ArgumentParser(description='VItamin: A user friendly Bayesian inference machine learning library.')
    parser.add_argument("--gen_dag", default=False, help="generate the dag file for training/val/test")
    parser.add_argument("--gen_train", default=False, help="generate the training data")
    parser.add_argument("--gen_rnoise", default=False, help="generate the real noise samples")
    parser.add_argument("--gen_val", default=False, help="generate the validation data")
    parser.add_argument("--gen_test", default=False, help="generate the testing data")
    parser.add_argument("--gen_real_test", default=False, help="load and run the real testing data")
    parser.add_argument("--num_files", default=1, help="number of files to generate")
    parser.add_argument("--start_ind", default=0, help="start file index")
    parser.add_argument("--params_dir", type=str, default="./params_files", help="directory containing params files")
    parser.add_argument("--run_type", type=str, default="train", help="train, test, val for dag file")
    
    args = parser.parse_args()

    params_dir = args.params_dir

    # Define default location of the parameters files
    params = os.path.join(params_dir, 'params.json')
    bounds = os.path.join(params_dir, 'bounds.json')
    fixed_vals = os.path.join(params_dir, 'fixed_vals.json')
    
    if args.gen_dag:
        make_train_dag(params, bounds, fixed_vals, params_dir, run_type = args.run_type)
    else:
        if args.gen_train:
            gen_train(params,bounds,fixed_vals, start_ind = int(args.start_ind), num_files = int(args.num_files))
        #if args.gen_rnoise:
        #    gen_rnoise(params,bounds,fixed_vals)
        if args.gen_val:
            gen_val(params,bounds,fixed_vals)
        if args.gen_test:
            gen_test(params,bounds,fixed_vals, sig_ind = int(args.start_ind))
        if args.gen_real_test:
            gen_real_test(params,bounds,fixed_vals, sig_ind = int(args.start_ind))

