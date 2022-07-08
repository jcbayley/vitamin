#!/usr/bin/env python
import numpy as np
from gwpy.timeseries import TimeSeries
import bilby
import os
import sys
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.signal as signal
import corner
import random
import copy
import time 

def run_test(data_dir, out_dir, sample_rate, duration):

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    
    with open(os.path.join(data_dir,"meta.json"),"r") as f:
        data_info = json.load(f)
    
    # set the times for signal
    trigger_time = data_info["trigger"]["geocent"]#1325030418.015282
    pad = 0.5
    end_time = trigger_time + 0.15*duration 
    start_time = end_time - duration

    print("START TIME: {}".format(start_time + 0.5))

    # set times for psd
    roll_off = 0.4  
    psd_duration = 16 * duration
    psd_start_time = start_time - psd_duration
    psd_end_time = start_time


    #params_dir = "../vitamin_c/params_files_256_2"
    params_dir = model_dir
    params_file = os.path.join(params_dir,'params.json')                          
    bounds_file = os.path.join(params_dir,'bounds.json')                                  
    fixed_vals_file = os.path.join(params_dir,'fixed_vals.json')                    

    # Load parameters files                                                                                               
    with open(params_file, 'r') as fp:                                                                                                            
        params = json.load(fp)                                                                             
    with open(bounds_file, 'r') as fp:                                                                                
        bounds = json.load(fp)                      
    fixed_vals = None
    #with open(fixed_vals_file, 'r') as fp:                                                                                                         
    #    fixed_vals = json.load(fp)


    params["ref_geocent_time"] = start_time + duration/2.0 # + 0.005#+ 12*3600

    asd_file = "/home/joseph.bayley/projects/o4_online_pe_mdc/data/asd_files/aLIGO_O4_high_asd.txt"

    truths = [data_info["injection"]["mass_1"],
              data_info["injection"]["mass_2"],
              data_info["injection"]["luminosity_distance"],
              data_info["injection"]["geocent_time"],
              data_info["injection"]["phase"],
              data_info["injection"]["theta_jn"],
              data_info["injection"]["psi"],
              data_info["injection"]["a_1"],
              data_info["injection"]["a_2"],
              data_info["injection"]["tilt_1"],
              data_info["injection"]["tilt_2"],
              data_info["injection"]["phi_12"],
              data_info["injection"]["phi_jl"],
              data_info["injection"]["ra"],
              data_info["injection"]["dec"]]
    
    ifo_list1 = bilby.gw.detector.InterferometerList([])
    ifo_list2 = bilby.gw.detector.InterferometerList([])
    for i, det in enumerate(["H1","L1"]):

        ifo1 = bilby.gw.detector.get_empty_interferometer(det)
        ifo2 = bilby.gw.detector.get_empty_interferometer(det)
        #file_name = "/home/joseph.bayley/data/CBC/O4MDC/o4_online/U1/{}-O4MDC-1325029266-1184.gwf".format(det)
        #file_name = os.path.join(data_dir,"{}-O4MDC-1325029268-1152.gwf".format(det))
        #file_name = os.path.join(data_dir,"{}-O4MDC-1325029268-1152.gwf".format(det))

        file_end = data_info[det]["filename"].split("/")[-1]
        file_name = os.path.join(data_dir,file_end)
        channel_name = data_info[det]["channel"]
        #channel_name = "{}:O4MDC".format(det)
        

        # load timeseries from start to end time for signal
        ts = TimeSeries.read(file_name,channel_name, start = start_time, end = end_time)
        ts2 = TimeSeries.read(file_name,channel_name, start = start_time-pad, end = end_time+pad)
        ts_resamp = ts.resample(sample_rate)
        ts_resamp2 = ts2.resample(sample_rate)
        #ts_resamp1 = ts_resamp.crop(start=start_time, end = end_time, copy = True)
        #ts_resamp2 = ts_resamp.crop(start=start_time-pad, end = end_time+pad, copy = True)

        fig, ax = plt.subplots( figsize = (20,10))
        ax.plot(np.array(ts_resamp.data))
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        fig.savefig(os.path.join(out_dir,"waveform_pre_wh_{}.png".format(i)))

        fig, ax = plt.subplots( figsize = (20,10))
        ax.plot(np.array(ts_resamp2.data))
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        fig.savefig(os.path.join(out_dir,"waveform_pre_wh_{}_wide.png".format(i)))


        ifo1.strain_data.set_from_gwpy_timeseries(ts_resamp)
        ifo2.strain_data.set_from_gwpy_timeseries(ts_resamp2)

        # set psd as same from o4
        ifo1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=asd_file)
        ifo2.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=asd_file)
        
        """
        # load data for psd
        psd_data = TimeSeries.read(file_name,channel_name, start = psd_start_time, end = psd_end_time)

        # calcualte the psd
        psd_alpha = 2 * roll_off / duration
        psd = psd_data.psd(
            fftlength=duration,
            overlap=0,
            window=("tukey", psd_alpha),
            method="median"
        )

        ifo1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=psd.frequencies.value, psd_array=psd.value)
        ifo2.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=psd.frequencies.value, psd_array=psd.value)
        """
        
        ifo_list1.append(ifo1)
        ifo_list2.append(ifo2)

    whitened_h1 = []
    whitened_h2 = []
    for ind in range(2):
        ifo1 = ifo_list1[ind]
        ifo2 = ifo_list2[ind]

        Nt1 = int(ifo1.sampling_frequency*ifo1.duration)
        # whiten the data by dividing by psd
        #h_fd1 = ifo1.strain_data.frequency_domain_strain
        h_fd1, h_fa1 = bilby.utils.nfft(ifo1.strain_data.time_domain_strain, ifo1.strain_data.sampling_frequency)
        whitened_h_fd1 = h_fd1/ifo1.amplitude_spectral_density_array
        whitened_h_td1 = np.sqrt(2.0*Nt1)*np.fft.irfft(whitened_h_fd1)

        whitened_h1.append([whitened_h_td1/params["y_normscale"]])


        Nt2 = int(ifo2.sampling_frequency*(ifo2.duration))
        # whiten the data by dividing by psd
        h_fd2 = ifo2.strain_data.frequency_domain_strain
        whitened_h_fd2 = h_fd2/ifo2.amplitude_spectral_density_array
        whitened_h_td2 = np.sqrt(2.0*Nt1)*np.fft.irfft(whitened_h_fd2)

        whitened_h2.append([whitened_h_td2[int(sample_rate*pad):int(len(whitened_h_td2) - int(sample_rate*pad))]/params["y_normscale"]])

    whitened_h1 = np.transpose(whitened_h1, [1,2,0])
    whitened_h2 = np.transpose(whitened_h2, [1,2,0])

    fig, ax = plt.subplots(nrows = 2, figsize = (20,10))
    ax[0].plot(whitened_h1[0][:,0])
    ax[1].plot(whitened_h1[0][:,1])
    ax[1].set_xlabel("Time")
    ax[0].set_ylabel("Amplitude")
    ax[1].set_ylabel("Amplitude")
    fig.savefig(os.path.join(out_dir,"waveform1.png"))
    
    fig, ax = plt.subplots(nrows = 2, figsize = (20,10))
    ax[0].plot(whitened_h2[0][:,0])
    ax[1].plot(whitened_h2[0][:,1])
    ax[0].axvline(0.85*sample_rate*duration,color="r")
    ax[0].axvline((0.85 - 0.005)*sample_rate*duration,color="g")
    ax[0].axvline((params["ref_geocent_time"] - start_time)*sample_rate,color="r")
    ax[0].axvline((params["ref_geocent_time"] - start_time - 0.005)*sample_rate,color="g")
    ax[0].axvline(0.5*sample_rate*duration,color="y",ls="--")
    ax[1].axvline(0.85*sample_rate*duration)
    ax[1].set_xlabel("Time")
    ax[0].set_ylabel("Amplitude")
    ax[1].set_ylabel("Amplitude")
    fig.savefig(os.path.join(out_dir,"waveform2.png"))
    """
    fig, ax = plt.subplots(nrows = 2, figsize = (20,10))
    ax[0].plot(whitened_h1[0][:,0] - whitened_h2[0][:,0])
    ax[1].plot(whitened_h1[0][:,1] - whitened_h2[0][:,1])
    ax[1].set_xlabel("Time")
    ax[0].set_ylabel("Amplitude difference")
    ax[1].set_ylabel("Amplitude difference")
    fig.savefig(os.path.join(out_dir,"waveform_diff.png"))
    """


    fig, ax = plt.subplots(nrows = 2, figsize = (20,10))
    ax[0].plot(tf.keras.activations.tanh(whitened_h1[0][:,0]))
    ax[1].plot(tf.keras.activations.tanh(whitened_h1[0][:,1]))
    ax[1].set_xlabel("Time")
    ax[0].set_ylabel("Amplitude")
    ax[1].set_ylabel("Amplitude")
    fig.savefig(os.path.join(out_dir,"waveform_norm1.png"))
    """
    fig, ax = plt.subplots(nrows = 2, figsize = (20,10))
    ax[0].plot(tf.keras.activations.tanh(whitened_h2[0][:,0]))
    ax[1].plot(tf.keras.activations.tanh(whitened_h2[0][:,1]))
    ax[1].set_xlabel("Time")
    ax[0].set_ylabel("Amplitude")
    ax[1].set_ylabel("Amplitude")
    fig.savefig(os.path.join(out_dir,"waveform_norm2.png"))
    """

    #params["ref_geocent_time"] += 12*3600
    params, bounds, masks, fixed_vals = get_params(params, bounds, fixed_vals)

    model = CVAE(params, bounds, masks)
    
    # load in trained model
    print("loading model weights....")
    latest = tf.train.latest_checkpoint(os.path.join(model_dir,"checkpoint"))
    model.load_weights(latest)
    
    print("generating samples....")

    samples1 = model.gen_samples(tf.convert_to_tensor(whitened_h1), nsamples = 5000)

    start_time = time.time()
    samples2t = model.gen_samples(tf.convert_to_tensor(whitened_h2), nsamples = 15000)
    end_time_1 = time.time()

    samples2 = samples2t.numpy()
    start_mask = time.time()
    mask1 = np.all((samples2>=0.0) & (samples2<=1.0), axis = 1)
    samples = samples2[mask1]
    end_mask = time.time()

    # randonly convert from reduced psi-phi space to full space
    _, psi_idx, _ = get_param_index(params['inf_pars'],['psi'])
    _, phi_idx, _ = get_param_index(params['inf_pars'],['phase'])
    psi_idx = psi_idx[0]
    phi_idx = phi_idx[0]
    samples[:,psi_idx], samples[:,phi_idx] = psiX_to_psiphi(samples[:,psi_idx], samples[:,phi_idx])

    start_resize = time.time()
    true_XS = np.zeros([samples.shape[0],masks["inf_ol_len"]])
    ol_pars = []
    cnt = 0
    for inf_idx in masks["inf_ol_idx"]:
        inf_par = params['inf_pars'][inf_idx]
        true_XS[:,cnt] = (samples[:,inf_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par+'_min']
        
        if inf_par == "geocent_time":
            true_XS[:,cnt] += params["ref_geocent_time"]
        if inf_par == "ra":
            true_XS[:,cnt] = np.remainder(true_XS[:,cnt], 2*np.pi)
        cnt += 1
    start_time_resamp = time.time()
    # resample distance to different prior
    new_distprior = bilby.gw.prior.BBHPriorDict()["luminosity_distance"]
    old_distprior = bilby.gw.prior.Uniform(name='luminosity_distance', minimum=bounds['luminosity_distance_min'], maximum=bounds['luminosity_distance_max'], unit='Mpc')
    prior_weights = []
    for ind in range(len(true_XS)):
        val = new_distprior.prob(true_XS[ind][2])/old_distprior.prob(true_XS[ind][2])
        prior_weights.append(val)
    #true_XS = true_XS[inds]
    #print(np.shape(true_XS))
    #print(np.shape(prior_weights))
    #true_XS = np.array(random.choices(true_XS, weights = prior_weights, k = len(true_XS) - 1))

    end_time = time.time()

    with open(os.path.join(out_dir, "sample_gen_time.txt"),"w+") as f:
        f.write("total_time: {} \n".format(end_time - start_time))
        f.write("sample_time: {} \n".format(end_time_1 - start_time))
        f.write("resample_time: {} \n".format(end_time - start_time_resamp))
        f.write("rescale_time: {} \n".format(start_time_resamp - start_resize))
        f.write("X-psiphi_time: {} \n".format(start_resize - end_mask))
        f.write("mask_time: {} \n".format(end_mask - start_mask))
        
    pars = ["mass_1","mass_2","luminosity_distance","geocent_time","phase","theta_jn","psi","a_1","a_2","tilt_1","tilt_2","phi_12","phi_jl","ra","dec"]
    
    params["bilby_pars"] = pars
    masks["inf_ol_mask"], masks["inf_ol_idx"], masks["inf_ol_len"] = get_param_index(params['inf_pars'],pars)
    masks["bilby_ol_mask"], masks["bilby_ol_idx"], masks["bilby_ol_len"] = get_param_index(params['bilby_pars'],pars)
    
    with open("/home/joseph.bayley/data/CBC/O4MDC/michael_posterior/DEV0_pesummary.dat","r") as f:
        samples_m = f.readlines()


    inds = []
    for par in pars:
        for ind,allp in enumerate(samples_m[0].strip("\n").split("\t")):
            if par == allp:
                inds.append(ind)
                break

    samples_m2 = []
    for ind in range(len(samples_m) - 1):
    
        temp_samp = np.array(samples_m[ind + 1].strip("\n").split("\t")).astype(float)[inds]
        samples_m2.append(temp_samp)
    

    with open("/home/joseph.bayley/data/CBC/O4MDC/greg_bilby/O4MDC_U1_data0_1325030417-924_analysis_H1L1_bilby_mcmc_merge_result.json","r") as f:
        samples_g = json.load(f)

    samples_g2 = []
    for par in pars:
        samples_g2.append(samples_g["posterior"]["content"][par])
    samples_g2 = np.array(samples_g2).T

    plot_posterior_n(samples1, truths,1, 1,run=out_dir, params=params, bounds = bounds, masks=masks, all_other_samples = [np.array(samples_g2),])
    plot_posterior_n(samples2t, truths,1, 2,run=out_dir, params=params, bounds = bounds, masks=masks, all_other_samples = [np.array(samples_g2),])
    plot_posterior_n(samples2t, truths,1, 3,run=out_dir, params=params, bounds = bounds, masks=masks, all_other_samples = [np.array(samples_m2),])
    params["bilby_pars"] = pars
    masks["bilby_ol_idx"] = masks["inf_ol_idx"]
    plot_posterior(samples2t, truths,1, 4,run=out_dir, params=params, bounds = bounds, masks=masks, all_other_samples = [np.array(samples_g2),], scale_other_samples=False)



def plot_posterior_n(samples,x_truth,epoch,idx,run='testing',all_other_samples=None, params = None, masks= None, bounds = None, transform_others = True):
    """
    plots the posteriors
    """

    # trim samples from outside the cube
    samples = samples.numpy()
    mask = np.all((samples>=0.0) & (samples<=1.0), axis = 1)
    samples = samples[mask]
    """
    mask = []
    for s in samples:
        if (np.all(s>=0.0) and np.all(s<=1.0)):
            mask.append(True)
        else:
            mask.append(False)
    samples = tf.boolean_mask(samples,mask,axis=0)
    """
    print('identified {} good samples'.format(samples.shape[0]))
    print(np.array(all_other_samples).shape)
    if samples.shape[0]<100:
        print('... Bad run, not doing posterior plotting.')
        return [-1.0] * len(params['samplers'][1:])

    # randonly convert from reduced psi-phi space to full space
    _, psi_idx, _ = get_param_index(params['inf_pars'],['psi'])
    _, phi_idx, _ = get_param_index(params['inf_pars'],['phase'])
    psi_idx = psi_idx[0]
    phi_idx = phi_idx[0]
    samples[:,psi_idx], samples[:,phi_idx] = psiX_to_psiphi(samples[:,psi_idx], samples[:,phi_idx])

    # define general plotting arguments
    defaults_kwargs = dict(
                    bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
                    title_kwargs=dict(fontsize=16),
                    truth_color='tab:orange', quantiles=[0.16, 0.84],
                    levels=(0.68,0.90,0.95), density=True,
                    plot_density=False, plot_datapoints=True,
                    max_n_ticks=3)

    # 1-d hist kwargs for normalisation
    hist_kwargs = dict(density=True,color='tab:red')
    hist_kwargs_other = dict(density=True,color='tab:blue')
    hist_kwargs_other2 = dict(density=True,color='tab:green')

    if all_other_samples is not None:
        KL_est = []
        for i, other_samples in enumerate(all_other_samples):
            true_post = np.zeros([other_samples.shape[0],masks["bilby_ol_len"]])
            true_x = np.zeros(masks["inf_ol_len"])
            true_XS = np.zeros([samples.shape[0],masks["inf_ol_len"]])
            ol_pars = []
            cnt = 0
            for inf_idx,bilby_idx in zip(masks["inf_ol_idx"],masks["bilby_ol_idx"]):
                inf_par = params['inf_pars'][inf_idx]
                bilby_par = params['bilby_pars'][bilby_idx]
                true_XS[:,cnt] = (samples[:,inf_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par+'_min']
                true_post[:,cnt] = other_samples[:,inf_idx]#(other_samples[:,bilby_idx] * (bounds[bilby_par+'_max'] - bounds[bilby_par+'_min'])) + bounds[bilby_par + '_min']

                if x_truth is not None:
                    true_x[cnt] = x_truth[inf_idx]#(x_truth[inf_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par + '_min']
                else:
                    true_x = None

                if inf_par == "geocent_time":
                    true_XS[:,cnt] += params["ref_geocent_time"]
                    #true_post[:,cnt] -= params["ref_geocent_time"]
                    #true_x[cnt] -= params["ref_geocent_time"]
                if inf_par == "ra":
                    true_XS[:,cnt] = np.remainder(true_XS[:,cnt], 2*np.pi)

                ol_pars.append(inf_par)
                cnt += 1
            parnames = []
            for k_idx,k in enumerate(params['rand_pars']):
                if np.isin(k, ol_pars):
                    parnames.append(params['corner_labels'][k])

            # convert to RA
            #true_XS = convert_hour_angle_to_ra(true_XS,params,ol_pars)

            #true_x = convert_hour_angle_to_ra(np.reshape(true_x,[1,true_XS.shape[1]]),params,ol_pars).flatten()
            old_true_post = true_post                 

            
            # resample distance to different prior
            new_distprior = bilby.gw.prior.BBHPriorDict()["luminosity_distance"]
            old_distprior = bilby.gw.prior.Uniform(name='luminosity_distance', minimum=bounds['luminosity_distance_min'], maximum=bounds['luminosity_distance_max'], unit='Mpc')
            prior_weights = []
            for ind in range(len(true_XS)):
                val = new_distprior.prob(true_XS[ind][2])/old_distprior.prob(true_XS[ind][2])
                prior_weights.append(val)
            #prior_weights = prior_weights/max(prior_weights)
            #r_num = np.random.rand(len(prior_weights))
            #inds = np.where(prior_weights > r_num)[0]
            #true_XS = true_XS[inds]
            true_XS = np.array(random.choices(true_XS, weights = prior_weights, k = len(true_XS)))

            samples_file = '{}/posterior_samples_epoch_{}_event_{}_vit.txt'.format(run,epoch,idx)
            np.savetxt(samples_file,true_XS)

            # compute KL estimate
            idx1 = np.random.randint(0,true_XS.shape[0],2000)
            idx2 = np.random.randint(0,true_post.shape[0],2000)
            """
            try:
                current_KL = 0.5*(estimate(true_XS[idx1,:],true_post[idx2,:],n_jobs=4) + estimate(true_post[idx2,:],true_XS[idx1,:],n_jobs=4))
            except:
                current_KL = -1.0
                pass
            """
            current_KL = -1
            KL_est.append(current_KL)

            other_samples_file = '{}/posterior_samples_epoch_{}_event_{}_{}.txt'.format(run,epoch,idx,i)
            np.savetxt(other_samples_file,true_post)

            if i==0:
                figure = corner.corner(true_post, **defaults_kwargs,labels=parnames,
                           color='tab:blue',
                           show_titles=True, hist_kwargs=hist_kwargs_other)
            else:

                # compute KL estimate
                idx1 = np.random.randint(0,old_true_post.shape[0],2000)
                idx2 = np.random.randint(0,true_post.shape[0],2000)
                """
                try:
                    current_KL = 0.5*(estimate(old_true_post[idx1,:],true_post[idx2,:],n_jobs=4) + estimate(true_post[idx2,:],old_true_post[idx1,:],n_jobs=4))
                except:
                    current_KL = -1.0
                    pass
                """
                current_KL=-1
                KL_est.append(current_KL)

                corner.corner(true_post,**defaults_kwargs,
                           color='tab:green',
                           show_titles=True, fig=figure, hist_kwargs=hist_kwargs_other2)
        
        for j,KL in enumerate(KL_est):    
            plt.annotate('KL = {:.3f}'.format(KL),(0.2,0.95-j*0.02),xycoords='figure fraction',fontsize=18)

        if np.any(true_XS[:,-2] < 0):
            print("LESS THAN ZERO")
        if np.any(true_XS[:,13] < 0):
            print("LESS THAN ZERO")

        corner.corner(true_XS,**defaults_kwargs,
                           color='tab:red',
                           fill_contours=False, truths=true_x,
                           show_titles=True, fig=figure, hist_kwargs=hist_kwargs)
        if epoch == 'pub_plot':
            print('Saved output to %s/comp_posterior_%s_event_%d.png' % (run,epoch,idx))
            plt.savefig('%s/comp_posterior_%s_event_%d.png' % (run,epoch,idx))
        else:
            print('Saved output to %s/comp_posterior_epoch_%d_event_%d.png' % (run,epoch,idx))
            plt.savefig('%s/comp_posterior_epoch_%d_event_%d.png' % (run,epoch,idx))
        plt.close()
        return KL_est

    else:
        # Get corner parnames to use in plotting labels
        parnames = []
        for k_idx,k in enumerate(params['rand_pars']):
            if np.isin(k, params['inf_pars']):
                parnames.append(params['corner_labels'][k])
        # un-normalise full inference parameters
        full_true_x = np.zeros(len(params['inf_pars']))
        new_samples = np.zeros([samples.shape[0],len(params['inf_pars'])])
        for inf_par_idx,inf_par in enumerate(params['inf_pars']):
            new_samples[:,inf_par_idx] = (samples[:,inf_par_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par+'_min']

            if inf_par == "geocent_time":
                new_samples[:,inf_par_idx] += params["ref_geocent_time"]

            full_true_x[inf_par_idx] = x_truth[inf_par_idx]

        new_samples = convert_hour_angle_to_ra(new_samples,params,params['inf_pars'])

        figure = corner.corner(new_samples,**defaults_kwargs,labels=parnames,
                           color='tab:red',
                           fill_contours=True, truths=full_true_x,
                           show_titles=True, hist_kwargs=hist_kwargs)
        if epoch == 'pub_plot':
            plt.savefig('%s/full_posterior_%s_event_%d.png' % (run,epoch,idx))
        else:
            plt.savefig('%s/full_posterior_epoch_%d_event_%d.png' % (run,epoch,idx))
        plt.close()
    return -1#new_samples


if __name__ == "__main__":

    sample_rate = 1024
    duration = 2
    #model_dir = "/home/joseph.bayley/public_html/CBC/vitamin_O4MDC/vitamin_fit_c_run5_2048_tanhdata_allupdates_larger_conv_batchnorm/"
    #model_dir = "/home/joseph.bayley/public_html/CBC/vitamin_O4MDC/vitamin_2048Hz_1s_wide_spinmass/vitamin_fit_c_run4_2048/"
    #model_dir = "/home/joseph.bayley/public_html/CBC/vitamin_O4MDC/vitamin_c_2048_2s/vitamin_run1"
    model_dir = "/home/joseph.bayley/public_html/CBC/vitamin_O4MDC/vitamin_c_2048_2s/vitamin_run2"
    #model_dir = "/home/joseph.bayley/public_html/CBC/vitamin_O4MDC/vitamin_fit_c_run6_1024_tanhdata_widespin"
    sys.path.append(model_dir)
    from vitamin_c_model_fit import CVAE
    from load_data_fit import convert_hour_angle_to_ra, convert_ra_to_hour_angle, psiphi_to_psiX, psiX_to_psiphi
    #sys.path.append("../vitamin_c/")
    sys.path.remove(model_dir)
    sys.path.append("../vitamin_c/")
    from vitamin_c_fit import get_params, get_param_index, plot_posterior
    run = "U1"
    out_dir = "/home/joseph.bayley/public_html/CBC/vitamin_O4MDC/{}_1184_{}_{}s_dyn/".format(run,sample_rate, duration)
    data_dir = "/home/joseph.bayley/data/CBC/O4MDC/o4_online/{}/".format(run)
    #data_dir = "/home/joseph.bayley/projects/o4_online_pe_mdc/data/{}/".format(run)
    run_test(data_dir,out_dir, sample_rate, duration)
