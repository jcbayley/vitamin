import matplotlib.pyplot as plt
import numpy as np
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
import tensorflow as tf
from tensorflow.keras import regularizers
from scipy.spatial.distance import jensenshannon
import scipy.stats as st


def plot_losses(train_loss, val_loss, epoch, run='testing'):
    """
    plots the losses
    """
    plt.figure()
    plt.semilogx(np.arange(1,epoch+1),train_loss[:epoch,0],'b',label='RECON')
    plt.semilogx(np.arange(1,epoch+1),train_loss[:epoch,1],'r',label='KL')
    plt.semilogx(np.arange(1,epoch+1),train_loss[:epoch,2],'g',label='TOTAL')
    plt.semilogx(np.arange(1,epoch+1),val_loss[:epoch,0],'--b',alpha=0.5)
    plt.semilogx(np.arange(1,epoch+1),val_loss[:epoch,1],'--r',alpha=0.5)
    plt.semilogx(np.arange(1,epoch+1),val_loss[:epoch,2],'--g',alpha=0.5)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    plt.ylim([np.min(1.1*train_loss[int(0.1*epoch):epoch,:]),np.max(1.1*train_loss[int(0.1*epoch):epoch,:])])
    plt.savefig('%s/loss.png' % (run))
    plt.close()

    # save loss data to text file
    loss_file = '%s/loss.txt' % (run)
    data = np.concatenate([train_loss[:epoch,:],val_loss[:epoch,:]],axis=1)
    np.savetxt(loss_file,data)

def plot_batch_losses(train_loss, epoch, run='testing'):
    """
    plots the losses
    """
    fig, ax = plt.subplots()
    ax.plot(train_loss[:,0],'b',label='RECON')
    ax.plot(train_loss[:,1],'r',label='KL')
    ax.plot(train_loss[:,2],'g',label='TOTAL')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_yscale("symlog")
    ax.legend()
    ax.grid()
    #plt.ylim([np.min(1.1*train_loss[int(0.1*epoch):epoch,:]),np.max(1.1*train_loss[int(0.1*epoch):epoch,:])])
    fig.savefig('%s/batch_loss.png' % (run))
    plt.close(fig)

def plot_gauss_von(train_loss, epoch, run='testing'):
    """
    plots the losses
    """
    fig, ax = plt.subplots()
    ax.plot(train_loss[:,0],'b',label='Gauss')
    ax.plot(train_loss[:,1],'r',label='Vonm')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_yscale("symlog")
    ax.legend()
    ax.grid()
    #plt.ylim([np.min(1.1*train_loss[int(0.1*epoch):epoch,:]),np.max(1.1*train_loss[int(0.1*epoch):epoch,:])])
    fig.savefig('%s/batch_gauss_von_loss.png' % (run))
    plt.close(fig)


def plot_losses_zoom(train_loss, val_loss, epoch, ind_start, run='testing',label="TOTAL"):
    """
    plots the losses
    """
    plt.figure()
    if label == "TOTAL":
        ind = 2
    elif label == "RECON":
        ind = 0
    elif label == "KL":
        ind = 1
    #plt.semilogx(np.arange(1,epoch+1)[ind_start:],train_loss[ind_start:epoch,0],'b',label='RECON')
    #plt.semilogx(np.arange(1,epoch+1)[ind_start:],train_loss[ind_start:epoch,1],'r',label='KL')
    plt.semilogx(np.arange(1,epoch+1)[ind_start:],train_loss[ind_start:epoch,ind],'g',label="{}".format(label))
    #plt.semilogx(np.arange(1,epoch+1)[ind_start:],val_loss[ind_start:epoch,0],'--b',alpha=0.5)
    #plt.semilogx(np.arange(1,epoch+1)[ind_start:],val_loss[ind_start:epoch,1],'--r',alpha=0.5)
    plt.semilogx(np.arange(1,epoch+1)[ind_start:],val_loss[ind_start:epoch,ind],'--g',alpha=0.5)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    #plt.ylim([np.min(1.1*train_loss[int(0.1*epoch):epoch,:]),np.max(1.1*train_loss[int(0.1*epoch):epoch,:])])
    plt.savefig('{}/loss_zoom_{}.png'.format(run,label))
    plt.close()


 
def plot_KL(KL_samples, step, run='testing'):
    """
    plots the KL evolution
    """
    # arrives in shape n_kl,n_test,3
    N = KL_samples.shape[0]
#    KL_samples = np.transpose(KL_samples,[2,1,0])   # re-order axes
#    print(list(np.linspace(0,len(params['samplers'][1:])-1,num=len(params['samplers'][1:]), dtype=int))[::-1])
#    print(KL_samples.shape)
#    exit()
#    KL_samples = np.transpose(KL_samples, list(np.linspace(0,len(params['samplers'][1:])-1,num=len(params['samplers'][1:]), dtype=int))[::-1])    
    ls = ['-','--',':']
    c = ['C0','C1','C2','C3']
    fig, axs = plt.subplots(3, sharex=True, sharey=True, figsize=(6.4,14.4))
    for i,kl_s in enumerate(KL_samples):   # loop over samplers
        for j,kl in enumerate(kl_s):     # loop over test cases
            axs[i].semilogx(np.arange(1,N+1)*step,kl,ls[i],color=c[j])
            axs[i].plot(N*step,kl[-1],'.',color=c[j])
            axs[i].grid()
    plt.xlabel('epoch')
    plt.ylabel('KL')
    plt.ylim([-0.2,1.0])
    plt.savefig('%s/kl.png' % (run))
    plt.close()


def plot_posterior(samples,x_truth,epoch,idx,all_other_samples=None, config=None, scale_other_samples = True, unconvert_parameters = None):
    """
    plots the posteriors
    """

    config["data"]['corner_labels'] = config["data"]['rand_pars']

    # make save directories
    directories = {}
    for dname in ["comparison_posterior", "full_posterior","JS_divergence","samples"]:
        directories[dname] = os.path.join(config["output"]["output_directory"], dname)
        if not os.path.isdir(directories[dname]):
            os.makedirs(directories[dname])
    
    # trim samples from outside the cube
    mask = []
    for s in samples:
        if (np.all(s>=0.0) and np.all(s<=1.0)):
            mask.append(True)
        else:
            mask.append(False)

    samples = tf.boolean_mask(samples,mask,axis=0)
    print('identified {} good samples'.format(samples.shape[0]))

    if samples.shape[0]<100:
        print('... Bad run, not doing posterior plotting.')
        return [-1.0] * len(config["test"]['samplers'][1:])
    # make numpy arrays
    samples = samples.numpy()

    #samples_file = os.path.join(directories["samples"],'posterior_samples_epoch_{}_event_{}_normed.txt'.format(epoch,idx))
    #np.savetxt(samples_file,samples)

    # convert vitamin sample and true parameter back to truths
    if unconvert_parameters is not None:
        vit_samples = unconvert_parameters(samples)
    else:
        vit_samples = samples
        
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
        JS_est = []
        for i, other_samples in enumerate(all_other_samples):
            sampler_samples = np.zeros([other_samples.shape[0],config["masks"]["bilby_ol_len"]])
            true_params = np.zeros(config["masks"]["inf_ol_len"])
            vitamin_samples = np.zeros([vit_samples.shape[0],config["masks"]["inf_ol_len"]])
            ol_pars = []
            cnt = 0
            for inf_idx,bilby_idx in zip(config["masks"]["inf_ol_idx"],config["masks"]["bilby_ol_idx"]):
                inf_par = config["model"]['inf_pars'][inf_idx]
                bilby_par = config["testing"]['bilby_pars'][bilby_idx]
                vitamin_samples[:,cnt] = vit_samples[:,inf_idx]
                sampler_samples[:,cnt] = other_samples[:,bilby_idx]
                true_params[cnt] = x_truth[inf_idx]
                ol_pars.append(inf_par)
                cnt += 1
            parnames = []
            for k_idx,k in enumerate(config["data"]['rand_pars']):
                if np.isin(k, ol_pars):
                    parnames.append(config["data"]['corner_labels'][k])

            # convert to RA
            #vit_samples = convert_hour_angle_to_ra(vit_samples,params,ol_pars)
            #true_x = convert_hour_angle_to_ra(np.reshape(true_x,[1,true_XS.shape[1]]),params,ol_pars).flatten()
            old_true_post = true_post                 

            samples_file = os.path.join(directories["samples"],'vitamin_posterior_samples_epoch_{}_event_{}.txt'.format(epoch,idx))
            np.savetxt(samples_file,vitamin_samples)

            # compute KL estimate
            idx1 = np.random.randint(0,vitamin_samples.shape[0],3000)
            idx2 = np.random.randint(0,sampler_samples.shape[0],3000)
            temp_JS = []
            SMALL_CONST = 1e-162
            def my_kde_bandwidth(obj, fac=1.0):
                """We use Scott's Rule, multiplied by a constant factor."""
                return np.power(obj.n, -1./(obj.d+4)) * fac

            for pr in range(np.shape(vitamin_samples)[1]):
                #try:
                kdsampp = vitamin_samples[idx1, pr:pr+1][~np.isnan(vitamin_samples[idx1, pr:pr+1])].flatten()
                kdsampq = sampler_samples[idx1, pr:pr+1][~np.isnan(sampler_samples[idx1, pr:pr+1])].flatten()
                eval_pointsp = np.linspace(np.min(kdsampp), np.max(kdsampp), len(kdsampp))
                eval_pointsq = np.linspace(np.min(kdsampq), np.max(kdsampq), len(kdsampq))
                kde_p = st.gaussian_kde(kdsampp)(eval_pointsp)
                kde_q = st.gaussian_kde(kdsampq)(eval_pointsq)
                current_JS = np.power(jensenshannon(kde_p, kde_q),2)
                #current_JS = 0.5*(estimate(true_XS[idx1,pr:pr+1],true_post[idx2,pr:pr+1],n_jobs=4) + estimate(true_post[idx2,pr:pr+1],true_XS[idx1,pr:pr+1],n_jobs=4))
                temp_JS.append(current_JS)

            JS_est.append(temp_JS)

            other_samples_file = os.path.join(directories["samples"],'posterior_samples_epoch_{}_event_{}_{}.txt'.format(epoch,idx,i))
            np.savetxt(other_samples_file,sampler_samples)

            if i == 0:
                figure = corner.corner(sampler_samples, **defaults_kwargs,labels=parnames,
                                       color='tab:blue',
                                       show_titles=True, hist_kwargs=hist_kwargs_other)
            else:
                corner.corner(sampler_samples,**defaults_kwargs,
                              color='tab:green',
                              show_titles=True, fig=figure, hist_kwargs=hist_kwargs_other2)
        
        JS_file = os.path.join(directories["JS_divergence"],'JS_divergence_epoch_{}_event_{}_{}.txt'.format(epoch,idx,i))
        np.savetxt(JS_file,JS_est)

        for j,JS in enumerate(JS_est):    
            for j1,JS_ind in enumerate(JS):
                plt.annotate('JS_{} = {:.3f}'.format(parnames[j1],JS_ind),(0.2 + 0.05*j1,0.95-j*0.02 - j1*0.02),xycoords='figure fraction',fontsize=18)

        corner.corner(vitamin_samples,**defaults_kwargs,
                      color='tab:red',
                      fill_contours=False, truths=true_x,
                      show_titles=True, fig=figure, hist_kwargs=hist_kwargs)
        if epoch == 'pub_plot':
            print('Saved output to %s/comp_posterior_%s_event_%d.png' % (save_dir,epoch,idx))
            plt.savefig(os.path.join(directories["comparison_posterior"],'comp_posterior_{}_event_{}.png'.format(epoch,idx)))
        else:
            print('Saved output to %s/comp_posterior_epoch_%d_event_%d.png' % (save_dir,epoch,idx))
            plt.savefig(os.path.join(directories["comparison_posterior"],'comp_posterior_epoch_{}_event_{}.png'.format(epoch,idx)))
        plt.close()
        return JS_est

    else:
        # Get corner parnames to use in plotting labels
        parnames = []
        for k_idx,k in enumerate(config["data"]['rand_pars']):
            if np.isin(k, config["model"]['inf_pars']):
                parnames.append(config["data"]['corner_labels'][k_idx])

        figure = corner.corner(samples,**defaults_kwargs,labels=parnames,
                           color='tab:red',
                           fill_contours=True, truths=x_truth,
                           show_titles=True, hist_kwargs=hist_kwargs)
        if epoch == 'pub_plot':
            plt.savefig(os.path.join(directories["full_posterior"],'full_posterior_{}_event_{}.png'.format(save_dir,epoch,idx)))
        else:
            plt.savefig(os.path.join(directories["full_posterior"],'full_posterior_epoch_{}_event_{}.png'.format(epoch,idx)))
        plt.close()
    return -1.0

def plot_latent(mu_r1, z_r1, mu_q, z_q, epoch, idx, run='testing'):

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

    
    figure = corner.corner(np.array(z_q), **defaults_kwargs,
                           color='tab:blue',
                           show_titles=True, hist_kwargs=hist_kwargs_other)
    corner.corner(np.array(z_r1),**defaults_kwargs,
                           color='tab:red',
                           fill_contours=True,
                           show_titles=True, fig=figure, hist_kwargs=hist_kwargs)
    # Extract the axes
    z_dim = z_r1.shape[1]
    axes = np.array(figure.axes).reshape((z_dim, z_dim))

    # Loop over the histograms
    for yi in range(z_dim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.plot(mu_r1[0,:,xi], mu_r1[0,:,yi], "sr")
            ax.plot(mu_q[0,xi], mu_q[0,yi], "sb")
    if epoch == 'pub_plot':
        plt.savefig('%s/latent_%s_event_%d.png' % (run,epoch,idx))
    else:
        plt.savefig('%s/latent_epoch_%d_event_%d.png' % (run,epoch,idx))
    plt.close()

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
