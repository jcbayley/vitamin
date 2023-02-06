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
import pandas
import seaborn


def plot_losses(all_loss, epoch, run='testing'):
    """
    plots the losses
    """
    plt.figure()
    plt.semilogx(np.arange(1,epoch+1),all_loss[:epoch,1],'b',label='RECON')
    plt.semilogx(np.arange(1,epoch+1),all_loss[:epoch,2],'r',label='KL')
    plt.semilogx(np.arange(1,epoch+1),all_loss[:epoch,3],'g',label='TOTAL')
    plt.semilogx(np.arange(1,epoch+1),all_loss[:epoch,4],'--b',alpha=0.5)
    plt.semilogx(np.arange(1,epoch+1),all_loss[:epoch,5],'--r',alpha=0.5)
    plt.semilogx(np.arange(1,epoch+1),all_loss[:epoch,6],'--g',alpha=0.5)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    plt.ylim([np.min(1.1*all_loss[int(0.1*epoch):epoch,1:]),np.max(1.1*all_loss[int(0.1*epoch):epoch,1:])])
    plt.savefig('%s/loss.png' % (run))
    plt.close()

    # save loss data to text file
    #loss_file = '%s/loss.txt' % (run)
    #data = np.concatenate([train_loss[:epoch,:],val_loss[:epoch,:]],axis=1)
    #return data
    #np.savetxt(loss_file,data)

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


def plot_losses_zoom(all_loss, epoch, ind_start, run='testing',label="TOTAL"):
    """
    plots the losses
    """
    plt.figure()
    if label == "TOTAL":
        ind = 3
    elif label == "RECON":
        ind = 1
    elif label == "KL":
        ind = 2
    #plt.semilogx(np.arange(1,epoch+1)[ind_start:],train_loss[ind_start:epoch,0],'b',label='RECON')
    #plt.semilogx(np.arange(1,epoch+1)[ind_start:],train_loss[ind_start:epoch,1],'r',label='KL')
    plt.semilogx(np.arange(1,epoch+1)[ind_start:],all_loss[ind_start:epoch,ind],'g',label="{}".format(label))
    #plt.semilogx(np.arange(1,epoch+1)[ind_start:],val_loss[ind_start:epoch,0],'--b',alpha=0.5)
    #plt.semilogx(np.arange(1,epoch+1)[ind_start:],val_loss[ind_start:epoch,1],'--r',alpha=0.5)
    plt.semilogx(np.arange(1,epoch+1)[ind_start:],all_loss[ind_start:epoch,ind+3],'--g',alpha=0.5)
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

def plot_JS_div(par_vals, labels):
    seaborn.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize = (14,8))
    hm = seaborn.heatmap(np.array(par_vals)*1e3, annot=True, fmt='0.1f', annot_kws = {"fontsize":11}, cmap="cividis", cbar_kws={'label': 'JS divergence ($10^{-3}$)'}, linewidths=0.05)
    ax.set_xticks(np.arange(15) + 0.5,labels=labels, rotation=50)
    ax.set_ylabel("Injection", fontsize=20)
    ax.collections[0].colorbar.set_label('JS divergence ($10^{-3}$)', fontsize=20)
    plt.show()
    fig.savefig(os.path.join(""))

def compute_JS_div(vitamin_samples, sampler_samples, Nsamp=5000, nstep=100, ntest=100):
    """compute KL estimate
        take the mean JS divergence of {nstep} random draws of {Nsamp} samples from each of the sets of samples
        samples should be of shape (nsampes, nparameters)
    """
    temp_JS = np.zeros((ntest, np.shape(vitamin_samples)[1]))
    if np.shape(vitamin_samples)[1] != np.shape(sampler_samples)[1]:
        raise Exception("Samples should have the same number of parameters")
    SMALL_CONST = 1e-162
    def my_kde_bandwidth(obj, fac=1.0):
        """We use Scott's Rule, multiplied by a constant factor."""
        return np.power(obj.n, -1./(obj.d+4)) * fac
        
    for n in range(ntest):
        idx1 = np.random.randint(0,vitamin_samples.shape[0],Nsamp)
        idx2 = np.random.randint(0,sampler_samples.shape[0],Nsamp)
        for pr in range(np.shape(vitamin_samples)[1]):
            kdsampp = vitamin_samples[idx1, pr:pr+1][~np.isnan(vitamin_samples[idx1, pr:pr+1])].flatten()
            kdsampq = sampler_samples[idx2, pr:pr+1][~np.isnan(sampler_samples[idx2, pr:pr+1])].flatten()
            eval_points = np.linspace(np.min([np.min(kdsampp), np.min(kdsampq)]), np.max([np.max(kdsampp), np.max(kdsampq)]), nstep)
            kde_p = st.gaussian_kde(kdsampp)(eval_points)
            kde_q = st.gaussian_kde(kdsampq)(eval_points)
            current_JS = np.power(jensenshannon(kde_p, kde_q),2)
            temp_JS[n][pr] = current_JS

    temp_JS = np.mean(temp_JS, axis = 0)

    return temp_JS

def plot_pp(samples, injection_parameters, confidence_interval=[0.68, 0.95, 0.997],
                 lines=None, keys=None, 
                 confidence_interval_alpha=0.1, weight_list=None,
                 **kwargs):
    """
    Make a P-P plot for a set of runs with injected signals. (adapted from bilby)

    Parameters
    ==========
    results: list
        A list of Result objects, each of these should have injected_parameters
    filename: str, optional
        The name of the file to save, the default is "outdir/pp.png"
    save: bool, optional
        Whether to save the file, default=True
    confidence_interval: (float, list), optional
        The confidence interval to be plotted, defaulting to 1-2-3 sigma
    lines: list
        If given, a list of matplotlib line formats to use, must be greater
        than the number of parameters.
    legend_fontsize: float
        The font size for the legend
    keys: list
        A list of keys to use, if None defaults to search_parameter_keys
    confidence_interval_alpha: float, list, optional
        The transparency for the background condifence interval
    weight_list: list, optional
        List of the weight arrays for each set of posterior samples.
    kwargs:
        Additional kwargs to pass to matplotlib.pyplot.plot

    Returns
    =======
    fig, pvals:
        matplotlib figure and a NamedTuple with attributes `combined_pvalue`,
        `pvalues`, and `names`.
    """
    import bilby
    results = []
    for sp in range(len(samples)):
        res = bilby.result.Result()
        post = pandas.DataFrame(data = samples[sp], columns = labels)
        res.posterior = post
        res.search_parameter_keys = labels
        res.injection_parameters = {labels[i]:injection_parameters[sp][i] for i in range(len(labels))}
        #res.priors = {labels[i]:bilby.prior.Gaussian(0,1, name=labels[i]) for i in range(len(labels))}
        results.append(res)

    if keys is None:
        keys = results[0].search_parameter_keys

    if weight_list is None:
        weight_list = [None] * len(results)

    credible_levels = pandas.DataFrame()
    for i, result in enumerate(results):
        credible_levels = credible_levels.append(
            result.get_all_injection_credible_levels(keys),ignore_index=True)

    if lines is None:
        colors = ["C{}".format(i) for i in range(8)]
        linestyles = ["-", "--", ":"]
        lines = ["{}{}".format(a, b) for a, b in product(linestyles, colors)]
    if len(lines) < len(credible_levels.keys()):
        raise ValueError("Larger number of parameters than unique linestyles")

    x_values = np.linspace(0, 1, 1001)

    N = len(credible_levels)

    if isinstance(confidence_interval, float):
        confidence_interval = [confidence_interval]
    if isinstance(confidence_interval_alpha, float):
        confidence_interval_alpha = [confidence_interval_alpha] * len(confidence_interval)
    elif len(confidence_interval_alpha) != len(confidence_interval):
        raise ValueError(
            "confidence_interval_alpha must have the same length as confidence_interval")

    lowers = []
    uppers = []
    for ci, alpha in zip(confidence_interval, confidence_interval_alpha):
        edge_of_bound = (1. - ci) / 2.
        lower = st.binom.ppf(1 - edge_of_bound, N, x_values) / N
        upper = st.binom.ppf(edge_of_bound, N, x_values) / N
        # The binomial point percent function doesn't always return 0 @ 0,
        # so set those bounds explicitly to be sure
        lower[0] = 0
        upper[0] = 0
        lowers.append(lower)
        uppers.append(upper)

    pvalues = []
    pps = []
    for ii, key in enumerate(credible_levels):
        pp = np.array([sum(credible_levels[key].values < xx) /
                       len(credible_levels) for xx in x_values])
        pvalue = st.kstest(credible_levels[key], 'uniform').pvalue
        pvalues.append(pvalue)
        pps.append(pp)

        try:
            name = results[0].priors[key].latex_label
        except AttributeError:
            name = key
        label = "{} ({:2.3f})".format(name, pvalue)

    Pvals = namedtuple('pvals', ['combined_pvalue', 'pvalues', 'names'])
    pvals = Pvals(combined_pvalue=st.combine_pvalues(pvalues)[1],
                  pvalues=pvalues,
                  names=list(credible_levels.keys()))

    return pvals, x_values, pps, lowers, uppers

def plot_posterior(samples,x_truth,epoch,idx,all_other_samples=None, config=None, scale_other_samples = True, unconvert_parameters = None):
    """
    plots the posteriors
    """
    # samples shape [numsamples, numpar]
    # all other shape [numsamplers, numsamples, numpar]

    config["data"]['corner_labels'] = config["data"]["prior_pars"]

    # make save directories
    directories = {}
    for dname in ["comparison_posteriors", "full_posteriors","JS_divergence","samples"]:
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
        return [-1.0] * len(config["testing"]['samplers'][1:])
    # make numpy arrays
    samples = samples.numpy()
    #print(np.min(samples, axis=0), np.max(samples, axis = 0))
    #samples_file = os.path.join(directories["samples"],'posterior_samples_epoch_{}_event_{}_normed.txt'.format(epoch,idx))
    #np.savetxt(samples_file,samples)

    # convert vitamin sample and true parameter back to truths
    if unconvert_parameters is not None:
        vit_samples = unconvert_parameters(config, samples)
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
            if np.all(other_samples) == 0:
                continue
            #sampler_samples = np.zeros([other_samples.shape[0],config["masks"]["bilby_ol_len"]])
            #true_params = np.zeros(config["masks"]["inf_ol_len"])
            #vitamin_samples = np.zeros([vit_samples.shape[0],config["masks"]["inf_ol_len"]])
            sampler_samples = np.zeros([other_samples.shape[0],config["masks"]["inf_bilby_len"]])
            true_params = np.zeros(config["masks"]["inf_bilby_len"])
            vitamin_samples = np.zeros([vit_samples.shape[0],config["masks"]["inf_bilby_len"]])
            ol_pars = []
            cnt = 0
            #for inf_idx,bilby_idx in zip(config["masks"]["inf_ol_idx"],config["masks"]["bilby_ol_idx"]):
            for inf_idx, bilby_idx in config["masks"]["inf_bilby_idx"]:
                inf_par = config["model"]['inf_pars_list'][inf_idx]
                bilby_par = config["testing"]['bilby_pars'][bilby_idx]
                #print(inf_par, np.min(vit_samples[:, inf_idx]), np.max(vit_samples[:,inf_idx]))
                if inf_par == "geocent_time":
                    vitamin_samples[:,cnt] = vit_samples[:,inf_idx] - config["data"]["ref_geocent_time"]
                    sampler_samples[:,cnt] = other_samples[:,bilby_idx] - config["data"]["ref_geocent_time"]
                    true_params[cnt] = x_truth[inf_idx] - config["data"]["ref_geocent_time"]
                #elif inf_par == "phase":
                #    continue
                else:
                    vitamin_samples[:,cnt] = vit_samples[:,inf_idx]
                    sampler_samples[:,cnt] = other_samples[:,bilby_idx]

                true_params[cnt] = x_truth[inf_idx]
                ol_pars.append(inf_par)
                cnt += 1
            parnames = []
            for k_idx,k in enumerate(config["data"]['prior_pars']):
                if np.isin(k, ol_pars):
                    parnames.append(config["data"]['corner_labels'][k_idx])

            # convert to RA
            #vit_samples = convert_hour_angle_to_ra(vit_samples,params,ol_pars)
            #true_x = convert_hour_angle_to_ra(np.reshape(true_x,[1,true_XS.shape[1]]),params,ol_pars).flatten()
            #old_true_post = true_post                 

            samples_file = os.path.join(directories["samples"],'vitamin_posterior_samples_epoch_{}_event_{}.txt'.format(epoch,idx))
            np.savetxt(samples_file,vitamin_samples)

            temp_JS = compute_JS_div(vitamin_samples, sampler_samples)
            JS_est.append(temp_JS)

            other_samples_file = os.path.join(directories["samples"],'posterior_samples_epoch_{}_event_{}_{}.txt'.format(epoch,idx,i))
            np.savetxt(other_samples_file,sampler_samples)

            if i == 0:
                figure = corner.corner(sampler_samples, **defaults_kwargs,labels=ol_pars,
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
                plt.annotate('JS_{} = {:.3f}'.format(ol_pars[j1],JS_ind),(0.2 + 0.05*j1,0.95-j*0.02 - j1*0.02),xycoords='figure fraction',fontsize=18)

        corner.corner(vitamin_samples,**defaults_kwargs,
                      color='tab:red',
                      fill_contours=False, truths=true_params,
                      show_titles=True, fig=figure, hist_kwargs=hist_kwargs)
        
        if epoch == 'pub_plot':
            print('Saved output to', os.path.join(directories["comparison_posteriors"],'comp_posterior_{}_event_{}.png'.format(epoch,idx)))
            plt.savefig(os.path.join(directories["comparison_posteriors"],'comp_posterior_{}_event_{}.png'.format(epoch,idx)))
        else:
            print('Saved output to ', os.path.join(directories["comparison_posteriors"],'comp_posterior_epoch_{}_event_{}.png'.format(epoch,idx)))
            plt.savefig(os.path.join(directories["comparison_posteriors"],'comp_posterior_epoch_{}_event_{}.png'.format(epoch,idx)))
        plt.close()
        return JS_est, ol_pars


    else:
        # Get corner parnames to use in plotting labels
        parnames = []
        for k_idx,k in enumerate(config["data"]['rand_pars']):
            if np.isin(k, config["model"]['inf_pars_list']):
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

    z_q = np.array(z_q)
    z_r1 = np.array(z_r1)

    #z_q[np.isinf(z_q)] = np.nan
    #z_r1[np.isinf(z_r1)] = np.nan

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
