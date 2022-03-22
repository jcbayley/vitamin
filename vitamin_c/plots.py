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


def plot_posterior(samples,x_truth,epoch,idx,run='testing',all_other_samples=None, params = None, masks= None, bounds = None, scale_other_samples = True, unconvert_parameters = None):
    """
    plots the posteriors
    """

    # trim samples from outside the cube
    mask = []
    for s in samples:
        if (np.all(s>=0.0) and np.all(s<=1.0)):
            mask.append(True)
        else:
            mask.append(False)

    samples = tf.boolean_mask(samples,mask,axis=0)
    print('identified {} good samples'.format(samples.shape[0]))
    print(np.array(all_other_samples).shape)
    if samples.shape[0]<100:
        print('... Bad run, not doing posterior plotting.')
        return [-1.0] * len(params['samplers'][1:])
    # make numpy arrays
    samples = samples.numpy()
    #x_truth = x_truth.numpy()

    samples_file = '{}/posterior_samples_epoch_{}_event_{}_normed.txt'.format(run,epoch,idx)
    np.savetxt(samples_file,samples)

    # convert vitamin sample and true parameter back to truths
    if unconvert_parameters is not None:
        vit_samples = unconvert_parameters(samples)
    else:
        vit_samples = samples
    #x_truth = np.squeeze(unconvert_parameters(np.expand_dims(x_truth, axis=0)), axis=0)
    print("truth", x_truth)
    #print("samp_shape",np.shape(vit_samples), np.shape(x_truth))
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
            true_post = np.zeros([other_samples.shape[0],masks["bilby_ol_len"]])
            true_x = np.zeros(masks["inf_ol_len"])
            true_XS = np.zeros([vit_samples.shape[0],masks["inf_ol_len"]])
            ol_pars = []
            cnt = 0
            for inf_idx,bilby_idx in zip(masks["inf_ol_idx"],masks["bilby_ol_idx"]):
                inf_par = params['inf_pars'][inf_idx]
                bilby_par = params['bilby_pars'][bilby_idx]
                true_XS[:,cnt] = vit_samples[:,inf_idx]
                true_post[:,cnt] = other_samples[:,bilby_idx]
                true_x[cnt] = x_truth[inf_idx]

                ol_pars.append(inf_par)
                cnt += 1
            parnames = []
            for k_idx,k in enumerate(params['rand_pars']):
                if np.isin(k, ol_pars):
                    parnames.append(params['corner_labels'][k])

            # convert to RA
            #vit_samples = convert_hour_angle_to_ra(vit_samples,params,ol_pars)
            #true_x = convert_hour_angle_to_ra(np.reshape(true_x,[1,true_XS.shape[1]]),params,ol_pars).flatten()
            old_true_post = true_post                 

            samples_file = '{}/posterior_samples_epoch_{}_event_{}_vit.txt'.format(run,epoch,idx)
            np.savetxt(samples_file,true_XS)

            # compute KL estimate
            idx1 = np.random.randint(0,true_XS.shape[0],3000)
            idx2 = np.random.randint(0,true_post.shape[0],3000)
            temp_JS = []
            SMALL_CONST = 1e-162
            def my_kde_bandwidth(obj, fac=1.0):
                """We use Scott's Rule, multiplied by a constant factor."""
                return np.power(obj.n, -1./(obj.d+4)) * fac

            for pr in range(np.shape(true_XS)[1]):
                #try:
                kdsampp = true_XS[idx1, pr:pr+1][~np.isnan(true_XS[idx1, pr:pr+1])].flatten()
                kdsampq = true_post[idx1, pr:pr+1][~np.isnan(true_post[idx1, pr:pr+1])].flatten()
                eval_pointsp = np.linspace(np.min(kdsampp), np.max(kdsampp), len(kdsampp))
                eval_pointsq = np.linspace(np.min(kdsampq), np.max(kdsampq), len(kdsampq))
                kde_p = st.gaussian_kde(kdsampp)(eval_pointsp)
                kde_q = st.gaussian_kde(kdsampq)(eval_pointsq)
                #kde_q = st.gaussian_kde(kdsampq, bw_method=my_kde_bandwidth)(kdsampq)

                #kl_1 = 1./(len(kdsampp))*np.sum(kde_p(kdsampp)*np.log((kde_p(kdsampp) + SMALL_CONST)/(kde_q(kdsampp) + SMALL_CONST)))
                #kl_2 = 1./(len(kdsampq))*np.sum(kde_q(kdsampq)*np.log((kde_q(kdsampq) + SMALL_CONST)/(kde_p(kdsampq) + SMALL_CONST)))
                current_JS = np.power(jensenshannon(kde_p, kde_q),2)
                #kl_1 = 1./(len(kdsampp))*np.sum(np.log((kde_p(kdsampp) + SMALL_CONST)/(kde_q(kdsampp) + SMALL_CONST)))
                #kl_2 = 1./(len(kdsampq))*np.sum(np.log((kde_q(kdsampq) + SMALL_CONST)/(kde_p(kdsampq) + SMALL_CONST)))
                                                                
                #current_JS = kl_1 + kl_2
                #current_JS = 0.5*(estimate(true_XS[idx1,pr:pr+1],true_post[idx2,pr:pr+1],n_jobs=4) + estimate(true_post[idx2,pr:pr+1],true_XS[idx1,pr:pr+1],n_jobs=4))
                #except:
                #    current_JS = -1.0

                temp_JS.append(current_JS)

            JS_est.append(temp_JS)

            other_samples_file = '{}/posterior_samples_epoch_{}_event_{}_{}.txt'.format(run,epoch,idx,i)
            np.savetxt(other_samples_file,true_post)


            print("true_samples_shape", np.shape(true_post))
            if i==0:
                figure = corner.corner(true_post, **defaults_kwargs,labels=parnames,
                           color='tab:blue',
                           show_titles=True, hist_kwargs=hist_kwargs_other)
            else:
                """
                # compute KL estimate
                idx1 = np.random.randint(0,old_true_post.shape[0],2000)
                idx2 = np.random.randint(0,true_post.shape[0],2000)
                
                try:
                    current_KL = 0.5*(estimate(old_true_post[idx1,:],true_post[idx2,:],n_jobs=4) + estimate(true_post[idx2,:],old_true_post[idx1,:],n_jobs=4))
                except:
                    current_KL = -1.0
                    pass
                
                #current_KL=-1
                KL_est.append(current_KL)
                """
                corner.corner(true_post,**defaults_kwargs,
                           color='tab:green',
                           show_titles=True, fig=figure, hist_kwargs=hist_kwargs_other2)
        
        JS_file = '{}/JS_divergence_epoch_{}_event_{}_{}.txt'.format(run,epoch,idx,i)
        np.savetxt(JS_file,JS_est)

        for j,JS in enumerate(JS_est):    
            for j1,JS_ind in enumerate(JS):
                plt.annotate('JS_{} = {:.3f}'.format(parnames[j1],JS_ind),(0.2 + 0.05*j1,0.95-j*0.02 - j1*0.02),xycoords='figure fraction',fontsize=18)

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
        return JS_est

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
            new_samples[:,inf_par_idx] = samples[:,inf_par_idx]# * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par+'_min']
            if x_truth is not None:
                full_true_x[inf_par_idx] = x_truth[inf_par_idx]# * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par + '_min']
            else:
                full_true_x = None
        #new_samples = convert_hour_angle_to_ra(new_samples,params,params['inf_pars'])
        if full_true_x is not None:
            print(np.shape(full_true_x))
            print(samples.shape)
            #full_true_x = convert_hour_angle_to_ra(np.reshape(full_true_x,[1,samples.shape[1]]),params,params['inf_pars']).flatten()       

        figure = corner.corner(new_samples,**defaults_kwargs,labels=parnames,
                           color='tab:red',
                           fill_contours=True, truths=full_true_x,
                           show_titles=True, hist_kwargs=hist_kwargs)
        if epoch == 'pub_plot':
            plt.savefig('%s/full_posterior_%s_event_%d.png' % (run,epoch,idx))
        else:
            plt.savefig('%s/full_posterior_epoch_%d_event_%d.png' % (run,epoch,idx))
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
