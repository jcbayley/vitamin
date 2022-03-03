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
import plotting
from tensorflow.keras import regularizers
from scipy.spatial.distance import jensenshannon
import scipy.stats as st

model_fit_type = "basic"
if model_fit_type == "multidet":
    from vitamin_c_model_fit_multidet import CVAE
elif model_fit_type == "basic":
    from vitamin_c_model_fit import CVAE
elif model_fit_type == "multiscale":
    from vitamin_c_model_fit_multiscale import CVAE
elif model_fit_type == "kaggle":
    from vitamin_c_model_fit_kaggle1 import CVAE
elif model_fit_type == "resnet":
    from vitamin_c_model_fit_resnet import CVAE
elif model_fit_type == "freq":
    from vitamin_c_model_fit_freq import CVAE

from callbacks import  PlotCallback, TrainCallback, TestCallback, TimeCallback
from load_data_fit import load_data, load_samples, convert_ra_to_hour_angle, convert_hour_angle_to_ra, DataLoader, psiphi_to_psiX, psiX_to_psiphi, m1m2_to_chirpmassq, chirpmassq_to_m1m2
#from keras_adamw import AdamW

def get_param_index(all_pars,pars,sky_extra=None):
    """ 
    Get the list index of requested source parameter types
    """
    # identify the indices of wrapped and non-wrapped parameters - clunky code
    mask = []
    idx = []

    # loop over inference params
    for i,p in enumerate(all_pars):

        # loop over wrapped params
        flag = False
        for q in pars:
            if p==q:
                flag = True    # if inf params is a wrapped param set flag

        # record the true/false value for this inference param
        if flag==True:
            mask.append(True)
            idx.append(i)
        elif flag==False:
            mask.append(False)

    if sky_extra is not None:
        if sky_extra:
            mask.append(True)
            idx.append(len(all_pars))
        else:
            mask.append(False)

    return mask, idx, np.sum(mask)



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

def get_params(params, bounds, fixed_vals, params_dir = "./params_files", print_masks=True):
    """
    params_file = os.path.join(params_dir,'params.json')
    bounds_file = os.path.join(params_dir,'bounds.json')
    fixed_vals_file = os.path.join(params_dir,'fixed_vals.json')

    EPS = 1e-3

    # Load parameters files
    with open(params_file, 'r') as fp:
        params = json.load(fp)
    with open(bounds_file, 'r') as fp:
        bounds = json.load(fp)
    with open(fixed_vals_file, 'r') as fp:
        fixed_vals = json.load(fp)
    """
    EPS = 1e-3
    # if doing hour angle, use hour angle bounds on RA                                                                                                                     
    bounds['ra_min'] = convert_ra_to_hour_angle(bounds['ra_min'],params,None,single=True)
    bounds['ra_max'] = convert_ra_to_hour_angle(bounds['ra_max'],params,None,single=True)
    print('... converted RA bounds to hour angle')
    masks = {}
    masks["inf_ol_mask"], masks["inf_ol_idx"], masks["inf_ol_len"] = get_param_index(params['inf_pars'],params['bilby_pars'])
    masks["bilby_ol_mask"], masks["bilby_ol_idx"], masks["bilby_ol_len"] = get_param_index(params['bilby_pars'],params['inf_pars'])
    
    
    # identify the indices of different sets of physical parameters                                                                                                          
    masks["vonmise_mask"], masks["vonmise_idx_mask"], masks["vonmise_len"] = get_param_index(params['inf_pars'],params['vonmise_pars'])
    masks["gauss_mask"], masks["gauss_idx_mask"], masks["gauss_len"] = get_param_index(params['inf_pars'],params['gauss_pars'])
    masks["sky_mask"], masks["sky_idx_mask"], masks["sky_len"] = get_param_index(params['inf_pars'],params['sky_pars'])
    masks["ra_mask"], masks["ra_idx_mask"], masks["ra_len"] = get_param_index(params['inf_pars'],['ra'])
    masks["dec_mask"], masks["dec_idx_mask"], masks["dec_len"] = get_param_index(params['inf_pars'],['dec'])
    masks["m1_mask"], masks["m1_idx_mask"], masks["m1_len"] = get_param_index(params['inf_pars'],['mass_1'])
    masks["m2_mask"], masks["m2_idx_mask"], masks["m2_len"] = get_param_index(params['inf_pars'],['mass_2'])
    #masks["q_mask"], masks["q_idx_mask"], masks["q_len"] = get_param_index(params['inf_pars'],['mass_ratio'])
    #masks["M_mask"], masks["M_idx_mask"], masks["M_len"] = get_param_index(params['inf_pars'],['chirp_mass'])

    #idx_mask = np.argsort(gauss_idx_mask + vonmise_idx_mask + m1_idx_mask + m2_idx_mask + sky_idx_mask) # + dist_idx_mask)                                                  
    masks["idx_mask"] = np.argsort(masks["m1_idx_mask"] + masks["m2_idx_mask"] + masks["gauss_idx_mask"] + masks["vonmise_idx_mask"]) # + sky_idx_mask)                 
    #masks["idx_mask"] = np.argsort(masks["q_idx_mask"] + masks["M_idx_mask"] + masks["gauss_idx_mask"] + masks["vonmise_idx_mask"]) # + sky_idx_mask)                      
    masks["dist_mask"], masks["dist_idx_mask"], masks["dis_len"] = get_param_index(params['inf_pars'],['luminosity_distance'])
    masks["not_dist_mask"], masks["not_dist_idx_mask"], masks["not_dist_len"] = get_param_index(params['inf_pars'],['mass_1','mass_2','psi','phase','geocent_time','theta_jn','ra','dec','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl'])

    masks["phase_mask"], masks["phase_idx_mask"], masks["phase_len"] = get_param_index(params['inf_pars'],['phase'])
    masks["not_phase_mask"], masks["not_phase_idx_mask"], masks["not_phase_len"] = get_param_index(params['inf_pars'],['mass_1','mass_2','luminosity_distance','psi','geocent_time','theta_jn','ra','dec','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl'])

    masks["geocent_mask"], masks["geocent_idx_mask"], masks["geocent_len"] = get_param_index(params['inf_pars'],['geocent_time'])
    masks["not_geocent_mask"], masks["not_geocent_idx_mask"], masks["not_geocent_len"] = get_param_index(params['inf_pars'],['mass_1','mass_2','luminosity_distance','psi','phase','theta_jn','ra','dec','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl'])

    masks["xyz_mask"], masks["xyz_idx_mask"], masks["xyz_len"] = get_param_index(params['inf_pars'],['luminosity_distance','ra','dec'])
    masks["not_xyz_mask"], masks["not_xyz_idx_mask"], masks["not_xyz_len"] = get_param_index(params['inf_pars'],['mass_1','mass_2','psi','phase','geocent_time','theta_jn','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl'])

    masks["periodic_mask"], masks["periodic_idx_mask"], masks["periodic_len"] = get_param_index(params['inf_pars'],['phase','psi','phi_12','phi_jl'])
    masks["nonperiodic_mask"], masks["nonperiodic_idx_mask"], masks["nonperiodic_len"] = get_param_index(params['inf_pars'],['mass_1','mass_2','luminosity_distance','geocent_time','theta_jn','a_1','a_2','tilt_1','tilt_2'])
    masks["nonperiodic_nonm1m2_mask"], masks["nonperiodic_nonm1m2_idx_mask"], masks["nonperiodic_nonm1m2_len"] = get_param_index(params['inf_pars'],['luminosity_distance','geocent_time','theta_jn','a_1','a_2','tilt_1','tilt_2'])
    masks["nonperiodic_m1m2_mask"], masks["nonperiodic_m1m2_idx_mask"], masks["nonperiodic_m1m2_len"] = get_param_index(params['inf_pars'],['mass_1','mass_2'])

    masks["idx_xyz_mask"] = np.argsort(masks["xyz_idx_mask"] + masks["not_xyz_idx_mask"])
    masks["idx_dist_mask"] = np.argsort(masks["not_dist_idx_mask"] + masks["dist_idx_mask"])
    masks["idx_phase_mask"] = np.argsort(masks["not_phase_idx_mask"] + masks["phase_idx_mask"])
    masks["idx_geocent_mask"] = np.argsort(masks["not_geocent_idx_mask"] + masks["geocent_idx_mask"])
    masks["idx_periodic_mask"] = np.argsort(masks["nonperiodic_idx_mask"] + masks["periodic_idx_mask"] + masks["ra_idx_mask"] + masks["dec_idx_mask"])
    if print_masks:
        print(masks["xyz_mask"])
        print(masks["not_xyz_mask"])
        print(masks["idx_xyz_mask"])
        #masses_len = m1_len + m2_len                                                                                                                                            
        print(params['inf_pars'])
        print(masks["vonmise_mask"],masks["vonmise_idx_mask"])
        print(masks["gauss_mask"],masks["gauss_idx_mask"])
        print(masks["m1_mask"],masks["m1_idx_mask"])
        print(masks["m2_mask"],masks["m2_idx_mask"])
        print(masks["sky_mask"],masks["sky_idx_mask"])
        print(masks["idx_mask"])
        
    return params, bounds, masks, fixed_vals



def ramp_func(epoch,start,ramp_length, n_cycles):
    i = (epoch-start)/(2.0*ramp_length)
    print("ramp",epoch,i)
    if i<0:
        return 0.0
    if i>=n_cycles:
        return 1.0

    return min(1.0,2.0*np.remainder(i,1.0))

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
            KL_est = plot_posterior(samples,x_batch_test[0,:],epoch,step,all_other_samples=bilby_samples[:,step,:],run=plot_dir, params=params, bounds=bounds, masks=masks)
            _ = plot_posterior(samples,x_batch_test[0,:],epoch,step,run=plot_dir, params=params, bounds=bounds, masks=masks)
    print('... Finished making publication plots! Congrats fam.')

    # Make p-p plots
    plotter.plot_pp(model, y_data_test, x_data_test, params, bounds, inf_ol_idx, bilby_ol_idx)
    print('... Finished making p-p plots!')

    # Make KL plots
    plotter.gen_kl_plots(model,y_data_test, x_data_test, params, bounds, inf_ol_idx, bilby_ol_idx)
    print('... Finished making KL plots!')    

    return


def run_vitc(params, x_data_train, y_data_train, x_data_val, y_data_val, x_data_test, y_data_test, y_data_test_noisefree, save_dir, truth_test, bounds, fixed_vals, bilby_samples, snrs_test=None, params_dir = "./params_files"):

    #params, bounds, masks, fixed_vals = get_params(params_dir = params_dir)
    params, bounds, masks, fixed_vals = get_params(params, bounds, fixed_vals, params_dir = params_dir)
    run = time.strftime('%y-%m-%d-%X-%Z')

    # define which gpu to use during training
    gpu_num = str(params['gpu_num'])   
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num
    #print('... running on GPU {}'.format(gpu_num))
    print("CUDA DEV: ",os.environ["CUDA_VISIBLE_DEVICES"])
    # Let GPU consumption grow as needed
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    print('... letting GPU consumption grow as needed')
    
    train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_log_dir = params['plot_dir'] + '/logs'

    epochs = params['num_iterations']
    train_size = params['load_chunk_size']
    batch_size = params['batch_size']
    val_size = params['val_dataset_size']
    test_size = params['r']
    plot_dir = params['plot_dir']
    plot_cadence = int(0.5*params['plot_interval'])
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = os.path.join(plot_dir,"checkpoint","model.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    comp_post_dir = os.path.join(plot_dir, "comp_posterior")
    full_post_dir = os.path.join(plot_dir, "full_posterior")
    post_samp_dir = os.path.join(plot_dir, "posterior_samples")
    latent_dir = os.path.join(plot_dir, "latent_plot")
    dirs = [checkpoint_dir, comp_post_dir, full_post_dir, post_samp_dir, latent_dir]
    for direc in dirs:
        if not os.path.isdir(direc):
            os.makedirs(direc)

    make_paper_plots = params['make_paper_plots']
    hyper_par_tune = False

    # if doing hour angle, use hour angle bounds on RA
    bounds['ra_min'] = convert_ra_to_hour_angle(bounds['ra_min'],params,None,single=True)
    bounds['ra_max'] = convert_ra_to_hour_angle(bounds['ra_max'],params,None,single=True)
    print('... converted RA bounds to hour angle')

    # load the training data
    if not make_paper_plots:
        train_dataset = DataLoader(params["train_set_dir"],params = params,bounds = bounds, masks = masks,fixed_vals = fixed_vals, chunk_batch = 40, num_epoch_load = 10) 
        validation_dataset = DataLoader(params["val_set_dir"],params = params,bounds = bounds, masks = masks,fixed_vals = fixed_vals, chunk_batch = 2, val_set = True)

    test_dataset = DataLoader(params["test_set_dir"],params = params,bounds = bounds, masks = masks,fixed_vals = fixed_vals, chunk_batch = test_size, test_set = True)

    print("Loading intitial data...")
    train_dataset.load_next_chunk()
    validation_dataset.load_next_chunk()
    test_dataset.load_next_chunk()

    # load precomputed samples
    bilby_samples = []
    for sampler in params['samplers'][1:]:
        bilby_samples.append(load_samples(params,sampler, bounds = bounds))
    bilby_samples = np.array(bilby_samples)
    #bilby_samples = np.array([load_samples(params,'dynesty'),load_samples(params,'ptemcee'),load_samples(params,'cpnest')])

    train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    start_epoch = 0
    if params['resume_training']:
        model = CVAE(params, bounds, masks)
        # Load the previously saved weights
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest)
        print('... loading in previous model %s' % checkpoint_path)
        with open(os.path.join(params['plot_dir'], "loss.txt"),"r") as f:
            start_epoch = len(np.loadtxt(f))
    else:
        model = CVAE(params, bounds, masks)


    # Make publication plots
    if make_paper_plots:
        print('... Making plots for publication.')
        # Load the previously saved weights
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest)
        print('... loading in previous model %s' % checkpoint_path)
        paper_plots(test_dataset, y_data_test, x_data_test, model, params, plot_dir, run, bilby_samples)
        return

    # start the training loop
    train_loss = np.zeros((epochs,3))
    val_loss = np.zeros((epochs,3))
    ramp_start = params['ramp_start']
    ramp_length = params['ramp_end']
    ramp_cycles = 1
    KL_samples = []

    #tf.keras.mixed_precision.set_global_policy('float32')
    #optimizer = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay = 1e-8)
    optimizer = tf.keras.optimizers.Adam(params["initial_training_rate"])
    #optimizer = AdamW(lr=1e-4, model=model,
    #                  use_cosine_annealing=True, total_iterations=40)
    #optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    # Keras hyperparameter optimization
    if hyper_par_tune:
        import keras_hyper_optim
        del model
        keras_hyper_optim.main(train_dataset, val_dataset)
        exit()

    # log params used for this run
    path = params['plot_dir']
    root_dir = params['root_dir']
    shutil.copy(os.path.join(root_dir,'vitamin_c_fit.py'),path)
    if model_fit_type == "basic":
        shutil.copy(os.path.join(root_dir,'vitamin_c_model_fit.py'),path)
    elif model_fit_type == "multidet":
        shutil.copy(os.path.join(root_dir,'vitamin_c_model_fit_multidet.py'),path)
    elif model_fit_type == "multiscale":
        shutil.copy(os.path.join(root_dir,'vitamin_c_model_fit_multiscale.py'),path)
    elif model_fit_type == "kaggle":
        shutil.copy(os.path.join(root_dir,'vitamin_c_model_fit_kaggle1.py'),path)
    elif model_fit_type == "resnet":
        shutil.copy(os.path.join(root_dir,'vitamin_c_model_fit_resnet.py'),path)
    elif model_fit_type == "freq":
        shutil.copy(os.path.join(root_dir,'vitamin_c_model_fit_freq.py'),path)

    shutil.copy(os.path.join(root_dir,'load_data_fit.py'),path)
    shutil.copy(os.path.join(params_dir,'params.json'),path)
    shutil.copy(os.path.join(params_dir,'bounds.json'),path)

    # compile and build the model (hardcoded values will change soon)
    model.compile(run_eagerly = False, optimizer = optimizer, loss = model.compute_loss)
    #model.build((None, 1024,2))
    with open(os.path.join(path, "model_summary.txt"),"w") as f:
        model.encoder_r1.summary(print_fn=lambda x: f.write(x + '\n'))
        model.encoder_q.summary(print_fn=lambda x: f.write(x + '\n'))
        model.decoder_r2.summary(print_fn=lambda x: f.write(x + '\n'))


    callbacks = [PlotCallback(plot_dir, epoch_plot=100,start_epoch=start_epoch), TrainCallback(checkpoint_path, optimizer, plot_dir, train_dataset, model), TestCallback(test_dataset,comp_post_dir,full_post_dir, latent_dir, bilby_samples, test_epoch = 1000), TimeCallback(save_dir=plot_dir, save_interval = 100)]

    model.fit(train_dataset, use_multiprocessing = False, workers = 6,epochs = 30000, callbacks = callbacks, shuffle = False, validation_data = validation_dataset, max_queue_size = 100, initial_epoch = start_epoch)

    # not happy with this re-wrapping of the dataset
    #data_gen_wrap = tf.data.Dataset.from_generator(lambda : train_dataset,(tf.float32,tf.float32))
    #valdata_gen_wrap = tf.data.Dataset.from_generator(lambda : validation_dataset,(tf.float32,tf.float32))

    #model.fit_generator(data_gen_wrap, use_multiprocessing = False, workers = 6,epochs = 10000, callbacks = callbacks, shuffle = False, validation_data = valdata_gen_wrap, max_queue_size = 100)


