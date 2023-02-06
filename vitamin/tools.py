import bilby
import pandas
import numpy as np
import matplotlib.pyplot as plt
import corner
import os
from matplotlib.colors import LinearSegmentedColormap

def latent_corner_plot(zr_sample, zq_sample):
    """
    plot latent space samples as histogram
    """
    # set shape of latent space
    zrshape = np.shape(zr_sample)
    num_latent = zrshape[-1]
    randint = np.random.randint(0,len(zr_sample),size = 1000)
    cfig = corner.corner(zq_sample[randint].numpy(), labels = ["z_{}".format(i) for i in range(num_latent)], color="C0")
    corner.corner(zr_sample[randint], fig = cfig, color="C1")

    return cfig

def latent_samp_fig(zr_sample, zq_sample, truths):

    print(zr_sample.shape, zq_sample.shape)
    fig = corner.corner(zr_sample, truths = truths)
    corner.corner(zq_sample, fig = fig)

    return fig

def latent_samp_fig_old(zr_sample, zq_sample, truths):
    """
    plot latent space samples as histogram
    """
    # set shape of latent space
    zrshape = np.shape(zr_sample)
    if len(zrshape) == 2:
        num_latent = 1
        zr_sample = zr_sample.reshape(zrshape[0],zr_shape[1],1)
        zq_sample = zq_sample.reshape(zrshape[0],zr_shape[1],1)
    else:
        num_latent = zrshape[-1]
    
    fig, ax = plt.subplots(figsize=(8*num_latent,8), ncols=num_latent,nrows=2)
    rint = np.random.randint(0,zrshape[0],size=50)
    for inj in rint:
        color = "C0" if truths[inj][0] == 0 else "C1"
        if num_latent == 1:
            # get 1000 random samples from latent space and histogram
            randint = np.random.randint(0,len(zr_sample[inj]),size = 1000)
            hstr = ax[0].hist(zr_sample[inj][randint,0],bins=30,color=color,histtype = "step")
            hstq = ax[1].hist(zq_sample[inj][randint,0],bins=30,color=color,histtype="step")
        else:
            for lat in range(num_latent):
                randint = np.random.randint(0,len(zr_sample[inj]),size = 1000)
                hstr = ax[0,lat].hist(zr_sample[inj][randint,lat],bins=30,color=color,histtype = "step")
                hstq = ax[1,lat].hist(zq_sample[inj][randint,lat],bins=30,color=color,histtype="step")
    if num_latent == 1:
        ax[0].set_ylabel("R encoder latent")
        ax[1].set_ylabel("Q encoder latent")
    else:
        ax[0,0].set_ylabel("R encoder latent")
        ax[1,0].set_ylabel("Q encoder latent")

    return fig

def make_ppplot(samples, truths, savepath, labels):

    results = []
    for sp in range(len(samples)):
        res = bilby.result.Result()
        post = pandas.DataFrame(data = samples[sp], columns = labels)
        res.posterior = post
        res.search_parameter_keys = labels
        res.injection_parameters = {labels[i]:truths[sp][i] for i in range(len(labels))}
        res.priors = {labels[i]:bilby.prior.Gaussian(0,1, name=labels[i]) for i in range(len(labels))}
        results.append(res)

    fig, pv = bilby.result.make_pp_plot(results, filename = savepath)
    
def loss_plot(save_dir, loss, kl_loss, l_loss, val_loss, val_kl_loss, val_l_loss):
    """ Plot the loss curves"""
    fig, ax = plt.subplots(nrows = 2,figsize=(15,15))
    stind = int(0.1*len(loss))
    xs = np.linspace(stind,len(loss), len(loss) - stind)
    ax[0].plot(xs,loss[stind:],label="loss",color="C0",alpha=0.5)
    ax[1].plot(xs, np.array(kl_loss[stind:]),label="KL loss",color="C1",alpha=0.5)
    ax[0].plot(xs,-np.array(l_loss[stind:]),label="L loss",color="C2",alpha=0.5)
    
    ax[0].plot(xs,val_loss[stind:],label="val loss", color="C0", ls="--", lw=3, alpha=0.5)
    ax[1].plot(xs,np.array(val_kl_loss[stind:]), label="val KL loss", color="C1", ls="--", lw=3, alpha=0.5)
    ax[0].plot(xs,-np.array(val_l_loss[stind:]), label="val L loss", color="C2", ls="--", lw=3, alpha=0.5)

    ax[1].set_xlabel("Epoch")
    ax[0].legend(fontsize=20)
    ax[1].legend(fontsize=20)
    ax[0].grid()
    ax[1].grid()
    ax[1].set_yscale("log")
    ax[0].set_yscale("symlog")
    #ax.set_yscale("symlog")
    fig.savefig(os.path.join(save_dir, "loss.png"))

    ax[0].set_xscale("log")
    ax[1].set_xscale("log")
    fig.savefig(os.path.join(save_dir, "log_loss.png"))
    plt.close(fig)

    fig, ax = plt.subplots(nrows = 1,figsize=(15,15))
    stind = int(0.1*len(loss))
    xs = np.linspace(stind,len(loss), len(loss) - stind)
    ax.plot(xs,loss[stind:],label="loss",color="C0",alpha=0.5)
    ax.plot(xs,val_loss[stind:],label="val loss", color="C0", ls="--", lw=3, alpha=0.5)

    ax.set_xlabel("Epoch")
    ax.legend(fontsize=20)
    ax.grid()
    ax.set_yscale("symlog")
    #ax.set_yscale("symlog")
    fig.savefig(os.path.join(save_dir, "loss_TOTAL.png"))
    plt.close(fig)
    
    fig, ax = plt.subplots(nrows = 1,figsize=(15,15))
    stind = int(0.1*len(loss))
    xs = np.linspace(stind,len(loss), len(loss) - stind)
    ax.plot(xs,kl_loss[stind:],label="loss",color="C0",alpha=0.5)
    ax.plot(xs,val_kl_loss[stind:],label="val loss", color="C0", ls="--", lw=3, alpha=0.5)

    ax.set_xlabel("Epoch")
    ax.legend(fontsize=20)
    ax.grid()
    #ax.set_yscale("symlog")
    fig.savefig(os.path.join(save_dir, "loss_KL.png"))
    plt.close(fig)

    fig, ax = plt.subplots(nrows = 1,figsize=(15,15))
    stind = int(0.1*len(loss))
    xs = np.linspace(stind,len(loss), len(loss) - stind)
    ax.plot(xs,-np.array(l_loss[stind:]),label="loss",color="C0",alpha=0.5)
    ax.plot(xs,-np.array(val_l_loss[stind:]),label="val loss", color="C0", ls="--", lw=3, alpha=0.5)

    ax.set_xlabel("Epoch")
    ax.legend(fontsize=20)
    ax.grid()
    ax.set_yscale("symlog")
    #ax.set_yscale("symlog")
    fig.savefig(os.path.join(save_dir, "loss_RECON.png"))
    plt.close(fig)

    
def loss_plot_dec(save_dir, loss, val_loss):
    """ Plot the loss curves"""
    fig, ax = plt.subplots(nrows = 1,figsize=(15,15))
    stind = int(0.05*len(loss))
    xs = np.linspace(stind,len(loss), len(loss) - stind)
    ax.plot(xs,loss[stind:],label="loss",color="C0",alpha=0.5)
    ax.plot(xs,val_loss[stind:],label="val loss", color="C0", ls="--", lw=3, alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=20)
    ax.grid()
    ax.set_yscale("log")
    #ax.set_yscale("symlog")
    fig.savefig(os.path.join(save_dir, "loss_dec.png"))

    ax.set_xscale("log")
    ax.set_xscale("log")
    fig.savefig(os.path.join(save_dir, "log_loss_dec.png"))
    plt.close(fig)
    
    

    
