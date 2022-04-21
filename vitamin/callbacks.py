import tensorflow as tf
from tensorflow.keras import regularizers
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import time 
import matplotlib.pyplot as plt
import os
import sys
from . import plotting 
from .train_plots import plot_losses, plot_losses_zoom, plot_latent, plot_posterior


class PlotCallback(tf.keras.callbacks.Callback):

    def __init__(self, plot_dir, epoch_plot = 2, start_epoch = 0):

        self.plot_dir = plot_dir
        self.epoch_plot = epoch_plot
        self.train_losses = [[],[],[]]
        self.val_losses = [[],[],[]]
        if start_epoch != 0:
            with open(os.path.join(plot_dir, "loss.txt")) as f:
                losses = np.loadtxt(f)
            self.train_losses = [list(losses[:,0]),list(losses[:,1]),list(losses[:,2])]
            self.val_losses = [list(losses[:,3]),list(losses[:,4]),list(losses[:,5])]

    def on_epoch_end(self, epoch, logs = None):
        self.train_losses[2].append(logs["total_loss"])
        self.train_losses[0].append(logs["recon_loss"])
        self.train_losses[1].append(logs["kl_loss"])

        self.val_losses[2].append(logs["val_total_loss"])
        self.val_losses[0].append(logs["val_recon_loss"])
        self.val_losses[1].append(logs["val_kl_loss"])

        if epoch % self.epoch_plot == 0 and epoch > 3:
            ind_start = epoch - 1000 if epoch > 1000 else 0
            plot_losses(np.array(self.train_losses).T, np.array(self.val_losses).T, epoch, run = self.plot_dir)
            plot_losses_zoom(np.array(self.train_losses).T, np.array(self.val_losses).T, epoch, run = self.plot_dir, ind_start=ind_start, label="TOTAL")
            plot_losses_zoom(np.array(self.train_losses).T, np.array(self.val_losses).T, epoch, run = self.plot_dir, ind_start=ind_start, label="RECON")
            plot_losses_zoom(np.array(self.train_losses).T, np.array(self.val_losses).T, epoch, run = self.plot_dir, ind_start=ind_start, label="KL")


class TrainCallback(tf.keras.callbacks.Callback):

    def __init__(self, config,  optimizer, train_dataloader, model):
        super(TrainCallback, self).__init__()
        self.config = config
        self.model = model
        self.checkpoint_path = os.path.join(self.config["output"]["output_directory"], "checkpoint", "model.ckpt")
        self.plot_dir = self.config["output"]["output_directory"]
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.recon_losses = []
        self.kl_losses = []
        self.gauss_losses = []
        self.vm_losses = []
        self.old_recon_losses = []
        self.old_kl_losses = []
        self.old_gauss_losses = []
        self.old_vm_losses = []


    def inf_data_plot(self):
        """
        Plot the bad piece of data - not yet used
        """
        
        data = self.train_dataloader.Y_noisefree[bad_index]

        fig, ax = plt.subplots(nrows = 2)
        ax[0].plot(tf.layers.activations.tanh(data[:,0]))
        ax[0].plot(tf.layers.activations.tanh(data[:,1]))
        ax[0].plot(data[:,0])
        ax[0].plot(data[:,1])

        fig.savefig(os.path.join(self.config["output"]["output_directory"], "bad_data_plot.png"))
        plt.close(fig)
        sys.exit()

    def nan_plots(self):

        total_r_losses = np.append(self.old_recon_losses, self.recon_losses)
        total_kl_losses = np.append(self.old_kl_losses, self.kl_losses)
        total_gauss_losses = np.append(self.old_gauss_losses, self.gauss_losses)
        total_vm_losses = np.append(self.old_vm_losses, self.vm_losses)
        
        fig, ax = plt.subplots(nrows = 2, figsize = (16,10))
        ax[0].plot(total_r_losses, label = "recon")
        ax[0].plot(total_kl_losses, label = "kl")
        ax[0].set_yscale("symlog")
        ax[0].legend()
        ax[1].plot(total_gauss_losses, label = "gauss")
        ax[1].plot(total_vm_losses, label = "von mises")
        ax[1].set_yscale("symlog")
        ax[1].legend()
        
        fig.savefig(os.path.join(self.config["output"]["output_directory"], "batch_inf_plot.png"))
        plt.close(fig)
        sys.exit()
        
    def ramp_func(self,epoch):
        ramp = (epoch-self.config["training"]["ramp_start"])/(2.0*self.config["training"]["ramp_length"])
        #print(epoch,ramp)
        if ramp<0:
            ramp = 0.0
        elif ramp>=self.config["training"]["ramp_n_cycles"]:
            ramp = 1.0
        ramp = min(1.0,2.0*np.remainder(ramp,1.0))
        if epoch > self.config["training"]["ramp_start"] + self.config["training"]["ramp_length"]:
            ramp = 1.0
        tf.keras.backend.set_value(self.model.ramp, ramp)

    def learning_rate_modify(self, epoch, epochs_range = 100, decay_length = 4000, init_rate = 1e-4):
        dec_factor = 1
        cycle_factor = 1
        if epoch > self.config["training"]["cycle_lr_start"]:
            half_fact_arr = np.linspace(1/10, 10, int(self.config["training"]["cycle_lr_length"]/2))
            fact_arr = np.append(half_fact_arr, half_fact_arr[::-1])
            position = np.remainder(epoch - self.config["training"]["cycle_lr_start"], self.config["training"]["cycle_lr_length"]).astype(int)
            cycle_factor = fact_arr[position]

        if epoch > self.config["training"]["decay_lr_start"]:
            dec_array = np.logspace(-2, 0, self.config["training"]["decay_lr_length"])[::-1]
            decay_pos = epoch - self.config["training"]["decay_lr_start"]
            if decay_pos > self.config["training"]["decay_lr_length"]:
                decay_pos = -1
            dec_factor = dec_array[decay_pos]

        new_lr = cycle_factor*dec_factor*self.config["training"]["initial_learning_rate"]
            
        tf.keras.backend.set_value(self.optimizer.learning_rate, new_lr)
        print("learning_rate:, {}".format(self.optimizer.learning_rate))


    def learning_rate_it(self, epoch, epochs_range = 30):

        lrrange = np.linspace(-9, -1, 200)
        lr = 10**lrrange
        factor = lr[epoch]
        tf.keras.backend.set_value(self.optimizer.learning_rate, factor)
        print("learning_rate:, {}".format(self.optimizer.learning_rate))

    def on_epoch_begin(self, epoch, logs = None):
        self.learning_rate_modify(epoch)
        #self.learning_rate_it(epoch)
        if epoch > self.config["training"]["ramp_start"]:
            self.ramp_func(epoch)


    def on_batch_end(self, batch, logs=None):
        self.recon_losses.append(self.model.recon_loss_metric.result())
        self.kl_losses.append(self.model.kl_loss_metric.result())
        self.gauss_losses.append(self.model.gauss_loss_metric.result())
        self.vm_losses.append(self.model.vm_loss_metric.result())

        if not np.isfinite(self.recon_losses[-1]):
            print("recon loss inf \n")
            if not np.isfinite(self.gauss_losses[-1]):
                print("gauss inf \n")
            self.nan_plots()
        if not np.isfinite(self.kl_losses[-1]):
            print("kl loss inf")
            self.nan_plots()

        #print("mr",self.model.ramp)
        
    def on_epoch_end(self,epoch, logs = None):
        self.old_recon_losses = self.recon_losses
        self.old_kl_losses = self.kl_losses
        self.old_gauss_losses = self.gauss_losses
        self.old_vm_losses = self.vm_losses

        self.recon_losses = []
        self.kl_losses = []
        self.gauss_losses = []
        self.vm_losses = []

        #if epoch % self.config["training"]['plot_interval'] == 0:
            # Save the weights using the `checkpoint_path` format
        #    self.model.save_weights(self.checkpoint_path)
        #    print('... Saved model %s ' % self.checkpoint_path)

class TestCallback(tf.keras.callbacks.Callback):


    def __init__(self, config, test_dataset, bilby_samples,test_epoch = 500):
        self.config = config
        self.test_dataset = test_dataset
        self.comp_post_dir = os.path.join(config["output"]["output_directory"],"comparison_posteriors")
        self.full_post_dir = os.path.join(config["output"]["output_directory"],"full_posteriors")
        self.latent_dir = os.path.join(config["output"]["output_directory"],"latent_plots")
        self.bilby_samples = bilby_samples
        self.paper_plots = config["testing"]["make_paper_plots"]
        self.paper_plot_dir = os.path.join(self.config["output"]["output_directory"],"paper_plots")
        for direc in [self.comp_post_dir, self.full_post_dir, self.latent_dir, self.paper_plot_dir]:
            if not os.path.isdir(direc):
                os.makedirs(direc)

        self.test_epoch = test_epoch

    def on_epoch_end(self, epoch, logs = None):
        
        if epoch % self.test_epoch == 0 and epoch != 0:
            for step in range(self.config["data"]["n_test_data"]):
                mu_r1, z_r1, mu_q, z_q, scale_r1, scale_q, logvar_q = self.model.gen_z_samples(tf.expand_dims(self.test_dataset.X[step],0), tf.expand_dims(self.test_dataset.Y_noisy[step],0), nsamples=1000)

                if np.any(np.isinf(np.exp(logvar_q))):
                    print("maxminlogvar", tf.reduce_min(logvar_q), tf.reduce_max(logvar_q))
                if np.any(np.isnan(np.exp(logvar_q))):
                    print("maxminlogvar", tf.reduce_min(logvar_q), tf.reduce_max(logvar_q))

                plot_latent(mu_r1,z_r1,mu_q,z_q,epoch,step,run=self.latent_dir)
                start_time_test = time.time()
                samples = self.model.gen_samples(tf.expand_dims(self.test_dataset.Y_noisy[step],0), nsamples=self.config["testing"]['n_samples'])

                end_time_test = time.time()
                if np.any(np.isnan(samples)):
                    print('Epoch: {}, found nans in samples. Not making plots'.format(epoch))
                    KL_est = [-1,-1,-1]
                else:
                    print('Epoch: {}, Testing time elapsed for {} samples: {}'.format(epoch,self.config["testing"]['n_samples'],end_time_test - start_time_test))
                    if len(np.shape(self.bilby_samples)) == 4:
                        KL_est = plot_posterior(samples,self.test_dataset.truths[step],epoch,step,all_other_samples=self.bilby_samples[:,step,:], config=self.config, unconvert_parameters = self.test_dataset.unconvert_parameters)
                    else:
                        print("not plotting posterior, bilby samples wrong shape")
                        #KL_est = plot_posterior(samples,self.test_dataset.truths[step],epoch,step,all_other_samples=None, config=self.config, unconvert_parameters = self.test_dataset.unconvert_parameters)
            if self.paper_plots:
                # This needs to be checked and rewritten
                epoch = 'pub_plot'; ramp = 1
                plotter = plotting.make_plots(config, None, None, self.test_dataset.X) 
                # Make p-p plots
                plotter.plot_pp(self.model, self.test_dataset.Y_noisy, self.test_dataset.X, config, config["masks"]["inf_ol_idx"], config["masks"]["bilby_ol_idx"], self.bilby_samples)
                print('... Finished making p-p plots!')

                # Make KL plots
                #plotter.gen_kl_plots(self.model,self.test_dataset.Y_noisy, self.test_dataset.X, self.model.params, self.model.bounds, self.model.masks["inf_ol_idx"], self.model.masks["bilby_ol_idx"], self.bilby_samples)
                #print('... Finished making KL plots!')



class TimeCallback(tf.keras.callbacks.Callback):
    def __init__(self, config, start_epoch = 0):
        self.config = config
        self.save_interval = self.config["training"]["plot_interval"]
        self.fname = os.path.join(self.config["output"]["output_directory"], "epochs_times.txt")
        self.start_epoch = start_epoch
        if start_epoch != 0:
            with open(self.fname, "r") as f:
                self.times = np.loadtxt(f)
            self.total_elapsed = self.times[-1] - self.times[0]
        else:
            self.times = []
            self.total_elapsed = 0

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        temp_time = time.time() - self.epoch_time_start + self.total_elapsed
        self.times.append(temp_time)

        if batch % self.config["training"]["plot_interval"] == 0:
            with open(self.fname, "w") as f:
                np.savetxt(f, self.times)

