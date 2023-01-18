import tensorflow as tf
from tensorflow.keras import regularizers
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.keras import backend as K
import numpy as np
import time 
import matplotlib.pyplot as plt
import os
import sys
import pickle
from . import plotting 
from .train_plots import plot_losses, plot_losses_zoom, plot_latent, plot_posterior, plot_JS_div


class PlotCallback(tf.keras.callbacks.Callback):

    def __init__(self, plot_dir, epoch_plot = 2, start_epoch = 0, save_data = True):

        self.plot_dir = plot_dir
        self.epoch_plot = epoch_plot
        self.save_data = save_data
        self.all_losses = None#np.array([])#np.array([[],[],[],[],[],[],[]]).T

        if start_epoch != 0:
            with open(os.path.join(plot_dir, "loss.txt")) as f:
                losses = np.loadtxt(f)
            self.all_losses = np.array(losses)#[list(losses[:,0]),list(losses[:,1]),list(losses[:,2]),list(losses[:,3]),list(losses[:,4]),list(losses[:,5]),list(losses[:,6])])
            #if len(np.shape(self.all_losses)) == 1:
            #    self.all_losses = np.reshape(self.all_losses, (7,-1)).T
            #self.train_losses = [list(losses[:,0]),list(losses[:,1]),list(losses[:,2]),list(losses[:,3])]
            #self.val_losses = [list(losses[:,4]),list(losses[:,5]),list(losses[:,6])]

    def on_epoch_end(self, epoch, logs = None):
        """
        self.all_losses[3].append(logs["total_loss"])
        self.all_losses[1].append(logs["recon_loss"])
        self.all_losses[2].append(logs["kl_loss"])

        self.all_losses[6].append(logs["val_total_loss"])
        self.all_losses[4].append(logs["val_recon_loss"])
        self.all_losses[5].append(logs["val_kl_loss"])

        self.all_losses[0].append(time.time())
        """
        if self.all_losses is None:
            self.all_losses = np.array([[time.time(), logs["recon_loss"], logs["kl_loss"], logs["total_loss"], logs["val_recon_loss"], logs["val_kl_loss"], logs["val_total_loss"]]])
        else:
            self.all_losses = np.append(self.all_losses, [[time.time(), logs["recon_loss"], logs["kl_loss"], logs["total_loss"], logs["val_recon_loss"], logs["val_kl_loss"], logs["val_total_loss"]]], axis = 0)
        #print("train_loss_shape", np.shape(self.all_losses))
        
        if epoch % self.epoch_plot == 0 and epoch != 0 or epoch == 2:
            if self.save_data:
                ind_start = epoch - 1000 if epoch > 1000 else 0
                with open(os.path.join(self.plot_dir, "loss.txt"), "w") as f:
                    np.savetxt(f,np.array(self.all_losses))

                plot_losses(np.array(self.all_losses), epoch, run = self.plot_dir)
            
                plot_losses_zoom(np.array(self.all_losses), epoch, run = self.plot_dir, ind_start=ind_start, label="TOTAL")
                plot_losses_zoom(np.array(self.all_losses), epoch, run = self.plot_dir, ind_start=ind_start, label="RECON")
                plot_losses_zoom(np.array(self.all_losses), epoch, run = self.plot_dir, ind_start=ind_start, label="KL")


class AnnealCallback(tf.keras.callbacks.Callback):

    def __init__(self, ramp_start, ramp_length, ramp_n_cycles=1):
        self.ramp_start = ramp_start
        self.ramp_length = ramp_length
        self.ramp_n_cycles = ramp_n_cycles
        #self.model = model

    def ramp_func(self,epoch):
        ramp = (epoch-self.ramp_start)/(2.0*self.ramp_length)
        #print(epoch,ramp)
        if ramp<0:
            ramp = 0.0
        elif ramp>=self.ramp_n_cycles:
            ramp = 1.0
        ramp = min(1.0,2.0*np.remainder(ramp,1.0))
        if epoch > self.ramp_start + self.ramp_length:
            ramp = 1.0
        return ramp

    def on_epoch_end(self,epoch,logs=None):
        ramp = self.ramp_func(epoch)
        tf.keras.backend.set_value(self.model.ramp, ramp)

class LogminRampCallback(tf.keras.callbacks.Callback):

    def __init__(self, logvarmin_ramp_start, logvarmin_ramp_length, logvarmin_start, logvarmin_end, model):
        self.logvarmin_ramp_start = logvarmin_ramp_start
        self.logvarmin_ramp_length = logvarmin_ramp_length
        self.logvarmin_start = logvarmin_start
        self.logvarmin_end = logvarmin_end
        self.model = model

    def logmin_ramp_func(self,epoch):
        ramp = (epoch-self.logvarmin_ramp_start)/(self.logvarmin_ramp_length)
        if ramp<0:
            ramp = 0.0
        if epoch > self.logvarmin_ramp_start + self.logvarmin_ramp_length:
            ramp = 1.0

        newlogvar = self.logvarmin_start + ramp*self.logvarmin_end
        tf.keras.backend.set_value(self.model.minlogvar, newlogvar)

    def on_epoch_begin(self, epoch, logs=None):
        self.logmin_ramp_func(epoch)

class LearningRateCallback(tf.keras.callbacks.Callback):

    def __init__(self, initial_learning_rate, cycle_lr = False, cycle_lr_start = 1000, cycle_lr_length=100, cycle_lr_amp=5, decay_lr=False, decay_lr_start=1000, decay_lr_length=5000, decay_lr_logend = -3, optimizer = None):
        self.cycle_lr_start = cycle_lr_start
        self.cycle_lr = cycle_lr
        self.cycle_lr_amp = cycle_lr_amp
        self.cycle_lr_length = cycle_lr_length
        self.decay_lr_start = decay_lr_start
        self.decay_lr = decay_lr
        self.decay_lr_logend = decay_lr_logend
        self.decay_lr_length = decay_lr_length
        self.initial_learning_rate = initial_learning_rate
        self.optimizer = optimizer

    def learning_rate_modify(self, epoch):
        dec_factor = 1
        cycle_factor = 1
        if epoch > self.cycle_lr_start and self.cycle_lr:
            half_fact_arr = np.linspace(1/self.cycle_lr_amp, self.cycle_lr_amp, int(self.cycle_lr_length/2))
            fact_arr = np.append(half_fact_arr, half_fact_arr[::-1])
            position = np.remainder(epoch - self.cycle_lr_start, self.cycle_lr_length).astype(int)
            cycle_factor = fact_arr[position]

        if epoch > self.decay_lr_start and self.decay_lr:
            dec_array = np.logspace(self.decay_lr_logend, 0, self.decay_lr_length)[::-1]
            decay_pos = epoch - self.decay_lr_start
            if decay_pos >= self.decay_lr_length:
                decay_pos = -1
            dec_factor = dec_array[decay_pos]

        new_lr = cycle_factor*dec_factor*self.initial_learning_rate
            
        tf.keras.backend.set_value(self.optimizer.learning_rate, new_lr)
        print("learning_rate:, {}".format(self.optimizer.learning_rate))

    def on_epoch_begin(self, epoch, logs = None):
        self.learning_rate_modify(epoch)

class BatchRampCallback(tf.keras.callbacks.Callback):

    def __init__(self, batch_ramp_start, batch_ramp_length, batch_size, batch_size_end):
        self.batch_ramp_start = batch_ramp_start
        self.batch_ramp_length = batch_ramp_length
        self.batch_size = batch_size
        self.batch_size_end = batch_size_end

    def batch_ramp_func(self,epoch):
        ramp = (epoch-self.batch_ramp_start)/(self.batch_ramp_length)
        if ramp<0:
            ramp = 0.0
        if epoch > self.batch_ramp_start + self.batch_ramp_length:
            ramp = 1.0

        newbatch = int(self.batch_size + ramp*(self.batch_size_end-self.batch_size_end))
        self.train_dataloader.batch_size =  newbatch

    def on_epoch_begin(self, epoch, logs = None):
        self.batch_ramp_func(epoch)

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
        for name, group in self.model.grouped_params.items():
            setattr(self, "{}_losses".format(name), [])
            setattr(self, "old_{}_losses".format(name), [])

        self.old_recon_losses = []
        self.old_kl_losses = []


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
        all_losses = []
        
        fig, ax = plt.subplots(nrows = 2, figsize = (16,10))
        ax[0].plot(total_r_losses, label = "recon")
        ax[0].plot(total_kl_losses, label = "kl")
        ax[0].set_yscale("symlog")
        ax[0].legend()
        for name, group in self.model.grouped_params.items():
            temp_loss = np.append(getattr(self,"{}_losses".format(name)),getattr(self,"old_{}_losses".format(name)))
            ax[1].plot(temp_loss, label = name)
        ax[1].set_yscale("symlog")
        ax[1].legend()
        
        fig.savefig(os.path.join(self.config["output"]["output_directory"], "batch_inf_plot.png"))
        plt.close(fig)
        sys.exit()
        
    def on_batch_end(self, batch, logs=None):
        self.recon_losses.append(self.model.recon_loss_metric.result())
        self.kl_losses.append(self.model.kl_loss_metric.result())
        #metrics = {}
        #for name, group in self.model.grouped_params.items():
        #    getattr(self, "{}_losses".format(name)).append(getattr(self.model, "{}_loss_metric".format(name)).result())
        #    metrics[name] = getattr(self.model, "{}_loss_metric".format(name)).result()
        if not np.isfinite(self.recon_losses[-1]):
            print("\n recon loss not finite \n")
            print([layer.name for layer in self.model.decoder_r2.layers])
            for name, group in self.model.grouped_params.items():
                print(name, getattr(self, "{}_losses".format(name))[-3:])
                #print(name, "mean", self.model.decoder_r2.get_layer("{}_mean".format(name)))
                #print(name, "logvar", self.model.decoder_r2.get_layer("{}_logvar".format(name)))
            self.nan_plots()
        if not np.isfinite(self.kl_losses[-1]):
            print("kl loss inf")
            self.nan_plots()

        #print("mr",self.model.ramp)
        
    def on_epoch_end(self,epoch, logs = None):
        self.old_recon_losses = self.recon_losses
        self.old_kl_losses = self.kl_losses
        #self.old_gauss_losses = self.gauss_losses
        #self.old_vm_losses = self.vm_losses
        for name, group in self.model.grouped_params.items():
            setattr(self, "old_{}_losses".format(name), getattr(self, "{}_losses".format(name)))
            setattr(self, "{}_losses".format(name), [])

        self.recon_losses = []
        self.kl_losses = []
        #self.gauss_losses = []
        #self.vm_losses = []

        #if epoch % self.config["training"]['plot_interval'] == 0:
            # Save the weights using the `checkpoint_path` format
        #    self.model.save_weights(self.checkpoint_path)
        #    print('... Saved model %s ' % self.checkpoint_path)

class TestCallback(tf.keras.callbacks.Callback):


    def __init__(self, config, test_dataset, bilby_samples):
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


    def on_epoch_end(self, epoch, logs = None):
        
        if epoch % self.config["training"]["test_interval"] == 0 and epoch not in [0,1]:
            for step in range(self.config["data"]["n_test_data"]):
                if step > len(self.test_dataset):
                    break
                if self.config["training"]["plot_latent"]:
                    mu_r1, z_r1, mu_q, z_q, scale_r1, scale_q, logvar_q = self.model.gen_z_samples(tf.expand_dims(self.test_dataset.X[step],0), tf.expand_dims(self.test_dataset.Y_noisy[step],0), nsamples=1000)

                    plot_latent(mu_r1,z_r1,mu_q,z_q,epoch,step,run=self.latent_dir)

                allinds = []
                for samp, sampind in self.test_dataset.samples_available.items():
                    if step not in sampind:
                        allinds.append(step)
                if len(allinds) == len(self.test_dataset.samples_available):
                    print("No available samples: {}".format(step))
                    continue

                start_time_test = time.time()
                samples = self.model.gen_samples(tf.expand_dims(self.test_dataset.Y_noisy[step],0), nsamples=self.config["testing"]['n_samples'], max_samples = 100)

                end_time_test = time.time()
                if np.any(np.isnan(samples)):
                    print('Epoch: {}, found nans in samples. Not making plots'.format(epoch))
                    KL_est = [-1,-1,-1]
                else:
                    print('Epoch: {}, Testing time elapsed for {} samples: {}'.format(epoch,self.config["testing"]['n_samples'],end_time_test - start_time_test))
                    if len(np.shape(self.bilby_samples)) == 4:
                        JS_est, JS_labels = plot_posterior(samples,self.test_dataset.truths[step],epoch,step,all_other_samples=self.bilby_samples[:,step,:], config=self.config, unconvert_parameters = self.test_dataset.unconvert_parameters)
                        plot_JS_div(JS_est[:10], JS_labels)
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

class OptimizerSave(tf.keras.callbacks.Callback):

    def __init__(self, config, checkpoint_path, save_interval = 250):
        self.save_interval = save_interval
        self.checkpoint_path = checkpoint_path

    def on_epoch_end(self,epoch, logs = None):
        
        if epoch % self.save_interval == 0:
            sym_weights = getattr(self.model.optimizer, 'weights')
            weight_values = K.batch_get_value(sym_weights)

            with open(os.path.join(self.checkpoint_path, "optimizer.pkl"),"wb") as f:
                pickle.dump(weight_values, f)









