import torch
import numpy as np
import time 
import matplotlib.pyplot as plt
import os
import sys
import pickle
from . import plotting 
from .train_plots import plot_losses, plot_losses_zoom, plot_latent, plot_posterior, plot_JS_div
from .tools import make_ppplot, loss_plot, latent_corner_plot, latent_samp_fig

class LossPlotCallback():

    def __init__(self, save_dir, checkpoint_dir, save_interval = 50):

        self.save_dir = save_dir
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval

    def on_epoch_end(self, epoch, logs = None):

        if epoch % self.save_interval == 0:
            loss_plot(self.save_dir, logs["train_losses"], logs["kl_losses"], logs["lik_losses"], logs["val_losses"], logs["val_kl_losses"], logs["val_lik_losses"])

            with open(os.path.join(self.checkpoint_dir, "checkpoint_loss.txt"),"w+") as f:
                #if len(logs["train_losses"]) > epoch+1:
                #    epoch = len(logs["train_losses"]) - 1
                savearr =  np.array([np.arange(len(logs["train_losses"])), logs["train_times"], logs["train_losses"], logs["kl_losses"], logs["lik_losses"], logs["val_losses"], logs["val_kl_losses"], logs["val_lik_losses"]]).astype(float)
                np.savetxt(f,savearr)

class AnnealCallback():

    def __init__(self, model, ramp_start, ramp_length, ramp_n_cycles=1):
        self.ramp_start = ramp_start
        self.ramp_length = ramp_length
        self.ramp_end = ramp_start + ramp_length
        self.ramp_n_cycles = ramp_n_cycles
        self.model = model

        self.ramp_array = self.frange_cycle_linear(self.ramp_start, self.ramp_end, self.ramp_n_cycles)

    

    def frange_cycle_linear(self, start, stop, n_cycle=4):
        n_epochs = stop - start + 1
        ann = np.ones(n_epochs)
        
        period = int(n_epochs/n_cycle)
        step = (stop-start)/(period) # linear schedule

        for c in range(n_cycle):

            v , i = start , 0
            while v <= stop and (int(i+c*period) < n_epochs):
                ann[int(i+c*period)] = i/(period)
                v += step
                i += 1
        return ann

    def ramp_func(self,epoch):

        ramp = 0.0
        if epoch>self.ramp_start and epoch<=self.ramp_end:
            #ramp = (np.log(epoch)-np.log(kl_start))/(np.log(kl_end)-np.log(kl_start)) 
            #ramp = (epoch - self.ramp_start)/(self.ramp_end - self.ramp_start)
            print(epoch, self.ramp_start, int(epoch - self.ramp_start))
            ramp = self.ramp_array[int(epoch - self.ramp_start)]
        elif epoch>self.ramp_end:
            ramp = 1.0 
        else:
            ramp = 0.0
        return ramp


    def on_epoch_end(self,epoch,logs=None):
        self.model.ramp = self.ramp_func(epoch)

class LogminRampCallback():

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
        self.model.minlogvar = newlogvar

    def on_epoch_begin(self, epoch, logs=None):
        self.logmin_ramp_func(epoch)

class LearningRateCallback():

    def __init__(self, optimiser, initial_learning_rate, cycle_lr = False, cycle_lr_start = 1000, cycle_lr_length=100, cycle_lr_amp=5, decay_lr=False, decay_lr_start=1000, decay_lr_length=5000, decay_lr_logend = -3, optimizer = None):
        self.cycle_lr_start = cycle_lr_start
        self.cycle_lr = cycle_lr
        self.cycle_lr_amp = cycle_lr_amp
        self.cycle_lr_length = cycle_lr_length
        self.decay_lr_start = decay_lr_start
        self.decay_lr = decay_lr
        self.decay_lr_logend = decay_lr_logend
        self.decay_lr_length = decay_lr_length
        self.initial_learning_rate = initial_learning_rate
        self.optimiser = optimiser

    def learning_rate_modify(self, epoch):
        """Modify the learning rate by cyclic factor or a decay """
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
        
        for param_group in self.optimiser.param_groups:
            param_group['lr'] = new_lr
        #self.optimiser.learning_rate = new_lr
        #print("learning_rate:, {}".format(self.optimiser.learning_rate))

    def on_epoch_begin(self, epoch, logs = None):
        self.learning_rate_modify(epoch)

class BatchRampCallback():

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

class TrainCallback():

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



class TimeCallback():
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

class SaveModelCallback():

    def __init__(self, model, optimiser, checkpoint_path, save_interval = 250):
        self.save_interval = save_interval
        self.checkpoint_path = checkpoint_path
        self.model = model
        self.optimiser = optimiser
        self.model_filename = "model.pt"
        if not os.path.isdir(self.checkpoint_path):
            os.makedirs(checkpoint_path)

    def on_epoch_end(self,epoch, logs = None):

        if epoch % self.save_interval == 0:
            self.model_filename = "model.pt" if self.model_filename == "model_2.pt" else "model_2.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimiser_state_dict": self.optimiser.state_dict()
                }, 
                os.path.join(self.checkpoint_path, self.model_filename))
        









