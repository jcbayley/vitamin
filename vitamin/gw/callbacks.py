import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from ..tools import make_ppplot, loss_plot, latent_corner_plot, latent_samp_fig
from ..train_plots import plot_posterior, plot_JS_div

class PosteriorComparisonCallback():


    def __init__(self, save_directory, model, comparison_posteriors, test_dataset, device = "cpu", n_samples = 10000, config=None, plot_latent = False, save_interval = 1000):
        self.test_dataset = test_dataset
        self.comparison_posteriors = comparison_posteriors
        self.n_samples = n_samples 
        self.device = device
        self.config = config
        self.save_directory = save_directory
        self.model = model
        self.save_interval = save_interval


        #if len(self.test_dataset) != len(comparison_posteriors):
        #    raise Exception(f"test dataset muse have same length as comparison posteriors, datalen: {len(self.test_dataset)}, post_len: {len(comparison_posteriors)}")

    def on_epoch_end(self, epoch, logs = None):
        
        if epoch % self.save_interval == 0:

            savedir = os.path.join(self.save_directory, f"epoch_{epoch}")
            if not os.path.isdir(savedir):
                os.makedirs(savedir)

            self.model.eval()
            with torch.no_grad():
                n_test_data = len(self.test_dataset.Y_noisy)

                start_time_test = time.time()
                samples, samples_r, samples_q = self.model.test(
                        torch.Tensor(self.test_dataset.Y_noisy).to(self.device), 
                        num_samples=self.n_samples, 
                        transform_func = None,
                        return_latent = True,
                        par = torch.Tensor(self.test_dataset.X).to(self.device)
                        )

                end_time_test = time.time()
                for step in range(n_test_data):
                    for key in self.test_dataset.samples_available.keys():
                        if step not in self.test_dataset.samples_available[key]:
                            continue
                    
                    if step > len(self.test_dataset) - 1:
                        break
                    if self.config["training"]["plot_latent"]:
                        if not os.path.isdir(os.path.join(savedir, "latent_dir")):
                            os.makedirs(os.path.join(savedir, "latent_dir"))
                        fig = latent_corner_plot(samples_r[step].squeeze(), samples_q[step].squeeze())
                        fig.savefig(os.path.join(savedir, "latent_dir", f"latent_plot_{step}.png"))

                    allinds = []
                    for samp, sampind in self.test_dataset.samples_available.items():
                        if step not in sampind:
                            allinds.append(step)
                    if len(allinds) == len(self.test_dataset.samples_available):
                        print("No available samples: {}".format(step))
                        continue

                    
                    if np.any(np.isnan(samples[step])):
                        print('Epoch: {}, found nans in samples. Not making plots'.format(epoch))
                        KL_est = [-1,-1,-1]
                    else:
                        print('Epoch: {}, Testing time elapsed for all {} samples: {}'.format(epoch,self.n_samples,end_time_test - start_time_test))
                        if len(np.shape(self.comparison_posteriors)) == 4:
                            JS_est, JS_labels = plot_posterior(
                                savedir,
                                samples[step],
                                self.test_dataset.truths[step],
                                epoch,
                                step,
                                all_other_samples=self.comparison_posteriors[:,step,:], 
                                config=self.config, 
                                unconvert_parameters = self.test_dataset.unconvert_parameters)
                            #plot_JS_div(JS_est[:10], JS_labels)
                        else:
                            print("not plotting posterior, bilby samples wrong shape")



class LoadDataCallback():

    def __init__(self, train_iterator, num_epoch_load):
        self.train_iterator = train_iterator
        self.num_epoch_load = num_epoch_load

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.num_epoch_load == 0:
            self.train_iterator.load_next_chunk()