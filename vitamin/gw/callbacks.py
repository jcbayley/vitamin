import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import time
import scipy.stats as st
from ..tools import make_ppplot, loss_plot, latent_corner_plot, latent_samp_fig
from ..train_plots import plot_posterior, plot_JS_div, plot_posterior_seaborn

class PosteriorComparisonCallback():


    def __init__(self, save_directory, model, comparison_posteriors, test_dataset, device = "cpu", n_samples = 10000, config=None, plot_latent = False, save_interval = 1000, save_pp_plot=True, grid_points=None):
        self.test_dataset = test_dataset
        self.comparison_posteriors = comparison_posteriors
        self.grid_points = grid_points
        self.n_samples = n_samples 
        self.device = device
        self.config = config
        self.save_directory = save_directory
        self.model = model
        self.save_interval = save_interval
        self.save_pp_plot = save_pp_plot


        #if len(self.test_dataset) != len(comparison_posteriors):
        #    raise Exception(f"test dataset muse have same length as comparison posteriors, datalen: {len(self.test_dataset)}, post_len: {len(comparison_posteriors)}")

    def plot_JS_grid(self, par_vals, labels, output_file):
        seaborn.set(font_scale=1.5)
        fig, ax = plt.subplots(figsize = (14,8))
        hm = seaborn.heatmap(np.array(par_vals)*1e3, annot=True, fmt='0.1f', annot_kws = {"fontsize":11}, cmap="cividis", cbar_kws={'label': 'JS divergence ($10^{-3}$)'}, linewidths=0.05)
        ax.set_xticks(np.arange(len(labels)) + 0.5,labels=labels)
        ax.set_xticklabels(labels, rotation=50)
        ax.set_ylabel("Injection", fontsize=20)
        ax.collections[0].colorbar.set_label('JS divergence ($10^{-3}$)', fontsize=20)
        plt.show()
        fig.savefig(output_file)

    def mean_JS_plot(self, par_vals, labels, output_file):
        fig, ax = plt.subplots(figsize = (14,8))
        hm = seaborn.boxplot(np.array(par_vals))
        ax.set_xticks(np.arange(len(labels)),labels=labels)
        ax.set_xticklabels(labels, rotation=50)
        ax.set_ylabel("JS divergence", fontsize=20)
        plt.show()
        fig.savefig(output_file)


    def make_js_divergence_grid(self, JS_est):
        pass

    def gen_pp_plot(self, samples, truths, savepath, labels):

        cred_region = []
        for i,res in enumerate(samples):
            test_cred = []
            for j,par in enumerate(labels):
                cred_reg = sum(res[:,j] < truths[i,j])/len(res[:,j])
                test_cred.append(cred_reg)
            cred_region.append(test_cred)
        cred_region = np.array(cred_region)

        fig, ax = plt.subplots()

        x_values = np.linspace(0,1,100)
        N = len(cred_region)

        confidence_interval=[0.68, 0.95, 0.997]
        confidence_interval_alpha=[0.1] * len(confidence_interval)
        for ci, alpha in zip(confidence_interval, confidence_interval_alpha):
            edge_of_bound = (1. - ci) / 2.
            lower = st.binom.ppf(1 - edge_of_bound, N, x_values) / N
            upper = st.binom.ppf(edge_of_bound, N, x_values) / N
            # The binomial point percent function doesn't always return 0 @ 0,
            # so set those bounds explicitly to be sure
            lower[0] = 0
            upper[0] = 0
            ax.fill_between(x_values, lower, upper, alpha=alpha, color='k')

        
        ax.plot(x_values, x_values, "k")
        for i, par in enumerate(labels):
            pp = np.array([sum(cred_region[:,i] < x)/len(cred_region[:,i]) for x in x_values])

            ax.plot(x_values, pp, label=par)
        

        ax.legend()
        ax.set_xlabel("Confidence Interval (CI)")
        ax.set_ylabel("Fraction of events in CI")
        fig.savefig(savepath)

    def on_epoch_end(self, epoch, logs = None):
        
        if epoch % self.save_interval == 0:

            savedir = os.path.join(self.save_directory, f"epoch_{int(epoch)}")
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
                JS_all = []
                JS_labels = []
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
                            
                            if self.grid_points is not None:
                                gp = self.grid_points[step]
                            else: 
                                gp = None
                            """
                            JS_est, JS_labels = plot_posterior(
                                savedir,
                                samples[step],
                                self.test_dataset.truths[step],
                                epoch,
                                step,
                                all_other_samples=self.comparison_posteriors[:,step,:], 
                                config=self.config, 
                                unconvert_parameters = self.test_dataset.unconvert_parameters,
                                grid_points = gp)
                            """
                            temp_JS_est, temp_JS_labels = plot_posterior_seaborn(
                                savedir,
                                samples[step],
                                self.test_dataset.truths[step],
                                epoch,
                                step,
                                all_other_samples=self.comparison_posteriors[:,step,:], 
                                config=self.config, 
                                unconvert_parameters = self.test_dataset.unconvert_parameters,
                                grid_points = gp
                            )
                            JS_all.append(temp_JS_est)
                            if JS_labels == []:
                                JS_labels = temp_JS_labels

                        else:
                            print("not plotting posterior, bilby samples wrong shape")
                if self.save_pp_plot:
                    if JS_labels is not None:
                        self.gen_pp_plot(samples, self.test_dataset.X, os.path.join(savedir, "ppplot.png"), JS_labels)
                    else:
                        print("WARNING: No JS labels")
                try:
                    if temp_JS_est is not None:
                        self.plot_JS_grid(np.squeeze(JS_all), JS_labels, os.path.join(savedir, "JS_plot.png"))
                        self.mean_JS_plot(np.squeeze(JS_all), JS_labels, os.path.join(savedir, "JS_plot_all.png"))
                except Exception as e:
                    print(str(e))





class LoadDataCallback():

    def __init__(self, train_iterator, num_epoch_load):
        self.train_iterator = train_iterator
        self.num_epoch_load = num_epoch_load

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.num_epoch_load == 0:
            self.train_iterator.load_next_chunk()