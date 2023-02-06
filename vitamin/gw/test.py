import torch
import numpy as np
import time
from ..train_plots import plot_posterior, plot_JS_div
from ..tools import latent_corner_plot
import os

def test_model(
    save_dir, 
    model, 
    test_dataset, 
    bilby_samples,
    epoch, 
    n_samples,
    plot_latent = False, 
    latent_dir = None, 
    config=None,
    device="cpu"
    ):
    """ Test the model by comparing posterior distributions"""

    model.eval()
    with torch.no_grad():
        n_test_data = len(test_dataset)

        samples, samples_r, samples_q = model.test(
                torch.Tensor(test_dataset.Y_noisy).to(device), 
                num_samples=n_samples, 
                transform_func = None,
                return_latent = True,
                par = torch.Tensor(test_dataset.X).to(device)
                )

        for step in range(n_test_data):
            if step not in test_dataset.samples_available["dynesty"]:
                continue
            
            if step > len(test_dataset):
                break
            if plot_latent:
                if not os.path.isdir(os.path.join(save_dir, "latent_dir")):
                    os.makedirs(os.path.join(save_dir, "latent_dir"))
                fig = latent_corner_plot(samples_r[step].squeeze(), samples_q[step].squeeze())
                fig.savefig(os.path.join(save_dir, "latent_dir", f"latent_plot_{step}.png"))

            allinds = []
            for samp, sampind in test_dataset.samples_available.items():
                if step not in sampind:
                    allinds.append(step)
            if len(allinds) == len(test_dataset.samples_available):
                print("No available samples: {}".format(step))
                continue

            start_time_test = time.time()

            end_time_test = time.time()
            if np.any(np.isnan(samples[step])):
                print('Epoch: {}, found nans in samples. Not making plots'.format(epoch))
                KL_est = [-1,-1,-1]
            else:
                print('Epoch: {}, Testing time elapsed for {} samples: {}'.format(epoch,n_samples,end_time_test - start_time_test))
                if len(np.shape(bilby_samples)) == 4:
                    JS_est, JS_labels = plot_posterior(
                        save_dir,
                        samples[step],
                        test_dataset.truths[step],
                        epoch,
                        step,
                        all_other_samples=bilby_samples[:,step,:], 
                        config=config, 
                        unconvert_parameters = test_dataset.unconvert_parameters)
                    #plot_JS_div(JS_est[:10], JS_labels)
                else:
                    print("not plotting posterior, bilby samples wrong shape")

if __name__ == "__main__":

    import os
    import shutil
    import h5py
    import json
    import sys
    from sys import exit
    import time
    import argparse
    import matplotlib.pyplot as plt
    import numpy as np
    
    parser = argparse.ArgumentParser(description='Input files and options')
    parser.add_argument('--ini-file', metavar='i', type=str, help='path to ini file')
    #parser.add_argument('--gpu', metavar='i', type=int, help='path to ini file', default = None)
    args = parser.parse_args()

    """
    if args.gpu is not None:
        # define which gpu to use during training
        try:
            os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
            print("SET CUDA DEV: ",os.environ["CUDA_VISIBLE_DEVICES"])
        except:
            print("No CUDA devices")
    """

    from .gw_parser import GWInputParser

    vitamin_config = GWInputParser(args.ini_file)

    test(vitamin_config)

