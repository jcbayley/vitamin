import torch
from ..tools import make_ppplot, loss_plot, latent_corner_plot, latent_samp_fig
from .test import test_model
from collections import OrderedDict

def adjust_learning_rate(lr, optimiser, epoch, factor = 1.0, epoch_num = 5, low_cut = 1e-12):
    """Sets the learning rate to the initial LR decayed by a factor 0.999 (factor) every 5 (epoch_num) epochs"""
    if lr <= low_cut:
        lr = lr
    else:
        lr = lr * (factor ** (epoch // epoch_num))
    for param_group in optimiser.param_groups:
        param_group['lr'] = lr

def setup_and_train(config):

    #params, bounds, masks, fixed_vals = get_params(params_dir = params_dir)
    run = time.strftime('%y-%m-%d-%X-%Z')

    # define which gpu to use during training
    try:
        #gpu_num = str(vitamin_config["training"]['gpu_num'])   
        #os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num
        print("CUDA DEV: ",os.environ["CUDA_VISIBLE_DEVICES"])
    except:
        print("No CUDA devices")


    from lal import GreenwichMeanSiderealTime
    from astropy.time import Time
    from astropy import coordinates as coord
    import corner
    from universal_divergence import estimate
    import natsort
    from scipy.spatial.distance import jensenshannon
    import scipy.stats as st
    import pickle
    import torchsummary
    from ..vitamin_model import CVAE
    #pip infrom ..callbacks import  PlotCallback, TrainCallback, TestCallback, TimeCallback, optimiserSave, LearningRateCallback, LogminRampCallback, AnnealCallback, BatchRampCallback
    from .load_data import DataSet, convert_ra_to_hour_angle, convert_hour_angle_to_ra, psiphi_to_psiX, psiX_to_psiphi, m1m2_to_chirpmassq, chirpmassq_to_m1m2
        
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_log_dir = os.path.join(config["output"]['output_directory'],'logs')

    training_directory = os.path.join(config["data"]["data_directory"], "training")
    validation_directory = os.path.join(config["data"]["data_directory"], "validation")
    test_directory = os.path.join(config["data"]["data_directory"], "test", "waveforms")

    epochs = config["training"]['num_iterations']
    plot_cadence = int(0.5*config["training"]["plot_interval"])
    checkpoint_path = os.path.join(config["output"]["output_directory"],"checkpoint","model.pt")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    dirs = [checkpoint_dir]
    for direc in dirs:
        if not os.path.isdir(direc):
            os.makedirs(direc)

    make_paper_plots = config["testing"]['make_paper_plots']

    # load the training data
    if not make_paper_plots:
        train_dataset = DataSet(training_directory,config = config)
        validation_dataset = DataSet(validation_directory,config=config,val_set = True)
        train_dataset.load_next_chunk()
        validation_dataset.load_next_chunk()

        print("VAL_SIZE: ", len(validation_dataset))
        print("TRAIN_SIZE: ", len(train_dataset))
        
    if config["training"]["test_interval"] != False:
        test_dataset = DataSet(test_directory,config=config, test_set = True)
        test_dataset.load_next_chunk()
        test_dataset.load_bilby_samples()

        # load precomputed samples
        bilby_samples = []
        for sampler in config["testing"]["samplers"][1:]:
            bilby_samples.append(test_dataset.sampler_outputs[sampler])
        bilby_samples = np.array(bilby_samples)

    start_epoch = 0
    
    model = CVAE(config, device=device).to(device)


    if config["training"]["transfer_model_checkpoint"] and not config["training"]["resume_training"]:
        checkpoint = torch.load(config["training"]["transfer_model_checkpoint"])
        #std = checkpoint["model_state_dict"]
        #std_new = OrderedDict((key.replace("Normal_", "TruncatedNormal_") if "Normal_" in key else key, v) for key, v in std.items())
        model.load_state_dict(checkpoint["model_state_dict"])
        print('... loading in previous model %s' % config["training"]["transfer_model_checkpoint"])

    elif config["training"]['resume_training']:
        if config["training"]["transfer_model_checkpoint"]:
            print(f"Warning: Continuing training from trained weights, not from pretrained model from {config['training']['transfer_model_checkpoint']}")
        # Load the previously saved weights
        #model = torch.load(os.path.join(checkpoint_dir,"model.pt"))
        checkpoint = torch.load(os.path.join(checkpoint_dir,"model.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        print('... loading in previous model %s' % os.path.join(checkpoint_dir,"model.pt"))

    
    model.forward(torch.ones((2, model.n_channels, model.y_dim)).to(device), torch.ones((2, model.x_dim)).to(device))
    with open(os.path.join(config["output"]["output_directory"], "model_summary.txt"),"w") as f:
        summary = torchsummary.summary(model, [(model.n_channels, model.y_dim), (model.x_dim, )], depth = 3)
        f.write(str(summary))

    if config["training"]["optimiser"] == "adam":
        optimiser = torch.optim.AdamW(model.parameters(), lr=config["training"]["initial_learning_rate"])
        if config["training"]["resume_training"]:
            optimiser.load_state_dict(checkpoint["optimiser_state_dict"])


    train_loop(
        model=model, 
        device=device, 
        optimiser=optimiser, 
        epochs=config["training"]['num_iterations'], 
        train_iterator=train_dataset, 
        learning_rate=config["training"]["initial_learning_rate"],
        validation_iterator=validation_dataset, 
        ramp_start = config["training"]["ramp_start"],
        ramp_end = config["training"]["ramp_start"] + config["training"]["ramp_length"],
        save_dir = config["output"]["output_directory"], 
        dec_rate = 0.9999, 
        dec_start = config["training"]["decay_lr_start"], 
        do_test=True, 
        low_cut = 1e-10, 
        test_data = test_dataset, 
        do_ramp = True,
        continue_train = config["training"]['resume_training'],
        start_epoch = start_epoch,
        test_interval=config["training"]["test_interval"],
        checkpoint_dir=checkpoint_dir,
        samplers = config["testing"]["samplers"],
        plot_interval=config["training"]["plot_interval"],
        config=config
        )


def train_batch(
    epoch, 
    model, 
    optimiser, 
    device, 
    batch, 
    labels, 
    pause = 0, 
    train = True, 
    ramp = 1.0):

    model.train(train)
    if train:
        optimiser.zero_grad()
        
    length = float(batch.size(0))
    # calculate r2, q and r1 means and variances

    recon_loss, kl_loss, par_loss, recon_losses = model.compute_loss(batch, labels, ramp)

    # calcualte total loss
    loss = recon_loss + ramp*kl_loss + (2 - ramp)*50*par_loss
    if train:
        loss.backward()
        # update the weights                                                                                                                              
        optimiser.step()

    return loss.item(), kl_loss.item(), recon_loss.item() 


def train_loop(
    model, 
    device, 
    optimiser, 
    epochs, 
    train_iterator, 
    learning_rate, 
    validation_iterator, 
    ramp_start = -1,
    ramp_end =-1,
    save_dir = "./", 
    dec_rate = 1.0, 
    dec_start = 0, 
    do_test=True, 
    low_cut = 1e-12, 
    test_data = None, 
    do_ramp = True,
    continue_train = True,
    start_epoch = 0,
    num_epoch_load = 1,
    test_interval = 1000,
    checkpoint_dir=None, 
    samplers = None, 
    config=None,
    plot_interval = 100
    ):


    train_losses = []
    kl_losses = []
    lik_losses = []
    val_losses = []
    val_kl_losses = []
    val_lik_losses = []
    train_times = []
    kl_start = ramp_start
    kl_end = ramp_end
    min_val_loss = np.inf
    prev_save_ep = 0

    if continue_train:
        with open(os.path.join(checkpoint_dir, "checkpoint_loss.txt"),"r") as f:
            old_epochs, train_losses, kl_losses, lik_losses, val_losses, val_kl_losses, val_lik_losses = np.loadtxt(f)
            old_epochs = list(old_epochs)
            train_losses = list(train_losses)
            kl_losses = list(kl_losses)
            lik_losses = list(lik_losses)
            val_losses = list(val_losses)
            val_kl_losses = list(val_kl_losses)
            val_lik_losses = list(val_lik_losses)
            
    start_train_time = time.time()


    for epoch in range(epochs):
        if continue_train:
            epoch = epoch + old_epochs[-1]

        model.train()
        model.device = device
        model.to(device)

        if epoch > dec_start:
            adjust_learning_rate(learning_rate, optimiser, epoch - dec_start, factor=dec_rate, low_cut = low_cut)
            
        if do_ramp:
            ramp = 0.0
            if epoch>kl_start and epoch<=kl_end:
                #ramp = (np.log(epoch)-np.log(kl_start))/(np.log(kl_end)-np.log(kl_start)) 
                ramp = (epoch - kl_start)/(kl_end - kl_start)
            elif epoch>kl_end:
                ramp = 1.0 
            else:
                ramp = 0.0
        else:
            ramp = 1.0
        
        model.ramp = ramp
        print("Model ramp: ", model.ramp, ramp)
        # Training    

        temp_train_loss = 0
        temp_kl_loss = 0
        temp_lik_loss = 0
        it = 0
        total_time = 0

        #for local_batch, local_labels in train_iterator:
        for ind in range(len(train_iterator)):
            # Transfer to GPU            
            local_batch, local_labels = train_iterator[ind]
            local_batch, local_labels = torch.Tensor(local_batch).to(device), torch.Tensor(local_labels).to(device)
            start_time = time.time()
            train_loss,kl_loss,lik_loss = train_batch(epoch, model, optimiser, device, local_batch,local_labels, ramp=ramp, train=True)
            temp_train_loss += train_loss
            temp_kl_loss += kl_loss
            temp_lik_loss += lik_loss
            it += 1
            total_time += time.time() - start_time
        

        val_it = 0
        temp_val_loss = 0
        temp_val_kl_loss = 0
        temp_val_lik_loss = 0
        val_time = 0
        # validation
        #print("VAL_LEN:", len(validation_iterator))
        #for val_batch, val_labels in validation_iterator:
        for ind in range(len(validation_iterator)):
            # Transfer to GPU            
            val_batch, val_labels = validation_iterator[ind]
            val_batch, val_labels = torch.Tensor(val_batch).to(device), torch.Tensor(val_labels).to(device)
            val_loss,val_kl_loss,val_lik_loss = train_batch(epoch, model, optimiser, device, val_batch, val_labels, ramp=ramp, train=False)
            temp_val_loss += val_loss
            temp_val_kl_loss += val_kl_loss
            temp_val_lik_loss += val_lik_loss
            val_it += 1

        temp_val_loss /= val_it
        temp_val_kl_loss /= val_it
        temp_val_lik_loss /= val_it

        temp_train_loss /= it
        temp_kl_loss /= it
        temp_lik_loss /= it
        batch_time = total_time/it
        post_train_time = time.time()
        
        val_losses.append(temp_val_loss)
        val_kl_losses.append(temp_val_kl_loss)
        val_lik_losses.append(temp_val_lik_loss)
        train_losses.append(temp_train_loss)
        kl_losses.append(temp_kl_loss)
        lik_losses.append(temp_lik_loss)
        train_times.append(post_train_time - start_train_time)

        diff_ep = epoch - prev_save_ep

        if epochs % 1 == 0:
            print(f"Train:      Epoch: {epoch}, Training loss: {temp_train_loss}, kl_loss: {temp_kl_loss}, l_loss:{temp_lik_loss}, Epoch time: {total_time}, batch time: {batch_time}")
            print(f"Validation: Epoch: {epoch}, Training loss: {temp_val_loss}, kl_loss: {val_kl_loss}, l_loss:{val_lik_loss}, Epoch time: {total_time}, batch time: {batch_time}")

        if epoch % plot_interval == 0:
            loss_plot(save_dir, train_losses, kl_losses, lik_losses, val_losses, val_kl_losses, val_lik_losses)
            with open(os.path.join(checkpoint_dir, "checkpoint_loss.txt"),"w+") as f:
                if len(train_losses) > epoch+1:
                    epoch = len(train_losses) - 1
                savearr =  np.array([np.arange(epoch+1), train_losses, kl_losses, lik_losses, val_losses, val_kl_losses, val_lik_losses]).astype(float)
                np.savetxt(f,savearr)
            torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimiser_state_dict": optimiser.state_dict(),
                            "loss": temp_train_loss,
                        }, 
                        os.path.join(checkpoint_dir,"model.pt"))
            min_val_loss = temp_val_loss#np.inf
            prev_save_ep = 0
            """
            with open(os.path.join(checkpoint_dir, "checkpoint_times.txt"),"w+") as f:
                if len(train_times) > epoch+1:
                    t_epoch = len(train_times) - 1
                else:
                    t_epoch = epoch
                savearr =  np.array([np.arange(t_epoch+1), train_times]).astype(float)
                np.savetxt(f,savearr)
            """
                


        print("TEST:", epoch, test_interval, int(int(epoch) % int(test_interval)), do_test)
        if int(int(epoch) % int(test_interval)) == 0:
            if do_test or epoch == epochs - 1:
                # test plots
                if not os.path.isdir(os.path.join(save_dir, f"epochs_{int(epoch)}")):
                    os.makedirs(os.path.join(save_dir, f"epochs_{int(epoch)}"))
                #samples,tr,zr_sample,zq_sample = run_latent(model,validation_iterator,num_samples=1000,device=device)                                                   
                #lat2_fig = latent_samp_fig(zr_sample,zq_sample,tr)
                #lat2_fig.savefig(os.path.join(save_dir, f"epochs_{int(epoch)}", f"zsample_epoch{int(epoch)}.png"))
                #del samples, tr, zr_sample
                #plt.close(lat2_fig)
                if test_data is not None:

                    test_data_plot = True
                    if test_data_plot:
                        fig, ax = plt.subplots()
                        ax.plot(test_data.Y_noisy[0])
                        ax.set_title(test_data.X[0])
                        fig.savefig(os.path.join(save_dir, f"epochs_{int(epoch)}", f"test_data_plot.png"))

                        fig, ax = plt.subplots()
                        ax.plot(train_iterator.Y_noisefree[0].T)
                        ax.set_title(train_iterator.X[0])
                        fig.savefig(os.path.join(save_dir, f"epochs_{int(epoch)}", f"train_data_plot.png"))

                    # load precomputed samples
                    bilby_samples = []
                    for sampler in samplers[1:]:
                        bilby_samples.append(test_data.sampler_outputs[sampler])
                    bilby_samples = np.array(bilby_samples)
                    #model.device = "cpu"
                    #model.to("cpu")
                    test_model(
                        os.path.join(save_dir,f"epochs_{int(epoch)}"), 
                        model=model,
                        test_dataset=test_data, 
                        bilby_samples=bilby_samples,
                        epoch=epoch,
                        n_samples=5000,
                        config=config,
                        device=device,
                        plot_latent = True
                        ) 
                        
                    # save the model
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "loss": temp_train_loss,
                        }, 
                        os.path.join(save_dir,f"epochs_{int(epoch)}","model.pt"))

                    print("done_test")
                else:
                    print("No test data")
            else:
                print("Not doing test")

        
        if epoch % num_epoch_load == 0:
            train_iterator.load_next_chunk()

    return train_losses, kl_losses, lik_losses, val_losses, val_kl_losses, val_lik_losses


def run_latent(model,test_it,num_samples = 500,device="cpu",transform_func=None):
    # set the evaluation mode                                                                                                                                                                                    
    model.eval()

    # test loss for the data                                                                                                                                                                                     
    test_loss = 0
    samples = []
    # do not need to track gradients                                                                                                
    with torch.no_grad():
        #for local_batch, local_labels in test_it:
        for ind in range(len(test_it)):
            # Transfer to GPU         
            local_batch, local_labels  = test_it[ind]
            local_batch, local_labels = torch.Tensor(local_batch).to(device), torch.Tensor(local_labels).to(device)
            x_samples, zr_samples, zq_samples = model.test_latent(local_batch, local_labels, num_samples)
            truths = local_labels
            break
    return x_samples, truths, zr_samples, zq_samples


def run_latent2(model,test_it,num_samples = 500,device="cpu",transform_func=None, return_latent = True):
    # set the evaluation mode                                                                                                                                                                                    
    model.eval()

    # test loss for the data                                                                                                                                                                                     
    test_loss = 0
    samples = []
    # do not need to track gradients                                                                                                
    with torch.no_grad():
        local_batch, local_labels = torch.Tensor(test_it.dataset.data).to(device), torch.Tensor(test_it.dataset.labels).to(device)
        transformed_samples, net_samples, zr_samples, zq_samples = model.test(local_batch, local_freqs, num_samples, par = local_labels, transform_func = transform_func, return_latent=return_latent)
        truths = local_labels.to("cpu")
    return transformed_samples, net_samples, truths, zr_samples, zq_samples



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
    import bilby

    bilby.core.utils.log.setup_logger(outdir='./', label=None, log_level='warning', print_version=False)
    
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

    setup_and_train(vitamin_config)

