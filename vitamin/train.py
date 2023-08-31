import torch
import time
from .tools import make_ppplot, loss_plot, latent_corner_plot, latent_samp_fig
from collections import OrderedDict
import os
import numpy as np


def adjust_learning_rate(lr, optimiser, epoch, factor = 1.0, epoch_num = 5, low_cut = 1e-12):
    """Sets the learning rate to the initial LR decayed by a factor 0.999 (factor) every 5 (epoch_num) epochs"""
    if lr <= low_cut:
        lr = lr
    else:
        lr = lr * (factor ** (epoch // epoch_num))
    for param_group in optimiser.param_groups:
        param_group['lr'] = lr

def train_batch(
    epoch, 
    model, 
    optimiser, 
    device, 
    batch, 
    labels, 
    pause = 0, 
    train = True):

    model.train(train)
    if train:
        optimiser.zero_grad()
        
    length = float(batch.size(0))
    # calculate r2, q and r1 means and variances

    recon_loss, kl_loss, par_loss, recon_losses = model.compute_loss(batch, labels, model.ramp)

    # calcualte total loss
    loss = recon_loss + model.ramp*kl_loss + (2 - model.ramp)*50*par_loss
    if train:
        loss.backward()
        # update the weights                                                                                                                              
        optimiser.step()

    return loss.item(), kl_loss.item(), recon_loss.item(), par_loss, recon_losses


def train_loop(
    model, 
    device, 
    optimiser, 
    n_epochs, 
    train_iterator, 
    validation_iterator = None, 
    save_dir = "./", 
    continue_train = False,
    start_epoch = 0,
    num_epoch_load = 1,
    checkpoint_dir=None,
    callbacks = [],
    verbose = False
    ):


    logs = dict(
        train_losses = [],
        kl_losses = [],
        lik_losses = [],
        val_losses = [],
        val_kl_losses = [],
        val_lik_losses = [],
        train_times = [],
        )
    
    prev_save_ep = 0


    # set number of epochs from number of iterations
    try:
        n_batches = len(train_iterator)
    except:
        print('Error: could not find number of batches per epoch, setting n_batches=1 (n_iterations == n_epochs)')
        n_batches = 1
    #n_epochs = n_iterations/n_batches


    if continue_train:
        with open(os.path.join(checkpoint_dir, "checkpoint_loss.txt"),"r") as f:
            old_epochs, train_times, train_losses, kl_losses, lik_losses, val_losses, val_kl_losses, val_lik_losses = np.loadtxt(f)
            old_epochs = list(old_epochs)
            logs["train_times"] = list(train_times)
            logs["train_losses"] = list(train_losses)
            logs["kl_losses"] = list(kl_losses)
            logs["lik_losses"] = list(lik_losses)
            logs["val_losses"] = list(val_losses)
            logs["val_kl_losses"] = list(val_kl_losses)
            logs["val_lik_losses"] = list(val_lik_losses)

        #checkpoint = torch.load(os.path.join(checkpoint_dir,"model.pt"), map_location=device)
        #model.load_state_dict(checkpoint["model_state_dict"])
        #optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

    start_train_time = time.time()

    model_filename = "model.pt"


    for epoch in range(n_epochs):
        total_start_time = time.time()
        iteration = epoch*n_batches

        if continue_train:
            epoch = epoch + old_epochs[-1]

        model.train()
        model.device = device
        model.to(device)

        temp_train_loss = 0
        temp_kl_loss = 0
        temp_lik_loss = 0
        it = 0
        total_time = 0
        load_batch_time = 0
        
        #for local_batch, local_labels in train_iterator:
        for ind in range(len(train_iterator)):
            # Transfer to GPU            
            ld_time = time.time()
            local_batch, local_labels = train_iterator[ind]
            local_batch, local_labels = torch.Tensor(local_batch).to(device), torch.Tensor(local_labels).to(device)
            load_batch_time += time.time() - ld_time
            start_time = time.time()
            train_loss,kl_loss,lik_loss, par_loss, recon_losses = train_batch(epoch, model, optimiser, device, local_batch,local_labels, train=True)
            temp_train_loss += train_loss
            temp_kl_loss += kl_loss
            temp_lik_loss += lik_loss
            it += 1
            total_time += time.time() - start_time
        

        if validation_iterator is not None:
            val_it = 0
            temp_val_loss = 0
            temp_val_kl_loss = 0
            temp_val_lik_loss = 0
            val_time = time.time()
            # validation
            #print("VAL_LEN:", len(validation_iterator))
            #for val_batch, val_labels in validation_iterator:
            for ind in range(len(validation_iterator)):
                # Transfer to GPU            
                val_batch, val_labels = validation_iterator[ind]
                val_batch, val_labels = torch.Tensor(val_batch).to(device), torch.Tensor(val_labels).to(device)
                val_loss,val_kl_loss,val_lik_loss, par_loss, recon_losses = train_batch(epoch, model, optimiser, device, val_batch, val_labels, train=False)
                temp_val_loss += val_loss
                temp_val_kl_loss += val_kl_loss
                temp_val_lik_loss += val_lik_loss
                val_it += 1
            val_time = time.time() - val_time

            temp_val_loss /= val_it
            temp_val_kl_loss /= val_it
            temp_val_lik_loss /= val_it

            logs["val_losses"].append(temp_val_loss)
            logs["val_kl_losses"].append(temp_val_kl_loss)
            logs["val_lik_losses"].append(temp_val_lik_loss)

        temp_train_loss /= it
        temp_kl_loss /= it
        temp_lik_loss /= it
        batch_time = total_time/it
        load_batch_time = load_batch_time/it
        post_train_time = time.time()
        
        logs["train_losses"].append(temp_train_loss)
        logs["kl_losses"].append(temp_kl_loss)
        logs["lik_losses"].append(temp_lik_loss)
        logs["train_times"].append(post_train_time - start_train_time)

        diff_ep = epoch - prev_save_ep

        total_duration = time.time() - total_start_time

        if epoch % 1 == 0:
            print(f" ------- Epoch: {epoch}, Epoch time: {total_duration}, batch time: {batch_time}, load_batch_time: {load_batch_time}, train_time: {total_time}, ramp: {model.ramp}")
            print(f" ------- Train:      Iteration: {iteration}, Training loss: {temp_train_loss}, kl_loss: {temp_kl_loss}, l_loss:{temp_lik_loss}")
            if validation_iterator is not None:
                print(f" ------- Validation: Iteration: {iteration}, Training loss: {temp_val_loss}, kl_loss: {val_kl_loss}, l_loss:{val_lik_loss}")

        for callback in callbacks:
            #try:
            callback.on_epoch_end(epoch, logs)
            #except Exception as e:
            #    print(f"Could not run callback {type(callback).__name__}: ")
            #    print(e)

    return logs["train_losses"], logs["kl_losses"], logs["lik_losses"], logs["val_losses"], logs["val_kl_losses"], logs["val_lik_losses"]



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
    return transformed_samples, net_samples, truths, zr_samples, zq_sample