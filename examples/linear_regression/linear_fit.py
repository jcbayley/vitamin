from signal import SIGRTMAX
from tarfile import LENGTH_LINK
import vitamin
import numpy as np
import torch
#import tensorflow as tf
import os
import pickle
import pymc as pm
from torchsummary import summary
import matplotlib.pyplot as plt
import scipy.stats as st

output_dir = "./outputs_analytic_fulltest/"
appended = ""

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

def data_model(x, ms):
    #define a simple straight line model
    y = 0
    for i, m in enumerate(ms):
        y += ms[i]*(x**i)
    return y


def get_dataset(xdat, num_data, prior_mean, prior_cov, length = 100, sigma=0.1, num_params = 2):
    """generates some polynomial signals with a gaussian prior on the parameters

    Args:
        xdat (_type_): _description_
        num_data (_type_): _description_
        prior_mean (_type_): _description_
        prior_cov (_type_): _description_
        length (int, optional): _description_. Defaults to 100.
        sigma (float, optional): _description_. Defaults to 0.1.
        num_params (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    y = []
    x = []
    # this is to compare with the analytic case                                                                                                                       
    an_mvn = st.multivariate_normal(prior_mean, prior_cov)
    for i in range(num_data):
        ms = an_mvn.rvs(1)
        #ms = np.random.uniform(size = num_params)                                                                                                                    
        # make sure data has 3 dimensions (number_examples, number_datapoints, number_channels)                                                                       

        y.append(np.expand_dims(data_model(xdat,ms) + np.random.normal(0,sigma,size=len(xdat)),-1))
        x.append(ms)
    return np.array(x),np.swapaxes(np.array(y), 2, 1)

num_params = 2
length = 50
xdat = np.linspace(0,1,length)
sigma = 0.1
prior_mean = np.zeros(num_params) + 0.1
prior_cov = np.identity(num_params)*0.01

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
minbound, maxbound = -1,1

def convert_parameters(dataset, direction="forward"):
    if direction=="forward":
        rescale_par = (dataset - minbound)/(maxbound-minbound)
    elif direction=="backward":
        rescale_par = (dataset * (maxbound - minbound)) + minbound
    else:
        raise Exception(f"Please specify direction as forward or backward, not: {direction}")
        
    return rescale_par

# a few conditions to choose which samplers to run
generate_test = False
load_test = True
load_network = False
train_network = True
test_network = True
extra_tests = False
run_mcmc_sampler = False
make_test_plots = True

if generate_test:   
    #generate test_data
    test_dat = get_dataset(xdat, 500, prior_mean, prior_cov, num_params=num_params, length=length, sigma=sigma)
    with open(os.path.join(output_dir, "test_data.pkl"), "wb") as f:
        pickle.dump(test_dat, f)

elif load_test:
    # load test data if already generated
    with open(os.path.join(output_dir, "test_data.pkl"), "rb") as f:
        test_dat = pickle.load(f)


#Set up the CVAE parmaeters and bounds
#parameters to infer and the output distributions of the CVAE (default is Truncated Normal)
inf_pars = {f"p{i}":"Normal" for i in range(num_params)}
# the bounds for each of the parameters, used to rescale parameters internally
bounds = {f"p{i}_{bnd}":val for i in range(num_params) for bnd,val in zip(["min","max"], [0,1])}
# layers shared across the three networks (all available layers are defined in Docs)
# alternativly can define own models
shared_network = ['Conv1D(8, 16, 1, 2)', 'Conv1D(8, 16, 1, 2)', 'Conv1D(8, 3, 1, 2)']
# three individual networks designs to come after the shared network                                                                                                  
r1_network = ['Linear(64)', 'Linear(32)', 'Linear(16)']
r2_network = ['Linear(64)', 'Linear(32)', 'Linear(16)']
q_network = ['Linear(64)', 'Linear(32)', 'Linear(16)']


# initialise the model 
model = vitamin.vitamin_model.CVAE(z_dim=3, # latent space size
                                n_modes = 1, # number of modes in the latent space
                                x_dim = num_params,  # number of parameters to infer
                                inf_pars=inf_pars, # inference parameters
                                bounds=bounds, # inference parameters bounds
                                y_dim=length, # number of datapoints
                                n_channels=1, # number of input channels
                                shared_network=shared_network,
                                r1_network=r1_network,
                                r2_network=r2_network,
                                q_network=q_network,
                                logvarmin=True,
                                device = device).to(device)

model.forward(torch.ones((1, model.n_channels, model.y_dim)).to(device), torch.ones((1, model.x_dim)).to(device))
summary(model, [(1, length), (num_params, )])


# define the optimiser
optimiser = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.LinearLR(optimiser, start_factor=1.0, end_factor=0.01, total_iters=200)

# compile the model using optimiser (if using CPU you can run eagerly)
#model.compile(optimizer=optimizer,run_eagerly = True, loss=model.compute_loss)
#model.compile(optimizer=optimizer, loss=model.compute_loss)

if train_network == True:

    epoch_start = 0
    if load_network:
        checkpoint = torch.load(os.path.join(output_dir,f"model{appended}.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        epoch_start = checkpoint["epoch"]


    def train_batch(model, optimiser, data, labels, ramp = 1.0, r2_ramp = 1.0, train = True):

        model.train(train)
        if train:
            optimiser.zero_grad()
        length = float(data.size(0))
        # calculate r2, q and r1 means and variances
        
        recon_loss, kl_loss, outs = model.compute_loss(data, labels, ramp, r2_ramp)
        # calcualte total loss
        loss = recon_loss + ramp*kl_loss
        
        if train:
            loss.backward()
            # update the weights                                                                                                                              
            optimiser.step()

        return loss.item(), kl_loss.item(), recon_loss.item() 


    if load_network:
        with open(os.path.join(output_dir, "loss.pkl"), "rb") as f:
            all_losses = pickle.load(f)
    else:
        all_losses = [[], [], [], [], [], []]

    n_epochs = 200
    
    # generate the training dataset and the validation dataset
    tr_dataset = get_dataset(xdat, 500000, prior_mean, prior_cov, num_params=num_params, length=length, sigma=sigma)
    val_dataset = get_dataset(xdat, 1000, prior_mean, prior_cov, num_params=num_params, length=length, sigma=sigma)
    # rescale training parameters to be between 0 and 1 (from the range [-1,1])    
    rescale_train_par = convert_parameters(tr_dataset[0], direction="forward")
    rescale_val_par = convert_parameters(val_dataset[0], direction="forward")
    rescale_test_par = convert_parameters(test_dat[0], direction="forward")
    train_dat = torch.utils.data.DataLoader([[tr_dataset[1][i], rescale_train_par[i]] for i in range(len(tr_dataset[0]))], batch_size = 512, shuffle=True)
    val_dat = torch.utils.data.DataLoader([[val_dataset[1][i], rescale_val_par[i]] for i in range(len(val_dataset[0]))], batch_size = 512)   

    test_dataloader = torch.utils.data.DataLoader([[test_dat[1][i], rescale_test_par[i]] for i in range(len(test_dat[0]))], batch_size = 512)   

    print("Made dataset.....")
    plot_latent = True
    ramp_start, ramp_length = 10,20
    r2_ramp_start, r2_ramp_length = 30,20
    for epoch in range(n_epochs):
        epoch = epoch + epoch_start

        if not load_network:
            if epoch < ramp_start:
                ramp = 0.0
            elif epoch >= ramp_start and epoch <= ramp_start + ramp_length:
                ramp = (epoch - ramp_start)/(ramp_length)
            else:
                ramp = 1.0

            if epoch < 100:
                r2_ramp = 0.0
            elif epoch >= r2_ramp_start and epoch <= r2_ramp_start + r2_ramp_length:
                r2_ramp = (epoch - r2_ramp_start)/(r2_ramp_length)
            else:
                r2_ramp = 1.0
            r2_ramp = 1.0
        else:
            ramp = 1.0
            r2_ramp = 1.0

        if epoch > 400:
            scheduler.step()

        total_tr_loss = 0
        total_tr_kl_loss = 0
        total_tr_recon_loss = 0
        for tr_d, tr_l in train_dat:
            tr_loss, tr_kl_loss, tr_recon_loss = train_batch(model, optimiser, tr_d.to(device).float(), tr_l.to(device).float(), ramp, r2_ramp, train = True) 
            total_tr_loss += tr_loss
            total_tr_kl_loss += tr_kl_loss
            total_tr_recon_loss += tr_recon_loss
        total_tr_loss /= len(train_dat)
        total_tr_kl_loss /= len(train_dat)
        total_tr_recon_loss /= len(train_dat)

        total_v_loss = 0
        total_v_kl_loss = 0
        total_v_recon_loss = 0
        for v_d, v_l in val_dat:
            v_loss, v_kl_loss, v_recon_loss = train_batch(model, optimiser, v_d.to(device).float(), v_l.to(device).float(), ramp, r2_ramp, train = False) 
            total_v_loss += v_loss
            total_v_kl_loss += v_kl_loss
            total_v_recon_loss += v_recon_loss
        total_v_loss /= len(val_dat)
        total_v_kl_loss /= len(val_dat)
        total_v_recon_loss /= len(val_dat)

        all_losses[0].append(total_tr_loss)
        all_losses[3].append(total_v_loss)
        all_losses[1].append(total_tr_kl_loss)
        all_losses[4].append(total_v_kl_loss)
        all_losses[2].append(total_tr_recon_loss)
        all_losses[5].append(total_v_recon_loss)


        if epoch % 10 == 0 and epoch > 2:
            print(f"Train Epoch {epoch}: Totloss: {total_tr_loss}, klloss: {total_tr_kl_loss}, reconloss: {total_tr_recon_loss}")
            print(f"Val   Epoch {epoch}: Totloss: {total_v_loss}, klloss: {total_v_kl_loss}, reconloss: {total_v_recon_loss}")

            fig, ax = plt.subplots()
            ax.plot(np.arange(epoch + 1), all_losses[0], color = "C0", label = "total")
            ax.plot(np.arange(epoch + 1), all_losses[3], color = "C0", ls = "--")
            ax.plot(np.arange(epoch + 1), all_losses[1], color = "C1", label = "KL")
            ax.plot(np.arange(epoch + 1), all_losses[4], color = "C1", ls = "--")
            ax.plot(np.arange(epoch + 1), all_losses[2], color = "C2", label = "recon")
            ax.plot(np.arange(epoch + 1), all_losses[5], color = "C2", ls = "--")
            ax.set_yscale("symlog")
            fig.savefig(os.path.join(output_dir, "losses.png"))

            # save outputs
            with open(os.path.join(output_dir, "loss.pkl"), "wb") as f:
                pickle.dump(all_losses, f)

            if epoch > ramp_length + ramp_start:
                start_plot_ind = ramp_length + ramp_start
                fig, ax = plt.subplots()
                ax.plot(np.arange(epoch + 1)[start_plot_ind:], all_losses[0][start_plot_ind:], color = "C0", label = "total")
                ax.plot(np.arange(epoch + 1)[start_plot_ind:], all_losses[3][start_plot_ind:], color = "C0", ls = "--")
                #ax.plot(np.arange(epoch + 1)[start_plot_ind:], all_losses[1][start_plot_ind:], color = "C1", label = "KL")
                #ax.plot(np.arange(epoch + 1)[start_plot_ind:], all_losses[4][start_plot_ind:], color = "C1", ls = "--")
                ax.plot(np.arange(epoch + 1)[start_plot_ind:], all_losses[2][start_plot_ind:], color = "C2", label = "recon")
                ax.plot(np.arange(epoch + 1)[start_plot_ind:], all_losses[5][start_plot_ind:], color = "C2", ls = "--")
                ax.set_yscale("symlog")
                fig.savefig(os.path.join(output_dir, "losses_afterramp.png"))

        if plot_latent and epoch % 200 == 0:
            for data, pars in test_dataloader:
                samples, samples_r, samples_q = model.test(
                    torch.Tensor(data).float().to(device), 
                    num_samples=1000, 
                    transform_func = None,
                    return_latent = True,
                    par = torch.Tensor(pars).float().to(device)
                    )  
                latent_dir = os.path.join(output_dir, "latent_space")
                if not os.path.isdir(latent_dir):
                    os.makedirs(latent_dir)
                for step in range(2):
                    fig = vitamin.tools.latent_corner_plot(samples_r[step].squeeze(), samples_q[step].squeeze())
                    fig.savefig(os.path.join(latent_dir, f"latent_st{step}_ep{epoch}.png"))
                break

        if epoch % 50 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "loss": all_losses[0][-1],
                }, 
                os.path.join(output_dir,f"model{appended}.pt"))


    # save the weights of the model
    # save optimiser status
    #torch.save(optimiser.stat_dict(), os.path.join(output_dir,f"model{appended}.pt"))
    print("training done ....")


if test_network:
    # load the weights of pretrained model                                                                                                                            
    if train_network == False:
        checkpoint = torch.load(os.path.join(output_dir,f"model{appended}.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

    # generate some samples (Run each sample through individually with shape (1, datapoints, channels))   
    for data, pars in test_dataloader: 
        samples, samples_r, samples_q = model.test(
                    torch.Tensor(data).float().to(device), 
                    num_samples=10000, 
                    transform_func = None,
                    return_latent = True,
                    par = torch.Tensor(pars).float().to(device)
                    )     
        # convert back to original range
        samples = convert_parameters(samples, direction="backward")
        break                                                      

    # analytic posterior
    an_posts = []
    for td in range(len(test_dat[1])):
        # analytic posterior                                                                                                                                          
        phi = [np.ones((length, 1))]
        for i in range(num_params - 1):
            phi.append(np.expand_dims(np.power(xdat,i+1), -1))
        phi = np.concatenate(phi, axis = 1)
        prior_cov_inv = np.linalg.inv(prior_cov)
        posterior_cov = sigma**2 * np.linalg.inv(sigma**2 * prior_cov_inv + phi.T @ phi)
        posterior_mean = posterior_cov @ (prior_cov_inv @ prior_mean + phi.T @ np.array(test_dat[1][td]).flatten() / sigma**2)
        an_posts.append([posterior_mean, posterior_cov])

    with open(os.path.join(output_dir, f"samples{appended}.pkl"), "wb") as f:
        pickle.dump(samples, f)

    with open(os.path.join(output_dir, f"analytic_meancov{appended}.pkl"), "wb") as f:
        pickle.dump(an_posts, f)

    print("testing done ... ")
    
if extra_tests:
    samples, samples_r, samples_q = model.extra_test(
                    torch.Tensor(data).float().to(device), 
                    num_samples=5000, 
                    transform_func = None,
                    return_latent = True,
                    par = torch.Tensor(pars).float().to(device)
                    )  

    for i in range(len(samples_r)):
        fig = corner.corner(samples_r)
        fig = corner.corner(samples_q, fig=fig)
        fig.savefig(os.path.join(output_dir, "latent_comp", f"latent_zq_{i}.png"))

if run_mcmc_sampler:
    # run mcmc on the same test data using pymc                                                                                                                       
    mcmc_samples = []
    # initialise the x data                                                                                                                                           
    #loop over all of the test data                                                                                                                                   
    for td in range(len(test_dat[1])):
        # setup pymc model                                                                                                                                            
        with pm.Model() as gauss_model:
            # uniform priors on each of the parameters as in the training data                                                                                        
            priors = [pm.Normal(f"p{i}",prior_mean[i],np.sqrt(np.diag(prior_cov)[i])) for i in range(num_params)]
            # Gaussian likelihood with fixed sigma as in training                                                                                                     
            lik = pm.Normal("lik", mu=data_model(xdat,priors), sigma=sigma, observed = np.squeeze(test_dat[1][td]))

            # setup sampler and generate samples                                                                                                                      
            mcmc_samples.append(pm.sample(2000, chains=5))

    with open(os.path.join(output_dir,"mcmc_samples.pkl"),"wb") as f:
        pickle.dump(mcmc_samples, f)


if make_test_plots:

    with open(os.path.join(output_dir,"mcmc_samples.pkl"),"rb") as f:
        mcmc_samples = pickle.load(f)

    mc_samps = []
    for ind in range(len(mcmc_samples)):
        mc_samps.append(np.array([np.concatenate(np.array(getattr(mcmc_samples[ind].posterior,f"p{pnum}"))) for pnum in range(num_params)]))

    with open(os.path.join(output_dir, f"samples{appended}.pkl"), "rb") as f:
        vitamin_samples = pickle.load(f)

    with open(os.path.join(output_dir, f"analytic_meancov{appended}.pkl"), "rb") as f:
        meancov = pickle.load(f)

    an_samples = []
    for m,c in meancov:
        an_mvn = st.multivariate_normal(m.reshape(-1), c)
        an_samples.append(an_mvn.rvs(10000))

    kls_vit_mc = []
    kls_vit_an = []
    kls_an_mc = []

    for mc_samp, vit_samp, an_samp in zip(mc_samps, vitamin_samples, an_samples):
        kls_vit_mc.append(vitamin.train_plots.compute_JS_div(np.array(vit_samp), np.array(mc_samp).T, Nsamp=1000, ntest = 100, nstep = 100))
        kls_vit_an.append(vitamin.train_plots.compute_JS_div(np.array(vit_samp), np.array(an_samp), Nsamp=1000, ntest = 100, nstep = 100))
        kls_an_mc.append(vitamin.train_plots.compute_JS_div(np.array(an_samp), np.array(mc_samp).T, Nsamp=1000, ntest = 100, nstep = 100))

    with open(os.path.join(output_dir,f"kl_vit_mc_divs{appended}.pkl"),"wb") as f:
        pickle.dump(kls_vit_mc,f)

    with open(os.path.join(output_dir,f"kl_vit_an_divs{appended}.pkl"),"wb") as f:
        pickle.dump(kls_vit_an,f)

    with open(os.path.join(output_dir,f"kl_an_mc_divs{appended}.pkl"),"wb") as f:
        pickle.dump(kls_an_mc,f)



    """
    vit_pp = vitamin.train_plots.plot_pp(vitamin_samples, test_dat[1])
    mcmc_pp = vitamin.train_plots.plot_pp(mcmc_samples, test_dat[1])

    with open(os.path.join(output_dir,"vitamin_pp.pkl"),"wb") as f:
        pickle.dump(vit_pp, f)
    with open(os.path.join(output_dir,"mcmc_pp.pkl"),"wb") as f:
        pickle.dump(mcmc_pp, f)
    """
    
