from signal import SIGRTMAX
from tarfile import LENGTH_LINK
import vitamin
import numpy as np
import tensorflow as tf
import os
import pickle
import pymc as pm

output_dir = "./outputs/"

def data_model(x, ms):
    #define a simple straight line model
    y = 0
    for i, m in enumerate(ms):
        y += ms[i]*(x**i)
    return y

def get_dataset(num_data, length = 100, sigma=0.1, num_params = 2):
    """
    Generate the dataset with random gradient and offset parameters
    """
    xdat = np.linspace(0,1,length)
    y = []
    x = []
    for i in range(num_data):
        ms = np.random.uniform(size = num_params)
        # make sure data has 3 dimensions (number_examples, number_datapoints, number_channels)
        y.append(np.expand_dims(data_model(xdat,ms) + np.random.normal(0,sigma,len(xdat)),-1))
        x.append(ms)
    return np.array(x),np.array(y)

num_params = 5
length = 100
# generate the training dataset and the validation dataset
train_dat = get_dataset(500000, num_params=num_params, length=length)
val_dat = get_dataset(1000, num_params=num_params, length=length)

# a few conditions to choose which samplers to run
generate_test = False
load_test = True
train_network = False
test_network = True
run_mcmc_sampler = True
make_test_plots = True

if generate_test:   
    #generate test_data
    test_dat = get_dataset(500, num_params=num_params, length=length)
    with open(os.path.join(output_dir, "test_data.pkl"), "wb") as f:
        pickle.dump(test_dat, f)

elif load_test:
    # load test data if already generated
    with open(os.path.join(output_dir, "test_data.pkl"), "rb") as f:
        test_dat = pickle.load(f)


#Set up the CVAE parmaeters and bounds
#parameters to infer and the output distributions of the CVAE (default is Truncated Normal)
inf_pars = {f"p{i}":"TruncatedNormal" for i in range(num_params)}
# the bounds for each of the parameters, used to rescale parameters internally
bounds = {f"p{i}_{bnd}":val for i in range(num_params) for bnd,val in zip(["min","max"], [0,1])}
# layers shared across the three networks (all available layers are defined in Docs)
# alternativly can define own models
shared_network = ['Conv1D(16,16,1)','Conv1D(16,16,1)','Conv1D(16,8,2)','Conv1D(16,8,2)','Flatten()']
# three individual networks designs to come after the shared network
r1_network = ['Linear(128)','Linear(64)', 'Linear(32)']
r2_network = ['Linear(128)','Linear(64)', 'Linear(32)']
q_network = ['Linear(128)','Linear(64)', 'Linear(32)']

# initialise the model 
model = vitamin.vitamin_model.CVAE(z_dim=4, # latent space size
                                n_modes = 2, # number of modes in the latent space
                                x_dim = num_params,  # number of parameters to infer
                                inf_pars=inf_pars, # inference parameters
                                bounds=bounds, # inference parameters bounds
                                y_dim=LENGTH_LINK, # number of datapoints
                                n_channels=1, # number of input channels
                                shared_network=shared_network,
                                r1_network=r1_network,
                                r2_network=r2_network,
                                q_network=q_network)


# define the optimiser
optimizer = tf.keras.optimizers.Adam(2e-4)

# compile the model using optimiser (if using CPU you can run eagerly)
#model.compile(optimizer=optimizer,run_eagerly = True, loss=model.compute_loss)
model.compile(optimizer=optimizer, loss=model.compute_loss)

if train_network == True:
    # define some callbacks to record the loss and anneal the kl divergence loss
    # annealing can be important to avoid local minima
    loss_call = vitamin.callbacks.PlotCallback(None, 1000, save_data=False, start_epoch = 0)
    ann_call = vitamin.callbacks.AnnealCallback(50,10)

    # fit the model
    model.fit(train_dat[1], train_dat[0], validation_data=(val_dat[1],val_dat[0]), epochs = 2000, batch_size = 128, callbacks = [loss_call, ann_call])

    # save outputs
    with open(os.path.join(output_dir, "loss.pkl"), "wb") as f:
        pickle.dump(loss_call.all_losses, f)

    # save the weights of the model
    model.save_weights(os.path.join(output_dir,"model"))

if test_network:
    # load the weights of pretrained model
    if train_network == False:
        model.load_weights(os.path.join(output_dir,"model"))

    # generate some samples (Run each sample through individually with shape (1, datapoints, channels))
    samples = []
    for td in range(len(test_dat[1])):
        samples.append(model.gen_samples(test_dat[1][td:td+1], nsamples=10000))

    with open(os.path.join(output_dir, "samples.pkl"), "wb") as f:
        pickle.dump(samples, f)


if run_mcmc_sampler:
    # run mcmc on the same test data using pymc
    mcmc_samples = []
    # initialise the x data
    xdat = np.linspace(0,1,100)
    #loop over all of the test data
    for td in range(len(test_dat[1])):
        # setup pymc model
        with pm.Model() as gauss_model:
            # uniform priors on each of the parameters as in the training data
            priors = [pm.Uniform(f"p{i}",0,1) for i in range(num_params)]
            # Gaussian likelihood with fixed sigma as in training
            lik = pm.Normal("lik", mu=data_model(xdat,priors), sigma=0.1, observed = np.squeeze(test_dat[1][td]))

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

    with open(os.path.join(output_dir, "samples.pkl"), "rb") as f:
        vitamin_samples = pickle.load(f)

    kls = []
    for mc_samp, vit_samp in zip(mc_samps, vitamin_samples):
        kls.append(vitamin.train_plots.compute_KL(np.array(vit_samp), np.array(mc_samp).T, Nsamp=1000, ntest = 100, nstep = 100))

    with open(os.path.join(output_dir,"kl_divs.pkl"),"wb") as f:
        pickle.dump(kls,f)

    """
    vit_pp = vitamin.train_plots.plot_pp(vitamin_samples, test_dat[1])
    mcmc_pp = vitamin.train_plots.plot_pp(mcmc_samples, test_dat[1])

    with open(os.path.join(output_dir,"vitamin_pp.pkl"),"wb") as f:
        pickle.dump(vit_pp, f)
    with open(os.path.join(output_dir,"mcmc_pp.pkl"),"wb") as f:
        pickle.dump(mcmc_pp, f)
    """
    