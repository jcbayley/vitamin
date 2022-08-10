from signal import SIGRTMAX
import vitamin
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import corner
import os
import pickle
import pymc as pm

output_dir = "./"

def data_model(x, m, c):
    #define a simple straight line model
    return m*x + c

def get_dataset(num_data, length = 100, sigma=0.1):
    """
    Generate the dataset with random gradient and offset parameters
    """
    xdat = np.linspace(0,1,length)
    y = []
    x = []
    for i in range(num_data):
        m = np.random.uniform()
        c = np.random.uniform()
        # make sure data has 3 dimensions (number_examples, number_datapoints, number_channels)
        y.append(np.expand_dims(data_model(xdat,m,c) + np.random.normal(0,sigma,len(xdat)),-1))
        x.append([m,c])
    return np.array(x),np.array(y)

# generate the training dataset and the validation dataset
train_dat = get_dataset(100000)
val_dat = get_dataset(1000)

# a few conditions to choose which samplers to run
generate_test = False
load_test = True
train_network = False
test_network = True
run_mcmc_sampler = False

if generate_test:   
    #generate test_data
    test_dat = get_dataset(10)
    with open(os.path.join(output_dir, "test_data.pkl"), "wb") as f:
        pickle.dump(test_dat, f)

elif load_test:
    # load test data if already generated
    with open(os.path.join(output_dir, "test_data.pkl"), "rb") as f:
        test_dat = pickle.load(f)


#Set up the CVAE parmaeters and bounds
#parameters to infer and the output distributions of the CVAE (default is Truncated Normal)
inf_pars = {"m":"TruncatedNormal","c":"TruncatedNormal"}
# the bounds for each of the parameters, used to rescale parameters internally
bounds = {"m_min":0,"m_max":1,"c_min":0,"c_max":1}
# layers shared across the three networks (all available layers are defined in Docs)
# alternativly can define own models
shared_network = ['Conv1D(16,16,1)','Conv1D(16,8,2)','Conv1D(16,8,2)','Flatten()']
# three individual networks designs to come after the shared network
r1_network = ['Linear(128)','Linear(32)']
r2_network = ['Linear(128)','Linear(32)']
q_network = ['Linear(128)','Linear(32)']

# alternatively one can input their own network designs for each of these three networks

#shared_network = tf.keras.Sequential([tf.keras.layers.Conv1D(16,16, name="conv1", activation="relu"), 
#                                      tf.keras.layers.Conv1D(16,16, name="conv2", activation="relu"),
#                                      tf.keras.layers.Flatten()])

#r1_network = tf.keras.Sequential([tf.keras.layers.Dense(32, name="r1lin1",activation="relu"),
#                                  tf.keras.layers.Dense(16, name="r1lin2",activation="relu")])

#q_network = tf.keras.Sequential([tf.keras.layers.Dense(32, name="qlin1",activation="relu"),
#                                  tf.keras.layers.Dense(16, name="qlin2",activation="relu")])

#r2_network = tf.keras.Sequential([tf.keras.layers.Dense(32, name="r2lin1",activation="relu"),
#                                  tf.keras.layers.Dense(16, name="r2lin2",activation="relu")])

# initialise the model 
model = vitamin.vitamin_model.CVAE(z_dim=4, # latent space size
                                n_modes = 2, # number of modes in the latent space
                                x_dim = 2,  # number of parameters to infer
                                inf_pars=inf_pars, # inference parameters
                                bounds=bounds, # inference parameters bounds
                                y_dim=100, # number of datapoints
                                n_channels=1, # number of input channels
                                shared_network=shared_network,
                                r1_network=r1_network,
                                r2_network=r2_network,
                                q_network=q_network)


# define the optimiser
optimizer = tf.keras.optimizers.Adam(1e-4)

# compile the model using optimiser (if using CPU you can run eagerly)
#model.compile(optimizer=optimizer,run_eagerly = False, loss=model.compute_loss)
model.compile(optimizer=optimizer, loss=model.compute_loss)

if train_network == True:
    # define some callbacks to record the loss and anneal the kl divergence loss
    # annealing can be important to avoid local minima
    loss_call = vitamin.callbacks.PlotCallback(None, 1000, save_data=False, start_epoch = 0)
    ann_call = vitamin.callbacks.AnnealCallback(100,50)

    # fit the model
    model.fit(train_dat[1], train_dat[0], validation_data=(val_dat[1],val_dat[0]), epochs = 1000, batch_size = 128, callbacks = [loss_call, ann_call])

    # save outputs
    with open(os.path.join(output_dir, "loss.pkl"), "wb") as f:
        pickle.dump(loss_call.all_losses, f)

    fig.savefig(os.path.join(output_dir,"lossplot.png"))

    # save the weights of the model
    model.save_weights(os.path.join(output_dir,"model"))

if test_network:
    # load the weights of pretrained model
    if train_network == False:
        model.load_weights(os.path.join(output_dir,"model"))

    # generate some samples (Run each sample through individually with shape (1, datapoints, channels))
    samples = []
    for td in range(len(test_dat[1])):
        samples.append(model.gen_samples(test_dat[1][td:td+1], nsamples=4000))

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
            pr_m = pm.Uniform("m",0,1)
            pr_c = pm.Uniform("c",0,1)

            # Gaussian likelihood with fixed sigma as in training
            lik = pm.Normal("lik", mu=data_model(xdat,pr_m,pr_c), sigma=0.1, observed = np.squeeze(test_dat[1][td]))

            # setup sampler and generate samples
            step = pm.Slice()
            mcmc_samples.append(pm.sample(1000, step=step))

    with open(os.path.join(output_dir,"mcmc_samples.pkl"),"wb") as f:
        pickle.dump(mcmc_samples, f)

    