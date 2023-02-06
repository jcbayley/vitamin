import torch
import torch.nn as nn
import numpy as np
import time
from .truncated_gauss import TruncatedNormal

class CVAE(nn.Module):

    def __init__(self, config = None, device="cpu"):
        """
        args
        ----------
        y_dim: int
           zq_sample = self.multi_gauss_dist(mu_q, log_var_q, cat_weight_r).sample( length of input time series
        hidden_dim: int
            size of hidden layers in all encoders
        z_dim: int
            number of variables in latent space
        x_dim: int
            number of parameter to estimat
        fc_layers: list
            [num_neurons_layer_1, num_neurons_layer_2,....]
        conv_layers: list
            [(num_filters, conv_size, dilation, maxpool size), (...), (...)]
        """
        super().__init__()
        default_kwargs = dict(
            z_dim = 4,
            n_modes = 2,
            x_modes = 1,
            hidden_activation = "leakyrelu",
            x_dim = None,
            inf_pars = None,
            bounds = None,
            y_dim = None,
            n_channels = 1,
            split_channels = False,
            shared_network = ['Conv1D(96,64,2)','Conv1D(64,64,2)','Conv1D(64,64,2)','Conv1D(32,32,2)'],
            r1_network = ['Linear(2048)','Linear(1024)','Linear(512)'],
            q_network = ['Linear(2048)','Linear(1024)','Linear(512)'],
            r2_network = ['Linear(2048)','Linear(1024)','Linear(512)'],
            initial_learning_rate = 1e-4,
            logvarmin = False,
            logvarmin_start = -10,
            logvarmin_end = -20)

        for key, val in default_kwargs.items():
            setattr(self, key, val)

        if config is not None:
            self.config = config
            self.z_dim = self.config["model"]["z_dimension"]
            self.n_modes = self.config["model"]["n_modes"]
            self.include_psd = self.config["model"]["include_psd"]
            self.x_modes = 1   # hardcoded for testing
            self.x_dim = len(self.config["inf_pars"])
            self.y_dim = self.config["data"]["sampling_frequency"]*self.config["data"]["duration"]
            self.n_channels = len(self.config["data"]["detectors"])
            self.split_channels = self.config["model"]["split_channels"]
            self.logvarmin = self.config["training"]["logvarmin"]
            self.shared_network = []
            for ln in self.config["model"]["shared_network"]:
                new_ln = ln[7:].strip(")").split(",")
                if not new_ln[0] == "":
                    self.shared_network.append((int(new_ln[0]), int(new_ln[1]), 1, int(new_ln[2])))
            self.r1_network = [int(ln[7:].strip(")")) for ln in self.config["model"]["r1_network"]]
            self.r2_network = [int(ln[7:].strip(")")) for ln in self.config["model"]["r2_network"]]
            self.q_network = [int(ln[7:].strip(")")) for ln in self.config["model"]["q_network"]]
            self.bounds = self.config["bounds"]
            self.inf_pars = self.config["inf_pars"]
            self.hidden_activation = self.config["model"]["hidden_activation"]
            if self.logvarmin:
                self.logvarmin_start = self.config["training"]["logvarmin_start"]

        if self.logvarmin:
            self.minlogvar = self.logvarmin_start
            self.maxlogvar = 4

        # define useful variables of network
        self.device = device
        self.output_dim =self.x_dim

        # convolutional parts
        #self.fc_layers = fc_layers
        #self.fc_layers_decoder = np.array(0.5*np.array(fc_layers[:3])).astype(int)
        #self.conv_layers = conv_layers
        self.num_conv = len(self.shared_network)

        self.trunc_mins = torch.zeros(len(self.inf_pars)).to(self.device)
        self.trunc_maxs = torch.ones(len(self.inf_pars)).to(self.device)
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p=0.0)

        self.small_const = 1e-13
        self.ramp = 1.0
        self.logvarfactor = 1
        
        
        # encoder r1(z|y) 
        self.rencoder_conv, self.rencoder_lin = self.create_network(
            "r",
            self.y_dim, 
            self.z_dim, 
            append_dim=0, 
            fc_layers = self.r1_network, 
            conv_layers = self.shared_network,
            weight = True,
            n_modes = self.n_modes)

        # conv layers not used for next two networks, they both use the same rencoder
        # encoder q(z|x, y) 
        qencoder_conv, self.qencoder_lin = self.create_network(
            "q", 
            self.y_dim, 
            self.z_dim, 
            append_dim=self.x_dim , 
            fc_layers = self.q_network, 
            conv_layers = self.shared_network,
            weight=False,
            n_modes = self.n_modes)

        # decoder r2(x|z, y) 
        decoder_conv, self.decoder_lin = self.create_network(
            "d", 
            self.y_dim, 
            self.x_dim, 
            append_dim=self.z_dim, 
            fc_layers = self.r2_network, 
            conv_layers = self.shared_network, 
            meansize = self.output_dim)
        
    def create_network(self, name, y_dim, output_dim, append_dim=0, mean=True, variance=True, weight = False,fc_layers=[], conv_layers=[], meansize = None, n_modes = 1):
        """ Generate arbritrary network, with convolutional layers or not
        args
        ------
        name: str
            name of network
        y_dim: int
            size of input to network
        output_dim: int
            size of output of network
        append_dim: int (optional)
            number of neurons to append to fully connected layers after convolutional layers
        mean: bool (optional) 
            if True adds extra output layer for means 
        variance : bool (optional)
            if True adds extra layer of outputs for variances
        fc_layers: list
            list of fully connected layers, format: [64,64,...] = [num_neurons_layer1, num_neurons_layer2, .....]
        conv_layers: list
            list of convolutional layers, format:[(8,8,2,1), (8,8,2,1)] = [(num_filt1, conv_size1, max_pool_size1, dilation1), (num_filt2, conv_size2, max_pool_size2, dilation2)] 
        """

        conv_network = nn.Sequential() # initialise networks
        lin_network = nn.Sequential()
        layer_out_sizes = []
        num_fc = len(fc_layers)
        n_channels = self.n_channels
        insize = self.y_dim
        num_conv = 0
        if conv_layers is not None:
            num_conv = len(conv_layers)
            # add convolutional layers
            for i in range(num_conv):
                padding = int(conv_layers[i][1]/2.) # padding half width
                maxpool = nn.MaxPool1d(conv_layers[i][3]) #define max pooling for this layer
                # add convolutional/activation/maxpooling layers
                conv_network.add_module("r_conv{}".format(i), module = nn.Conv1d(n_channels, conv_layers[i][0], conv_layers[i][1], stride = 1,padding=padding, dilation = conv_layers[i][2]))    
                #conv_network.add_module("batch_norm_conv{}".format(i), module = nn.BatchNorm1d(conv_layers[i][0]))
                conv_network.add_module("act_r_conv{}".format(i), module = self.activation)
                conv_network.add_module("pool_r_conv{}".format(i), module = maxpool)
                # define the output size of the layer
                outsize = int(self.conv_out_size(insize, padding, conv_layers[i][2], conv_layers[i][1], 1)/conv_layers[i][3]) # output of one filter
                layer_out_sizes.append((conv_layers[i][0],outsize))
                insize = outsize
                n_channels = conv_layers[i][0]

        # define the input size to fully connected layer
        print(f"LIN OUTPUT SIZE: {layer_out_sizes} ------------------------------------------------------")
        lin_input_size = np.prod(layer_out_sizes[-1]) if num_conv > 0 else self.y_dim
        if append_dim:
            lin_input_size += append_dim
        
        layer_size = int(lin_input_size)
        # hidden layers
        for i in range(num_fc):
            lin_network.add_module("r_lin{}".format(i),module=nn.Linear(layer_size, fc_layers[i]))
            #lin_network.add_module("batch_norm_lin{}".format(i), module = nn.BatchNorm1d(fc_layers[i]))
            lin_network.add_module("r_drop{}".format(i),module=self.drop)
            lin_network.add_module("act_r_lin{}".format(i),module=self.activation)
            layer_size = fc_layers[i]
        # output mean and variance of gaussian with size of latent space

        if mean:
            if meansize is None:
                meansize = output_dim
            if weight:
                setattr(self,"mu_{}".format(name[0]),nn.Linear(layer_size, output_dim*n_modes))
            else:
                setattr(self,"mu_{}".format(name[0]),nn.Linear(layer_size, meansize))
        if variance:
            if weight:
                setattr(self,"log_var_{}".format(name[0]),nn.Linear(layer_size, output_dim*n_modes))
            else:
                setattr(self,"log_var_{}".format(name[0]),nn.Linear(layer_size, output_dim))

        if weight:
            setattr(self,"cat_weight_{}".format(name[0]),nn.Linear(layer_size, n_modes))


        return conv_network, lin_network

    def conv_out_size(self, in_dim, padding, dilation, kernel, stride):
        """ Get output size of a convolutional layer (or one filter from that layer)"""
        return int((in_dim + 2*padding - dilation*(kernel-1)-1)/stride + 1)

    def logvar_act(self, x):
        return (self.maxlogvar - self.minlogvar)*torch.sigmoid(x) + self.minlogvar
        
    def encode_r(self,y):
        """ encoder r1(z|y) , takes in observation y"""
        conv = self.rencoder_conv(torch.reshape(y, (y.size(0), self.n_channels, self.y_dim))) if self.num_conv > 0 else y
        lin_in = torch.flatten(conv,start_dim=1)
        lin = self.rencoder_lin(lin_in)
        z_mu = self.mu_r(lin) # latent means
        z_log_var = self.logvar_act(self.log_var_r(lin)) # latent variances
        z_cat_weight = self.sigmoid(self.cat_weight_r(lin))
        return z_mu.reshape(-1, self.n_modes, self.z_dim), z_log_var.reshape(-1, self.n_modes, self.z_dim), z_cat_weight
    
    def encode_q(self,y,par):
        """ encoder q(z|x, y) , takes in observation y and paramters par (x)"""
        conv = self.rencoder_conv(torch.reshape(y, (y.size(0), self.n_channels, self.y_dim))) if self.num_conv > 0 else y
        lin_in = torch.cat([torch.flatten(conv,start_dim=1), par],1)
        lin = self.qencoder_lin(lin_in)
        z_mu = self.mu_q(lin)  # latent means
        z_log_var = self.logvar_act(self.log_var_q(lin))  # latent vairances
        #z_cat_weight = self.sigmoid(self.cat_weight_q(lin))
        return z_mu, z_log_var
        #return z_mu.reshape(-1, self.n_modes, self.z_dim), z_log_var.reshape(-1, self.n_modes, self.z_dim), z_cat_weight

    
    def decode(self, z, y):
        """ decoder r2(x|z, y) , takes in observation y and latent paramters z"""
        conv = self.rencoder_conv(torch.reshape(y, (y.size(0), self.n_channels, self.y_dim))) if self.num_conv > 0 else y
        lin_in = torch.cat([torch.flatten(conv,start_dim=1),z],1) 
        lin = self.decoder_lin(lin_in)
        par_mu = self.mu_d(lin) # parameter means
        par_mu = self.sigmoid(par_mu)
        par_log_var = self.logvar_act(self.log_var_d(lin))  # parameter variances
        return par_mu, par_log_var

    def gauss_sample(self, mean, log_var, num_batch, dim):
        """ Sample trom a gaussian with given mean and log variance 
        (takes in a number (dim) of means and variances, and samples num_batch times)"""
        std = torch.exp(0.5 * (log_var))
        eps = torch.randn([num_batch, dim]).to(self.device)
        sample = torch.add(torch.mul(eps,std),mean)
        return sample

    def cat_gauss_dist(self, mean, log_var, cat_weight):
        """ KL divergence for Multi dimension categorical gaussian"""
        mix = torch.distributions.Categorical(probs=cat_weight)
        comp = torch.distributions.Independent(torch.distributions.Normal(mean, torch.exp(log_var + (1-self.ramp)*self.logvarfactor)), 1)
        gmm = torch.distributions.MixtureSameFamily(mix, comp)
        return gmm

    def multi_gauss_dist(self, mean, log_var):
        """ KL divergence for Multi dimension categorical gaussian"""
        comp = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag_embed(torch.exp(log_var+(1-self.ramp)*self.logvarfactor)))
        return comp

    def trunc_gauss_sample(self, mu, log_var):
        """Gaussian log-likelihood """
        sigma = torch.sqrt(torch.exp(log_var))
        #dist = TruncatedNormal(mu, sigma, self.trunc_mins - 10 + self.ramp*10, self.trunc_maxs + 10 - self.ramp*10)
        dist = TruncatedNormal(mu, sigma, 0, 1)
        return dist.sample()

    def log_likelihood_trunc_gauss(self,par, mu, log_var):
        """Gaussian log-likelihood """
        sigma = torch.sqrt(torch.exp(log_var))
        #dist = TruncatedNormal(mu, sigma, a=self.trunc_mins - 10 + self.ramp*10, b=self.trunc_maxs + 10 - self.ramp*10)
        dist = TruncatedNormal(mu, sigma, a=0, b=1)
        return dist.log_prob(par)

    
    def KL_gauss(self,mu_r,log_var_r,mu_q,log_var_q):
        """Gaussian KL divergence between two distributions"""
        # user torch.distributions.kl_divergence(qdist, r1dist)
        sigma_q = torch.exp(0.5 * (log_var_q))
        sigma_r = torch.exp(0.5 * (log_var_r))
        t2 = torch.log(sigma_r/sigma_q)
        t3 = (torch.square(mu_q - mu_r) + torch.square(sigma_q))/(2*torch.square(sigma_r))
        # take sum of KL divergences in the latent space
        kl_loss = torch.sum(t2 + t3 - 0.5,dim=1)
        return kl_loss

    def KL_multigauss(self, r1_dist, q_dist, z_samp):
        """ KL divergence for Multi dimension categorical gaussian"""
        selfent_q = -1.0*torch.mean(q_dist.entropy())
        log_r1_q = r1_dist.log_prob(z_samp)   # evaluate the log prob of r1 at the q samples
        cost_KL = selfent_q - torch.mean(log_r1_q)
        return cost_KL

    def forward(self, y, par):
        """forward pass for training"""
        batch_size = y.size(0) # set the batch size
        # encode data into latent space
        mu_r, log_var_r, cat_weight_r = self.encode_r(y) # encode r1(z|y)
        mu_q, log_var_q = self.encode_q(y, par) # encode q(z|x, y)
        
        # sample z from gaussian with mean and variance from q(z|x, y)
        z_sample = self.multi_gauss_dist(mu_q, log_var_q).sample()

        #z_sample = self.gauss_sample(mu_q, log_var_q, batch_size, self.z_dim)
        #z_sample = self.lorentz_sample(mu_q, log_var_q)[0]
        # get the mean and variance in parameter space from decoder
        mu_par, log_var_par = self.decode(z_sample,y) # decode r2(x|z, y)                                                                              
        return mu_par, log_var_par, mu_q, log_var_q, mu_r, log_var_r

    def compute_loss(self, y, par, ramp):

        mu_r, log_var_r, cat_weight_r = self.encode_r(y) # encode r1(z|y)     
        mu_q, log_var_q = self.encode_q(y, par) # encode q(z|x, y)  

        #z_sample = self.gauss_sample(mu_q, log_var_q, mu_q.size(0), mu_q.size(1))

        z_dist = self.multi_gauss_dist(mu_q, log_var_q)
        r1_dist = self.cat_gauss_dist(mu_r, log_var_r, cat_weight_r)

        z_sample = z_dist.sample()

        mu_par, log_var_par = self.decode(z_sample,y) # decode r2(x|z, y)

        #kl_loss = torch.mean(self.KL_gauss(mu_r, log_var_r, mu_q, log_var_q))
        kl_loss = self.KL_multigauss(r1_dist, z_dist, z_sample)

        #print(np.shape(par), np.shape(mu_par), np.shape(log_var_par))
        gauss_loss = self.log_likelihood_trunc_gauss(par, mu_par, log_var_par)
        #print(np.shape(gauss_loss))

        recon_loss = -1.0*torch.mean(torch.sum(gauss_loss, axis = 1))
        
        return recon_loss, kl_loss

        
    
    def draw_samples(self, y, mu_r, log_var_r, cat_weight_r, num_samples, z_dim, x_dim, return_latent = False):
        """ Draw samples from network for testing"""
        #z_sample = self.gauss_sample(mu_r, log_var_r, num_samples, z_dim)
        z_sample = self.cat_gauss_dist(
            mu_r.repeat(num_samples, 1, 1), 
            log_var_r.repeat(num_samples, 1, 1), 
            cat_weight_r.repeat(num_samples, 1)).sample()

        # input the latent space samples into decoder r2(x|z, y)  
        #ys = y.repeat(1,num_samples).view(-1, y.size(0)) # repeat data so same size as the z samples
        ys = y.repeat(num_samples,1,1).view(-1, y.size(0), y.size(1)) # repeat data so same size as the z samples

        # sample parameter space from returned mean and variance 
        mu_par, log_var_par = self.decode(z_sample,ys) # decode r2(x|z, y) from z        
        samp = self.trunc_gauss_sample(mu_par, log_var_par)

        if return_latent:
            return samp.cpu().numpy(), z_sample.cpu().numpy()
        else:
            return samp.cpu().numpy(), None
    
    def test_latent(self, y, par, num_samples):
        """generating samples when testing the network, returns latent samples as well (used during training to get latent samples)"""
        num_data = y.size(0)                                                                                                                                                     
        x_samples = []
        # encode the data into latent space with r1(z,y)          
        mu_r, log_var_r, cat_weight_r = self.encode_r(y) # encode r1(z|y) 
        mu_q, log_var_q = self.encode_q(y,par) # encode q(z|y) 
        # get the latent space samples
        zr_samples = []
        zq_samples = []
        for i in range(num_data):
            # sample from both r and q networks
            #zr_sample = self.gauss_sample(mu_r[i], log_var_r[i], num_samples, self.z_dim)
            #zq_sample = self.gauss_sample(mu_q[i], log_var_q[i], num_samples, self.z_dim)
            zr_sample = self.cat_gauss_dist(
                mu_r[i].repeat(num_samples, 1, 1), 
                log_var_r[i].repeat(num_samples, 1, 1), 
                cat_weight_r[i].repeat(num_samples, 1)).sample()
            zq_sample = self.multi_gauss_dist(
                mu_q[i].repeat(num_samples, 1), 
                log_var_q[i].repeat(num_samples, 1)).sample()
            zr_samples.append(zr_sample.cpu().numpy())
            zq_samples.append(zq_sample.cpu().numpy())
            # input the latent space samples into decoder r2(x|z, y)  
            ys = y[i].repeat(num_samples,1,1).view(-1, y.size(1), y.size(2)) # repeat data so same size as the z samples

            mu_par, log_var_par = self.decode(zr_sample,ys) # decode r2(x|z, y) from z        
            samp = self.trunc_gauss_sample(mu_par, log_var_par)

            # add samples to list    
            x_samples.append(samp.cpu().numpy())
        return np.array(x_samples),np.array(zr_samples),np.array(zq_samples)
    
    def test(self, y, num_samples, transform_func=None, return_latent = False, par = None):
        """generating samples when testing the network 
        args
        --------
        model : pytorch model
            the input model to test
        y: Tensor
            Tensor of all observation data to generate samples from 
        num_samples: int
            number of samples to draw
        transform_func: function (optional)
            function which transforms the parameters into real parameter space
        return_latent: bool
            if true returns the samples in the latent space as well as output
        par: list (optional)
            parameter for each injection, used if returning the latent space samples (return_latent=True)
        """
        num_data = y.size(0)                                                                                                                                                     
        transformed_samples = []
        net_samples = []
        if return_latent:
            z_samples = []
            q_samples = []
        # encode the data into latent space with r1(z,y)          
        mu_r, log_var_r, cat_weight_r = self.encode_r(y) # encode r1(z|y) 
        if return_latent:
            mu_q, log_var_q = self.encode_q(y,par) # encode q(z|y) 
        # get the latent space samples for each input
        for i in range(num_data):
            print(f"index: {i}")
            # generate initial samples
            t_net_samples, t_znet_samples = self.draw_samples(
                y[i], 
                mu_r[i], 
                log_var_r[i], 
                cat_weight_r[i],
                num_samples, 
                self.z_dim, 
                self.x_dim, 
                return_latent = return_latent
                )

            if return_latent:
                q_samples.append(
                    self.multi_gauss_dist(
                        mu_q[i].repeat(num_samples, 1, 1), 
                        log_var_q[i].repeat(num_samples, 1, 1)).sample()
                )
                    #self.gauss_sample(
                    #    mu_q[i], 
                    #    log_var_q[i], 
                    #    num_samples, 
                    #   self.z_dim
                    #   ).cpu().numpy()
                    

            # if nans in samples then keep drawing samples until there are no Nans (for whan samples are outside prior)
            if np.any(np.isnan(t_net_samples)):
                num_nans = np.inf
                stime = time.time()
                while num_nans > 0:
                    nan_locations = np.where(np.any(np.isnan(t_net_samples), axis=1))
                    num_nans = len(nan_locations[0])
                    if num_nans == 0: 
                        break
                    temp_new_net_samp, temp_new_z_sample = self.draw_samples(
                        y[i], 
                        mu_r[i], 
                        log_var_r[i], 
                        cat_weight_r[i],
                        num_nans, 
                        self.z_dim, 
                        self.x_dim, 
                        return_latent = return_latent
                        )

                    t_net_samples[nan_locations] = temp_new_net_samp
                    if return_latent:
                        t_znet_samples[nan_locations] = temp_new_z_sample

                    etime = time.time()
                    # if it still nans after 1 min cancel
                    if etime - stime > 0.5*60:
                        print("Failed to find samples within 3 mins")
                        num_nans = 0
                        break

            net_samples.append(t_net_samples)
            if return_latent:
                z_samples.append(t_znet_samples)
        
        if return_latent:
            return np.array(net_samples), np.array(z_samples), np.array(q_samples)
        else:
            return np.array(net_samples)
    
    def test_old(self, y, num_samples, transform_func=None, return_latent = False, par = None):
        """generating samples when testing the network 
        args
        --------
        model : pytorch model
            the input model to test
        y: Tensor
            Tensor of all observation data to generate samples from 
        num_samples: int
            number of samples to draw
        transform_func: function (optional)
            function which transforms the parameters into real parameter space
        return_latent: bool
            if true returns the samples in the latent space as well as output
        par: list (optional)
            parameter for each injection, used if returning the latent space samples (return_latent=True)
        """
        num_data = y.size(0)                                                                                                                                                     
        transformed_samples = []
        net_samples = []
        if return_latent:
            z_samples = []
            q_samples = []
        # encode the data into latent space with r1(z,y)          
        mu_r, log_var_r = self.encode_r(y) # encode r1(z|y) 
        if return_latent:
            mu_q, log_var_q = self.encode_q(y,par) # encode q(z|y) 
        # get the latent space samples for each input
        for i in range(num_data):
            print("index: {}".format(i))
            # generate initial samples
            t_net_samples, t_znet_samples = self.draw_samples(y[i], mu_r[i], log_var_r[i], num_samples, self.z_dim, self.x_dim, return_latent = return_latent)
            if return_latent:
                q_samples.append(self.gauss_sample(mu_q[i], log_var_q[i], num_samples, self.z_dim).cpu().numpy())
            if transform_func is None:
                # if nans in samples then keep drawing samples until there are no Nans (for whan samples are outside prior)
                if np.any(np.isnan(t_net_samples)):
                    num_nans = np.inf
                    stime = time.time()
                    while num_nans > 0:
                        nan_locations = np.where(np.any(np.isnan(t_net_samples), axis=1))
                        num_nans = len(nan_locations[0])
                        if num_nans == 0: break
                        temp_new_net_samp, temp_new_z_sample = self.draw_samples(y[i], mu_r[i], log_var_r[i], num_nans, self.z_dim, self.x_dim, return_latent = return_latent)
                        t_net_samples[nan_locations] = temp_new_net_samp
                        if return_latent:
                            t_znet_samples[nan_locations] = temp_new_z_sample

                        etime = time.time()
                        # if it still nans after 1 min cancel
                        if etime - stime > 0.5*60:
                            print("Failed to find samples within 3 mins")
                            num_nans = 0
                            break

                transformed_samples.append(t_net_samples)
                net_samples.append(t_net_samples)
                if return_latent:
                    z_samples.append(t_znet_samples)
                
            else:
                # transform all samples to new parameter space
                new_samples = transform_func(self.config, t_net_samples)
                # if transformed samples are outside prior (nan) then redraw nans until all real values
                if np.any(np.isnan(new_samples)):
                    stime = time.time()
                    num_nans = np.inf
                    while num_nans > 0:
                        nan_locations = np.where(np.any(np.isnan(new_samples), axis=1))
                        num_nans = len(nan_locations[0])
                        if num_nans == 0: break
                        #redraw samples at nan locations
                        temp_new_net_samples, temp_new_z_samples = self.draw_samples(y[i], mu_r[i], log_var_r[i], num_nans, self.z_dim, self.x_dim, return_latent = return_latent)
                        transformed_newsamp = transform_func(temp_new_net_samples, i)
                        new_samples[nan_locations] = transformed_newsamp
                        t_net_samples[nan_locations] = temp_new_net_samples
                        if return_latent:
                            t_znet_samples[nan_locations] = temp_new_z_samples
                        etime = time.time()
                        # if it still nans after 1 min cancel
                        if etime - stime > 0.5*60:
                            print("Failed to find samples within 30s")
                            num_nans = 0
                            break
                    

                transformed_samples.append(new_samples)
                net_samples.append(t_net_samples)

                if return_latent:
                    z_samples.append(t_znet_samples)
        
        if return_latent:
            return np.array(transformed_samples), np.array(net_samples), np.array(z_samples), np.array(q_samples)
        else:
            return np.array(transformed_samples), np.array(net_samples)

