import torch
import torch.nn as nn
import numpy as np
import time
from .new_distributions.truncated_gauss import TruncatedNormalDist as TruncatedNormal
from .new_distributions.mixture_same_family import ReparametrizedMixtureSameFamily
from .group_inference_parameters import group_outputs

class CVAE(nn.Module):

    def __init__(self, config = None, device="cpu", **kwargs):
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
            dropout = 0.0,
            initial_learning_rate = 1e-4,
            logvarmin = False,
            include_parameter_network = False,
            separate_channels = False,
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
                new_ln = ln.split("(")[1].strip(")").split(",")
                print(new_ln)
                if not new_ln[0] == "":
                    self.shared_network.append((int(new_ln[0]), int(new_ln[1]), int(new_ln[2]), int(new_ln[3])))

            self.r1_network = [int(ln[7:].strip(")")) for ln in self.config["model"]["r1_network"]]
            self.r2_network = [int(ln[7:].strip(")")) for ln in self.config["model"]["r2_network"]]
            self.q_network = [int(ln[7:].strip(")")) for ln in self.config["model"]["q_network"]]
            self.dropout = int(self.config["model"]["dropout"])
            self.bounds = self.config["bounds"]
            self.inf_pars = self.config["inf_pars"]
            self.include_parameter_network = self.config["model"]["include_parameter_network"]
            self.separate_channels = self.config["model"]["separate_channels"]
            self.hidden_activation = self.config["model"]["hidden_activation"]
            if self.logvarmin:
                self.logvarmin_start = self.config["training"]["logvarmin_start"]


        for key, val in kwargs.items():
            if key in default_kwargs.keys():
                if key == "shared_network":
                    new_val = []
                    for ln in val:
                        new_ln = ln[7:].strip(")").split(",")
                        if not new_ln[0] == "":
                            new_val.append((int(new_ln[0]), int(new_ln[1]), 1, int(new_ln[2])))
                    val = new_val
                if key in ["r1_network", "r2_network", "q_network"]:
                    val = [int(ln[7:].strip(")")) for ln in val]

                setattr(self, key, val)
            else:
                raise Exception("Key {} not valid, please choose from {}".format(key, list(default_kwargs.keys())))

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
        print("numconv: ", self.num_conv)

        self.trunc_mins = torch.zeros(len(self.inf_pars)).to(self.device)
        self.trunc_maxs = torch.ones(len(self.inf_pars)).to(self.device)
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.drop = nn.Dropout(p=self.dropout)

        self.small_const = 1e-13
        self.ramp = 1.0
        self.r2_ramp = 1.0
        self.logvarfactor = 1
        self.latent_minlogvar = -7
        self.latent_maxlogvar = 1

        self.grouped_params, self.new_params_order, self.reverse_params_order = group_outputs(self.inf_pars, self.bounds)

        # encoder r1(z|y) 
        if self.separate_channels:
            t_channels = 1
        else:
            t_channels = self.n_channels
        self.shared_conv, self.rencoder_lin, shared_out_sizes = self.create_network(
            "r",
            self.y_dim, 
            self.z_dim, 
            append_dim=0, 
            fc_layers = self.r1_network, 
            conv_layers = self.shared_network,
            weight = True,
            n_modes = self.n_modes,
            n_channels=t_channels)

        # conv layers not used for next two networks, they both use the same rencoder
        # encoder q(z|x, y) 
        self.qencoder_lin, qlinout = self.create_network(
            "q", 
            self.y_dim, 
            self.z_dim, 
            append_dim=self.x_dim, 
            fc_layers = self.q_network, 
            conv_layers = None,
            weight=False,
            layer_out_sizes = shared_out_sizes)

        # decoder r2(x|z, y) 
        self.decoder_lin, dlinout = self.create_network(
            "d", 
            self.y_dim, 
            self.x_dim, 
            append_dim=self.z_dim, 
            fc_layers = self.r2_network, 
            conv_layers = None, 
            mean = False,
            variance = False,
            layer_out_sizes = shared_out_sizes)

        if self.include_parameter_network:
            print("including parameter network")
            self.par_encode_network = nn.Sequential(
                nn.Linear(self.x_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU(),
                nn.Linear(64, self.x_dim),
                nn.Sigmoid(),
            )

            self.par_decode_network = nn.Sequential(
                nn.Linear(self.x_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU(),
                nn.Linear(64, self.x_dim),
                nn.Sigmoid(),
            )


        outputs = []
        self.group_par_sizes = []
        self.group_output_sizes = []
        for name, group in self.grouped_params.items():
            means, logvars = group.get_networks(dlinout)
            setattr(self, f"{name}_mean", means)
            setattr(self, f"{name}_logvars", logvars)
            self.group_par_sizes.append(group.num_pars)
            self.group_output_sizes.extend(group.num_outputs)
            #setattr(self, "{}_loss_metric".format(name), torch.mean('{}_loss'.format(name), dtype=tf.float32))
 
    def par_network_scale(self, par):
        scale_pars = self.par_encode_network(par)#*0.9 + 0.1
        #return torch.mul(par,scale_pars)
        return scale_pars

    def par_network_unscale(self, par):
        scale_pars = self.par_decode_network(par)#*0.9 + 0.1
        #return torch.divide(par,scale_pars)
        return scale_pars
        
    def create_network(self, name, y_dim, output_dim, append_dim=0, mean=True, variance=True, weight = False,fc_layers=[], conv_layers=[], meansize = None, n_modes = 1, layer_out_sizes=[], n_channels=3):
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
         # initialise networks
        lin_network = nn.Sequential()
        num_fc = len(fc_layers)
        insize = self.y_dim
        num_conv = 0
        if conv_layers is not None:
            layer_out_sizes = []
            conv_network = nn.Sequential()
            num_conv = len(conv_layers)
            # add convolutional layers
            for i in range(num_conv):
                padding = int(int(conv_layers[i][1])/2.) # padding half width
                maxpool = nn.MaxPool1d(conv_layers[i][3]) #define max pooling for this layer
                # add convolutional/activation/maxpooling layers
                conv_network.add_module("r_conv{}".format(i), module = nn.Conv1d(n_channels, conv_layers[i][0], conv_layers[i][1], stride = 1,padding=padding, dilation = conv_layers[i][2]))    
                #conv_network.add_module("batch_norm_conv{}".format(i), module = nn.BatchNorm1d(conv_layers[i][0]))
                conv_network.add_module("act_r_conv{}".format(i), module = nn.ReLU())
                conv_network.add_module("pool_r_conv{}".format(i), module = maxpool)
                # define the output size of the layer
                outsize = int(self.conv_out_size(insize, padding, conv_layers[i][2], conv_layers[i][1], 1)/conv_layers[i][3]) # output of one filter
                layer_out_sizes.append((conv_layers[i][0],outsize))
                insize = outsize
                n_channels = conv_layers[i][0]

        # define the input size to fully connected layer
        lin_input_size = np.prod(layer_out_sizes[-1]) if len(layer_out_sizes) > 0 else self.y_dim
        if append_dim:
            lin_input_size += append_dim
    
        
        layer_size = int(lin_input_size)
        # hidden layers
        for i in range(num_fc):
            lin_network.add_module("r_lin{}".format(i),module=nn.Linear(layer_size, fc_layers[i]))
            #lin_network.add_module("batch_norm_lin{}".format(i), module = nn.BatchNorm1d(fc_layers[i]))
            lin_network.add_module("r_drop{}".format(i),module=self.drop)
            lin_network.add_module("act_r_lin{}".format(i),module=nn.ReLU())
            layer_size = fc_layers[i]
        # output mean and variance of gaussian with size of latent space

        lin_out_size = layer_size

        if mean:
            if meansize is None:
                meansize = output_dim
            if weight:
                setattr(self,"mu_{}".format(name[0]),nn.Linear(layer_size, meansize*n_modes))
                #torch.nn.init.xavier_uniform_(getattr(self,"mu_{}".format(name[0])).weight)
                getattr(self,"mu_{}".format(name[0])).bias.data.uniform_(-1.0, 1.0)
            else:
                setattr(self,"mu_{}".format(name[0]),nn.Linear(layer_size, meansize))
                #torch.nn.init.xavier_uniform_(getattr(self,"mu_{}".format(name[0])).weight)
                getattr(self,"mu_{}".format(name[0])).bias.data.uniform_(-1.0, 1.0)
        if variance:
            if weight:
                setattr(self,"log_var_{}".format(name[0]),nn.Linear(layer_size, output_dim*n_modes))
                #getattr(self,"log_var_{}".format(name[0])).weight.data.fill_(1.0)
                getattr(self,"log_var_{}".format(name[0])).bias.data.fill_(0.0)
            else:
                setattr(self,"log_var_{}".format(name[0]),nn.Linear(layer_size, output_dim))
                #getattr(self,"log_var_{}".format(name[0])).weight.data.fill_(1.0)
                getattr(self,"log_var_{}".format(name[0])).bias.data.fill_(0.0)

        if weight:
            setattr(self,"cat_weight_{}".format(name[0]),nn.Linear(layer_size, n_modes))

        if conv_layers is not None:
            return conv_network, lin_network, layer_out_sizes
        else:
            return lin_network, lin_out_size

    def conv_out_size(self, in_dim, padding, dilation, kernel, stride):
        """ Get output size of a convolutional layer (or one filter from that layer)"""
        return int((in_dim + 2*padding - dilation*(kernel-1)-1)/stride + 1)

    def logvar_act(self, x):
        return (self.maxlogvar - self.minlogvar)*torch.sigmoid(x) + self.minlogvar

    def latent_logvar_act(self, x):
        return (self.latent_maxlogvar - self.latent_minlogvar)*torch.sigmoid(x) + self.latent_minlogvar
        
    def shared_encode(self, y):
        #conv = self.shared_conv(torch.reshape(y, (y.size(0), self.n_channels, self.y_dim))) if self.num_conv > 0 else y
        if self.separate_channels:
            for i in range(self.n_channels):
                ch_out = self.shared_conv(torch.reshape(y[:,i:i+1], (y.size(0), 1, self.y_dim))) if self.num_conv > 0 else y[:,i:i+1]
                if i==0:
                    outputs = ch_out
                else:
                    torch.cat([outputs, ch_out], dim = 1)
        else:
            outputs = self.shared_conv(torch.reshape(y, (y.size(0), self.n_channels, self.y_dim))) if self.num_conv > 0 else y
        
        return torch.flatten(outputs,start_dim=1)
    
    def encode_r(self,y):
        """ encoder r1(z|y) , takes in observation y"""
        lin = self.rencoder_lin(y)
        z_mu = self.mu_r(lin) # latent means
        z_log_var = self.latent_logvar_act(self.log_var_r(lin)) + (1-self.ramp)*4 # latent variances
        z_cat_weight = self.softmax(self.cat_weight_r(lin))
        return z_mu.reshape(-1, self.n_modes, self.z_dim), z_log_var.reshape(-1, self.n_modes, self.z_dim), z_cat_weight
    
    def encode_q(self,y,par):
        """ encoder q(z|x, y) , takes in observation y and paramters par (x)"""
        lin_in = torch.cat([y, par],1)
        #lin_in = par
        lin = self.qencoder_lin(lin_in)
        z_mu = self.mu_q(lin)  # latent means
        z_log_var = self.latent_logvar_act(self.log_var_q(lin)) + (1-self.ramp)*4 # latent vairances
        #z_cat_weight = self.sigmoid(self.cat_weight_q(lin))
        return z_mu, z_log_var
        #return z_mu.reshape(-1, self.n_modes, self.z_dim), z_log_var.reshape(-1, self.n_modes, self.z_dim), z_cat_weight

    
    def decode(self, z, y):
        """ decoder r2(x|z, y) , takes in observation y and latent paramters z"""
        lin_in = torch.cat([y*self.r2_ramp,z],1) 
        lin = self.decoder_lin(lin_in)
        outputs = []
        for name, group in self.grouped_params.items():
            outputs.append(getattr(self, f"{name}_mean")(lin))
            outputs.append(self.logvar_act(getattr(self, f"{name}_logvars")(lin)) + (1-self.ramp)*4)
        return torch.split(torch.cat(outputs, dim = 1), split_size_or_sections=self.group_output_sizes, dim=1)
    
    def cat_gauss_dist(self, mean, log_var, cat_weight):
        """ KL divergence for Multi dimension categorical gaussian"""
        mix = torch.distributions.Categorical(probs=cat_weight)
        #comp = torch.distributions.Independent(torch.distributions.Normal(mean, torch.exp(log_var + (1-self.ramp)*self.logvarfactor)), 1)
        comp = torch.distributions.Independent(torch.distributions.Normal(mean, torch.exp(log_var)), 1)
        gmm = ReparametrizedMixtureSameFamily(mix, comp)
        return gmm

    def multi_gauss_dist(self, mean, log_var):
        """ KL divergence for Multi dimension categorical gaussian"""
        #comp = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag_embed(torch.exp(log_var+(1-self.ramp)*self.logvarfactor)))
        comp = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag_embed(torch.exp(log_var)))
        return comp

    def gauss_dist(self, mu, log_var):
        sigma = torch.sqrt(torch.exp(log_var))
        dist = torch.distributions.Normal(mu, sigma)
        return dist

    def trunc_gauss_sample(self, mu, log_var):
        """Gaussian log-likelihood """
        sigma = torch.sqrt(torch.exp(log_var))
        #dist = TruncatedNormal(mu, sigma, self.trunc_mins - 10 + self.ramp*10, self.trunc_maxs + 10 - self.ramp*10)
        dist = TruncatedNormal(mu, sigma, 0, 1)
        return dist.rsample()

    def log_likelihood_trunc_gauss(self,par, mu, log_var):
        """Gaussian log-likelihood """
        sigma = torch.sqrt(torch.exp(log_var))
        #dist = TruncatedNormal(mu, sigma, a=self.trunc_mins - 10 + self.ramp*10, b=self.trunc_maxs + 10 - self.ramp*10)
        dist = TruncatedNormal(mu, sigma, a=0, b=1)
        return dist.log_prob(par)

    def log_likelihood_gauss(self, par, mu, log_var):
        """ Gausisan log likelihood"""
        sigma = torch.sqrt(torch.exp(log_var))
        dist = torch.distributions.Normal(mu, sigma)
        return dist.log_prob(par)

    def gauss_sample(self, mu, log_var):
        """ Gausisan log likelihood"""
        sigma = torch.sqrt(torch.exp(log_var))
        dist = torch.distributions.Normal(mu, sigma)
        return dist.rsample()
    
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

    def KL_multigauss(self, q_dist, r1_dist, n_samp):
        """ KL divergence for Multi dimension categorical gaussian
        args
        --------
        r1_dist: torch.distribution
        q_dist: torch.distribution
        z_samp: torch.Tensor
            Tensor of samples of shape [batch_size, z_dim]
        returns
        --------
        mean KL over batch
        """
        #selfent_q = -1.0*q_dist.entropy() # shape [batch_size]
        #log_r1_q = r1_dist.log_prob(z_samp)   # evaluate the log prob of r1 at the q samples
        #cost_KL = selfent_q - tf.reduce_mean(log_r1_q, 1)
        selfent_q = -1.0*q_dist.entropy()
        log_r1_q = r1_dist.log_prob(q_dist.rsample((n_samp, )))   # evaluate the log prob of r1 at the q samples
        cost_KL = selfent_q - torch.mean(log_r1_q)
        return cost_KL


    def forward(self, y, par):
        """forward pass for training"""
        batch_size = y.size(0) # set the batch size
        # encode data into latent space
        if self.include_parameter_network:
            par_scale = self.par_network_scale(par)
            par_rescale = self.par_network_unscale(par_scale)
        else:
            par_scale = par
            par_loss = 0

        shared_y = self.shared_encode(y)
        mu_r, log_var_r, cat_weight_r = self.encode_r(shared_y) # encode r1(z|y)
        mu_q, log_var_q = self.encode_q(shared_y, par) # encode q(z|x, y)
        
        # sample z from gaussian with mean and variance from q(z|x, y)
        z_sample = self.multi_gauss_dist(mu_q, log_var_q).rsample()
        r1_sample = self.cat_gauss_dist(mu_r, log_var_r, cat_weight_r).rsample()

        #z_sample = self.gauss_sample(mu_q, log_var_q, batch_size, self.z_dim)
        #z_sample = self.lorentz_sample(mu_q, log_var_q)[0]
        # get the mean and variance in parameter space from decoder
        xpars = self.decode(z_sample,shared_y) # decode r2(x|z, y)                                                                              
        return xpars, mu_q, log_var_q, mu_r, log_var_r

    def compute_loss(self, y, par, ramp, r2_ramp=1.0):
        """ 
        Compute the cost over a batch of input data (y) and its associated parameters (par)
        args
        ---------
        y: Tensor
            Input data
        par: Tensor
            Parameters associated with y data

        Returns
        ---------
        recon_loss: Tensor
            Reconstruction loss of the r2 distribution
        kl_loss: Tensor
            KL divergence between the q and r1 distributions
        """
        if y[0].shape != (self.n_channels, self.y_dim):
            raise Exception(f"input wrong shape: {y.shape} nut should be (N, {self.n_channels}, {self.y_dim})")
            
        self.ramp = ramp
        self.r2_ramp = r2_ramp

        if self.include_parameter_network:
            par_scale = self.par_network_scale(par)
            par_rescale = self.par_network_unscale(par_scale)
            par_loss = torch.mean((par_scale - par)**2)
        else:
            par_scale = par
            par_loss = 0

        # reorder for distribution grouping
        par_scale = par_scale[:,self.new_params_order]

        shared_y = self.shared_encode(y)

        mu_r, log_var_r, cat_weight_r = self.encode_r(shared_y) # encode r1(z|y)     
        mu_q, log_var_q = self.encode_q(shared_y, par_scale) # encode q(z|x, y)  

        #z_sample = self.gauss_sample(mu_q, log_var_q, mu_q.size(0), mu_q.size(1))

        #z_dist = self.multi_gauss_dist(mu_q, log_var_q)
        #r1_dist = self.cat_gauss_dist(mu_r, log_var_r, cat_weight_r)

        q_dist  = self.multi_gauss_dist(mu_q, log_var_q)
        r1_dist = self.cat_gauss_dist(mu_r, log_var_r, cat_weight_r)

        z_sample = q_dist.rsample()

        #mu_par, log_var_par = self.decode(z_sample,y) # decode r2(x|z, y)

        decoded_outputs = self.decode(z = z_sample, y=shared_y)
        par_grouped = torch.split(par_scale, split_size_or_sections=self.group_par_sizes, dim=1)
        cost_recon = 0 
        ind = 0
        indx = 0
        outs = {}
        for name, group in self.grouped_params.items():
            dist = group.get_distribution(decoded_outputs[ind], decoded_outputs[ind + 1], ramp = self.ramp)
            cr = group.get_cost(dist, par_grouped[indx])
            outs[name] = cr
            cost_recon += cr
            ind += 2
            indx += 1

        #kl_loss = torch.mean(self.KL_gauss(mu_r, log_var_r, mu_q, log_var_q))
        kl_div = self.KL_multigauss(q_dist, r1_dist, n_samp = 100)
        #kl_div = torch.distributions.kl.kl_divergence(q_dist, r1_dist)

        """
        if torch.any(torch.isnan(mu_par)) or torch.any(torch.isnan(log_var_par)):
            nanind = torch.where(torch.isnan(mu_par))[0]
            print(nanind)
            print(z_sample[nanind])
            print(mu_q[nanind], log_var_q[nanind])
            print(mu_r[nanind], log_var_r[nanind])
        """
        #cost_recon = self.log_likelihood_gauss(par, mu_par, log_var_par)
        #print(np.shape(gauss_loss))

        recon_loss = torch.mean(cost_recon)
        kl_loss = torch.mean(kl_div)

        return recon_loss, kl_loss, par_loss, outs

        
    
    def draw_samples(self, shared_y, mu_r, log_var_r, cat_weight_r, num_samples, z_dim, x_dim, return_latent = False):
        """ Draw samples from network for testing"""
        #z_sample = self.cat_gauss_dist(mu_r.repeat(num_samples, 1),  mu_r.repeat(num_samples, 1)).rsample()
        
        z_sample = self.cat_gauss_dist(
            mu_r.repeat(num_samples, 1, 1), 
            log_var_r.repeat(num_samples, 1, 1),
            cat_weight_r.repeat(num_samples, 1)).sample()
        

        # input the latent space samples into decoder r2(x|z, y)  
        #ys = y.repeat(1,num_samples).view(-1, y.size(0)) # repeat data so same size as the z samples
        ys = shared_y.repeat(num_samples,1).view(-1, shared_y.size(0)) # repeat data so same size as the z samples

        # sample parameter space from returned mean and variance 
        #mu_par, log_var_par = self.decode(z_sample,ys) # decode r2(x|z, y) from z     
        #samp = self.gauss_sample(mu_par, log_var_par)
        decoded_outputs = self.decode(z = z_sample, y=ys)
        ind = 0
        indx = 0
        dist_samples = []
        for group in self.grouped_params.values():
            dist = group.get_distribution(decoded_outputs[ind], decoded_outputs[ind + 1],  ramp = self.ramp)
            dist_samples.append(group.sample(dist, num_samples))
            ind += 2
            indx += 1   

        samp = torch.cat(dist_samples, dim=1)
        samp = samp[:,self.reverse_params_order]

        if self.include_parameter_network:
            samp = self.par_network_unscale(samp)

        if return_latent:
            return samp.detach().cpu().numpy(), z_sample.detach().cpu().numpy()
        else:
            return samp.detach().cpu().numpy(), None
    
    def test_latent(self, y, par, num_samples):
        """generating samples when testing the network, returns latent samples as well (used during training to get latent samples)"""
        num_data = y.size(0)                                                                                                                                                     
        x_samples = []

        if self.include_parameter_network:
            par = self.par_network_unscale(par)

        # reorder for distribution grouping
        par = par[:,self.new_params_order]

        shared_y = self.shared_encode(y)       
        mu_r, log_var_r, cat_weight_r = self.encode_r(shared_y) # encode r1(z|y) 
        mu_q, log_var_q = self.encode_q(shared_y,par) # encode q(z|y) 
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
                cat_weight_r[i].repeat(num_samples, 1)).rsample()
            zq_sample = self.multi_gauss_dist(
                mu_q[i].repeat(num_samples, 1), 
                log_var_q[i].repeat(num_samples, 1)).rsample()
            zr_samples.append(zr_sample.cpu().numpy())
            zq_samples.append(zq_sample.cpu().numpy())
            # input the latent space samples into decoder r2(x|z, y)  
            ys = shared_y[i].repeat(num_samples,1).view(-1, shared_y.size(1)) # repeat data so same size as the z samples

            mu_par, log_var_par = self.decode(zr_sample,ys) # decode r2(x|z, y) from z        
            samp = self.gauss_sample(mu_par, log_var_par)


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
        if y[0].shape != (self.n_channels, self.y_dim):
            raise Exception(f"input wrong shape: {y.shape} nut should be (N, {self.n_channels}, {self.y_dim})")                                                                                                                                                 
        transformed_samples = []
        net_samples = []
        if return_latent:
            z_samples = []
            q_samples = []
        
        if par is not None:
            if self.include_parameter_network:
                par = self.par_network_scale(par)

            # reorder for distribution grouping
            par = par[:,self.new_params_order]   

        shared_y = self.shared_encode(y)
        # encode the data into latent space with r1(z,y)          
        mu_r, log_var_r, cat_weight_r = self.encode_r(shared_y) # encode r1(z|y) 
        if return_latent:
            mu_q, log_var_q = self.encode_q(shared_y,par) # encode q(z|y) 
        # get the latent space samples for each input
        for i in range(num_data):
            #print(f"index: {i}")
            # generate initial samples
            t_net_samples, t_znet_samples = self.draw_samples(
                shared_y[i], 
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
                        log_var_q[i].repeat(num_samples, 1, 1)).rsample().detach().cpu().numpy()
                )

            net_samples.append(t_net_samples)
            if return_latent:
                z_samples.append(t_znet_samples)
        
        if return_latent:
            return np.array(net_samples), np.array(z_samples), np.array(q_samples)
        else:
            return np.array(net_samples)
    

    def extra_test(self, y, num_samples, transform_func=None, return_latent = False, par = None):
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
        net_samples = []
        r_samples = []
        q_samples = []
        # encode the data into latent space with r1(z,y)          
        mu_r, log_var_r, cat_weight_r = self.encode_r(y) # encode r1(z|y) 
        if return_latent:
            mu_q, log_var_q = self.encode_q(y,par) # encode q(z|y) 
        # get the latent space samples for each input
        for i in range(num_data):
            #print(f"index: {i}")
            # generate initial samples
            t_net_samples, t_znet_samples = self.draw_samples(
                y[i], 
                mu_r[i], 
                log_var_r[i], 
                cat_weight_r[i],
                num_samples, 
                self.z_dim, 
                self.x_dim, 
                return_latent = True
                )

            new_mu_q, new_log_var_q = self.encode_q(y.repeat(num_samples, 1, 1), t_net_samples)
            
            t_q_samples = self.multi_gauss_dist(
                            new_mu_q, 
                            new_log_var_q).rsample().detach().cpu().numpy()

            q_samples.append(t_q_samples)
            r_samples.append(t_znet_sample)
            net_samples.append(t_net_samples)
        
        return np.array(net_samples), np.array(r_samples), np.array(q_samples)

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
            #print("index: {}".format(i))
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


    def get_network(self, layers, label = "share"):
        """ create the layers of the network from the list of layers in config file or list"""
        conv = in_tensor
        channel_size = self.n_channels
        data_length = self.y_dim
        out_layers = []
        for i, layer in enumerate(layers):
            if layer.split("(")[0] == "Conv1D":
                nfilters, filter_size, stride = layer.split("(")[1].strip(")").split(",")
                out_layers = self.ConvBlock(out_layers, channel_size, int(nfilters), int(filter_size), int(stride), name = "{}_conv_{}".format(label,i))
                channel_size = int(nfilters)
                data_length = data_length/int(stride)
            elif layer.split("(")[0] == "SepConv1D":
                raise Exception("Not implemented")
                #nfilters, filter_size, stride = layer.split("(")[1].strip(")").split(",")
                #conv = self.SepConvBlock(conv, int(nfilters), int(filter_size), int(stride), name = "{}_conv_{}".format(label,i))
            elif layer.split("(")[0] == "ResBlock":
                raise Exception("Not implemented")
                #nfilters, filter_size, stride = layer.split("(")[1].strip(")").split(",")
                # add a bottleneck block
                #conv = self.ResBlock(conv, [int(nfilters), int(nfilters), int(nfilters)], [1, int(filter_size)], int(stride), name = "{}_res_{}".format(label, i))
            elif layer.split("(")[0] == "Linear":
                num_neurons = layer.split("(")[1].strip(")")
                out_layers = self.LinearBlock(out_layers, data_length, int(num_neurons), name = "{}_dense_{}".format(label, i))
                data_length = int(num_neurons)
            elif layer.split("(")[0] == "Reshape":
                s1,s2 = layer.split("(")[1].strip(")").split(",")
                out_layers.append(nn.Unflatten((int(s1),int(s2))))
                data_length = int(s1)
                channel_size = int(s2)
            elif layer.split("(")[0] == "Flatten":
                out_layers.append(nn.Flatten())
                data_length = channel_size * data_length
                channel_size = 0
            elif layer.split("(")[0] == "Transformer":
                raise Exception("Not implemented")
                #head_size, num_heads, ff_dim = layer.split("(")[1].strip(")").split(",")
                #conv = self.TransformerEncoder(conv, int(head_size), int(num_heads), int(ff_dim), dropout=0)
            elif layer.split("(")[0] == "MaxPool":
                pool_size = layer.split("(")[1].strip(")")
                out_layers.append(nn.MaxPool1d(kernel_size=int(pool_size)))
                data_length = int(data_length/int(pool_size))
            else:
                raise Exception(f"Error: No layer with name {layer.split('(')[0]}, please use one of Conv1D, SepConv1D, ResBlock, Linear, Reshape, Flatten, Transformer, MaxPool")

        return torch.nn.Sequential(out_layers)

    def ConvBlock(self, layers, in_channels, filters, kernel_size, stride, name = ""):
        #, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer
        layers.append(nn.Conv1d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, stride=stride, padding="same", kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer, name = name))
        layers.append(nn.BatchNorm1d(filters, name = "{}_batchnorm".format(name)))
        layers.append(self.activation)
        return layers
    def LinearBlock(self, layers, input_neurons, num_neurons, name = ""):

        layers.append(nn.Linear(input_neruons, num_neurons, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer, name = name))
        layers.append(nn.BatchNorm1d(num_neurons, name = "{}_batchnorm".format(name)))
        layers.append(self.activation)
        return layers