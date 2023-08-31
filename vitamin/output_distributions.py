#import tensorflow as tf
#import tensorflow_probability as tfp
import numpy as np
import torch
from .new_distributions.truncated_gauss import TruncatedNormalDist

class TruncatedNormal:

    def __init__(self, pars, config, index = None):
        """
        Truncated normal distribution within some bounds defined by configs
        args
        ---------------
        pars: list
            list of the parameters input into this distribution (in order)
        config: dict
            config file to be used with this setup (not used for this distribution)
        index : str or int (default None)
            index of the distribution, if multiple groups with same distribution index with this value

        """
        self.name = "TruncatedNormal"
        if index is not None:
            self.name += "_{}".format(index)

        self.pars = pars
        self.num_pars = len(self.pars)
        self.num_outputs = [self.num_pars, self.num_pars]

    def get_distribution(self, mean, logvar, ramp = 1.0):
        # truncated normal for non-periodic params    
        dist = TruncatedNormalDist(loc=mean, scale=torch.sqrt(torch.exp(logvar)), a=-10.0 + ramp*10.0, b=1.0 + 10.0 - ramp*10.0)                  
        return dist

    def get_networks(self,indim):
        mean =  torch.nn.Linear(indim, self.num_pars)
        logvar = torch.nn.Linear(indim, self.num_pars)
        return mean, logvar

    def get_cost(self, dist, x):
        return -1.0*torch.mean(torch.sum(dist.log_prob(x),dim=1),dim=0)

    def sample(self, dist, max_samples):
        return dist.sample()

class TruncatedNormalCat:

    def __init__(self, pars, config, index = None):
        """
        Truncated normal distribution within some bounds defined by configs
        args
        ---------------
        pars: list
            list of the parameters input into this distribution (in order)
        config: dict
            config file to be used with this setup (not used for this distribution)
        index : str or int (default None)
            index of the distribution, if multiple groups with same distribution index with this value

        """
        self.name = "TruncatedNormal"
        if index is not None:
            self.name += "_{}".format(index)

        self.pars = pars
        self.num_pars = len(self.pars)
        self.num_outputs = [self.num_pars, self.num_pars]

    def get_distribution(self, mean, logvar, ramp = 1.0):
        # truncated normal for non-periodic params    
        dist = TruncatedNormalDist(loc=mean, scale=torch.sqrt(torch.exp(logvar)), a=-10.0 + ramp*10.0, b=1.0 + 10.0 - ramp*10.0) 
        mix = torch.distributions.Categorical(probs=cat_weight)
        #comp = torch.distributions.Independent(torch.distributions.Normal(mean, torch.exp(log_var + (1-self.ramp)*self.logvarfactor)), 1)
        comp = torch.distributions.Independent(torch.distributions.Normal(mean, torch.exp(log_var)), 1)
        gmm = ReparametrizedMixtureSameFamily(mix, comp)                 
        return dist

    def get_networks(self,indim):
        mean =  torch.nn.Linear(indim, self.num_pars)
        logvar = torch.nn.Linear(indim, self.num_pars)
        return mean, logvar

    def get_cost(self, dist, x):
        return -1.0*torch.mean(torch.sum(dist.log_prob(x),dim=1),dim=0)

    def sample(self, dist, max_samples):
        return dist.sample()

class Normal:

    def __init__(self, pars, config, index = None):
        """
        Truncated normal distribution within some bounds defined by configs
        args
        ---------------
        pars: list
            list of the parameters input into this distribution (in order)
        config: dict
            config file to be used with this setup (not used for this distribution)
        index : str or int (default None)
            index of the distribution, if multiple groups with same distribution index with this value

        """
        self.name = "Normal"
        if index is not None:
            self.name += "_{}".format(index)

        self.pars = pars
        self.num_pars = len(self.pars)
        self.num_outputs = [self.num_pars, self.num_pars]

    def get_distribution(self, mean, logvar, ramp = 1.0):
        # truncated normal for non-periodic params                      
        tmvn = torch.distributions.Normal(
            loc=mean,
            scale=torch.sqrt(torch.exp(logvar)))

        return tmvn

    def get_networks(self,indim):
        mean =  torch.nn.Linear(indim, self.num_pars)
        logvar = torch.nn.Linear(indim, self.num_pars)
        return mean, logvar

    def get_cost(self, dist, x):
        return -1.0*torch.mean(torch.sum(dist.log_prob(x),dim=1),dim=0)

    def sample(self, dist, max_samples):
        return dist.sample()


class VonMises:

    def __init__(self, pars, config, index = None):
        """
        Von mises distribution in 1d
        args
        ---------------
        pars: list
            list of the parameters input into this distribution (in order)
        config: dict
            config file to be used with this setup (not used for this distribution)
        index : str or int (default None)
            index of the distribution, if multiple groups with same distribution index with this value

        """
        self.name = "VonMises"
        if index is not None:
            self.name += "_{}".format(index)

        self.pars = pars
        self.num_pars = len(self.pars)
        self.num_outputs = [2*self.num_pars, self.num_pars]
        self.fourpisq = 4.0*np.pi*np.pi
        self.lntwopi = np.log(2.0*np.pi)
        self.lnfourpi = np.log(4.0*np.pi)

    def get_distribution(self, mean, logvar, ramp = 1):
        mean_x, mean_y = torch.tensor_split(mean, 2, dim=1)
        # the mean angle (0,2pi) [size (batch,p)]
        mean_angle = torch.remainder(torch.atan2(mean_y,mean_x),2.0*np.pi)

        # define the 2D Gaussian scale [size (batch,2*p)]
        vm = torch.distributions.VonMises(
            mean_angle,
            torch.reciprocal(torch.exp(logvar)))
        
        return vm

    def get_networks(self, indim):
        mean = torch.nn.Linear(indim, 2*self.num_pars)
        logvar = torch.nn.Linear(indim, self.num_pars)
        return mean, logvar


    def get_cost(self, dist, x):
        return -1.0*torch.mean(torch.sum(self.lntwopi + dist.log_prob(2.0*np.pi*x),dim=1),dim=0)

    def sample(self, dist, max_samples):
        return torch.remainder(dist.sample(),(2.0*np.pi))/(2.0*np.pi)


