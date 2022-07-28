import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class JointM1M2:

    def __init__(self, pars, bounds, index = None):
        """ Distribution that us uniform in m1 and m2 where m1 is always greater than m2
        usage in config
        mass_1 = JointM1M2
        mass_2 = JointM1M2

        args
        ---------------
        pars: list
            list of the mass parameters [M_{solar}] input into this distribution (in order)
        config: dict
            config file to be used with this setup (not used for this distribution)
        index : str or int (default None)
            index of the distribution, if multiple groups with same distribution index with this value

        """
        self.name = "JointM1M2"
        if index is not None:
            self.name += "_{}".format(index)
        self.pars = pars
        if self.pars[0] == "mass_2":
            self.order_flipped = True
        elif self.pars[0] == "mass_1":
            self.order_flipped = False

        self.num_pars = len(self.pars)
        if self.num_pars != 2:
            raise Exception("Please only use two variables for JointM1M2")
        self.num_outputs = [self.num_pars,self.num_pars]
        self.get_cost = self.cost_setup()
        self.sample = self.sample_setup()
        self.Root = tfp.distributions.JointDistributionCoroutine.Root

    def get_distribution(self, mean, logvar, ramp = 1.0):
        """
        Sets up this distribution
        """
        mean1, mean2 = tf.split(mean, num_or_size_splits=2, axis=1)
        logvar1, logvar2 = tf.split(logvar, num_or_size_splits=2, axis=1)

        # condition m2 on the m1 distribution
        joint = tfp.distributions.JointDistributionSequential([
            tfp.distributions.TruncatedNormal(
                loc=tf.cast(mean1, tf.float32),
                scale=tf.cast(tf.sqrt(tf.exp(logvar1)), tf.float32),
                low=0, high=1),
            lambda b0: tfp.distributions.TruncatedNormal(
                loc=tf.cast(mean2, tf.float32),
                scale=tf.cast(tf.sqrt(tf.exp(logvar2)), tf.float32),
                low=0, high=b0),
        ])
        return joint

    def get_networks(self,logvar_activation = "none"):
        """Get the Dense layers for this output distribution
        """
        mean =  tf.keras.layers.Dense(2,activation='sigmoid', use_bias = True, name="{}_mean".format(self.name))
        logvar = tf.keras.layers.Dense(2,use_bias=True, name="{}_logvar".format(self.name), activation=logvar_activation)
        return mean, logvar

    def cost_setup(self):
        """ Evaluate the log probability for this output and get the reduce cost for a batch
        """
        if self.order_flipped:
            def get_cost(dist, x):
                x2, x1 = tf.split(x, num_or_size_splits=2, axis=1)
                return -1.0*tf.reduce_mean(tf.reduce_sum(dist.log_prob(x1, x2),axis=1),axis=0)
        else:
            def get_cost(dist, x):
                x1, x2 = tf.split(x, num_or_size_splits=2, axis=1)
                return -1.0*tf.reduce_mean(tf.reduce_sum(dist.log_prob(x1, x2),axis=1),axis=0)

        return get_cost

    def sample_setup(self):
        """Sample from this distribution """
        if self.order_flipped:
            # reverse the order of the two parmaeters when sampling
            def sample(dist, max_samples):
                # transpose amd squeeze to get [samples, parameters]
                return tf.squeeze(tf.transpose(dist.sample(), [1, 0, 2])[:,::-1], 2)
        else:
            def sample(dist, max_samples):
                return tf.squeeze(tf.transpose(dist.sample(), [1, 0, 2]), 2)
        return sample

class JointChirpmassMR:

    def __init__(self, pars, bounds, index=None):
        """
        Joint distribution for Chirpmass and symetric mass ratio
        Takes in the chirp mass and mass ratio as parameters 

        usage in config
        mass_ratio = JointChirpmassMR
        chirpmass = JointChirpmassMR
        args
        ---------------
        pars: list
            list of the parameters input into this distribution (in order)
        config: dict
            config file to be used with this setup
        index : str or int (default None)
            index of the distribution, if multiple groups with same distribution index with this value

        """
        self.name = "JointChirpmassMR"
        if index is not None:
            self.name += "_{}".format(index)
        self.bounds = bounds
        if self.config is None:
            raise("Please input a config file with bounds for chirpmass/massration/m1/m2")
        self.pars = pars
        if self.pars[0] == "chirp_mass":
            self.order_flipped = True
        elif self.pars[0] == "mass_ratio":
            self.order_flipped = False

        self.num_pars = len(self.pars)
        if self.num_pars != 2:
            raise Exception("Please only use two variables for JointM1M2")
        self.num_outputs = [self.num_pars,self.num_pars]
        self.get_cost = self.cost_setup()
        self.sample = self.sample_setup()

    def Mconstrainm1(self, q_norm, m):
        """Contraint for the chirp mass based on maxmium mass of m1                                                                         
        """
        q = q_norm*(self.bounds["mass_ratio_max"] - self.bounds["mass_ratio_min"]) + self.bounds["mass_ratio_min"]
        num = (q*m*m)**(3./5.)
        den = (m*(1 + q))**(1./5.)
        chirpmass = num/den
        inds = tf.where(chirpmass > self.bounds["chirp_mass_max"])
        if inds.shape[0] is not None:
            updates = tf.ones(inds.shape[0], dtype=tf.float32)*np.float32(self.bounds["chirp_mass_max"])
            chirpmass = tf.tensor_scatter_nd_update(chirpmass, inds, updates)
        return (chirpmass - self.bounds["chirp_mass_min"])/(self.bounds["chirp_mass_max"] - self.bounds["chirp_mass_min"])

    def Mconstrainm2(self, q_norm, m):
        """Contraint for the chirp mass based on minimum mass of m2"""
        q = q_norm*(self.bounds["mass_ratio_max"] - self.bounds["mass_ratio_min"]) + self.bounds["mass_ratio_min"]
        num = ((1/q)*m*m)**(3./5.)
        den = (m*(1 + 1/q))**(1./5.)
        chirpmass = num/den
        inds = tf.where(chirpmass < self.bounds["chirp_mass_min"])
        if inds.shape[0] is not None:
            updates = tf.ones(inds.shape[0], dtype=tf.float32)*np.float32(self.bounds["chirp_mass_min"])
            chirpmass = tf.tensor_scatter_nd_update(chirpmass, inds, updates)
        return (chirpmass  - self.bounds["chirp_mass_min"])/(self.bounds["chirp_mass_max"] - self.bounds["chirp_mass_min"])

    def get_distribution(self, mean, logvar, ramp = 1.0):
        # this is not working yet
        mean_q, mean_cm = tf.split(mean, num_or_size_splits=2, axis=1)
        logvarq, logvarcm = tf.split(logvar, num_or_size_splits=2, axis=1)

        joint = tfp.distributions.JointDistributionSequential([
            tfp.distributions.TruncatedNormal(
                loc=tf.cast(mean_q,dtype=tf.float32),
                scale=tf.cast(tf.sqrt(tf.exp(logvarq)),dtype=tf.float32),
                low=0, high=1),
            lambda b0: tfp.distributions.TruncatedNormal(
                loc=tf.cast(mean_cm,dtype=tf.float32),
                scale=tf.cast(tf.sqrt(tf.exp(logvarcm)),dtype=tf.float32),
                low=self.Mconstrainm2(b0, self.bounds["mass_1_min"]), 
                high=self.Mconstrainm1(b0, self.bounds["mass_1_max"])),
        ])
        return joint

    def get_networks(self,logvar_activation="none"):
        # setup network for joint sitribution
        mean =  tf.keras.layers.Dense(2,activation='sigmoid', use_bias = True, name="{}_mean".format(self.name))
        logvar = tf.keras.layers.Dense(2,use_bias=True, name="{}_logvar".format(self.name),activation = logvar_activation)
        return mean, logvar

    def cost_setup(self):
        if self.order_flipped:
            def get_cost(dist, x):
                # reverse order of true parmaeters to estimate logprob
                x2, x1 = tf.split(x, num_or_size_splits=2, axis=1)
                return -1.0*tf.reduce_mean(tf.reduce_sum(dist.log_prob(x1, x2),axis=1),axis=0)
        else:
            def get_cost(dist, x):
                x1, x2 = tf.split(x, num_or_size_splits=2, axis=1)
                return -1.0*tf.reduce_mean(tf.reduce_sum(dist.log_prob(x1, x2),axis=1),axis=0)

        return get_cost

    def sample_setup(self):
        if self.order_flipped:
            # reverse order of samples based on inputs
            def sample(dist, max_samples):
                # transpose amd squeeze to get [samples, parameters]
                return tf.squeeze(tf.transpose(dist.sample(), [1, 0, 2])[:,::-1], 2)
        else:
            def sample(dist, max_samples):
                return tf.squeeze(tf.transpose(dist.sample(), [1, 0, 2]), 2)
        return sample

class JointChirpmassMRM1M2:

    def __init__(self, pars, bounds, index = None):
        """                                                                                                                                
        Joint distribution for Chirpmass and symetric mas ratio
        This takes in m1 and m2 as the parameters and converts to mass ratio and chirp mass internally. 
        The distribution is made in chirpmass and mass ratio space.                                                                            
      
        usage in config
        mass_1 = JointChirpmassMRM1M2
        mass_2 = JointChirpmassMRM1M2
     
        args
        ---------------
        pars: list
            list of the parameters input into this distribution (in order)
        config: dict
            config file to be used with this setup (not used for this distribution)
        index : str or int (default None)
            index of the distribution, if multiple groups with same distribution index with this value

        """
        self.name = "JointChirpmassMRM1M2"
        if index is not None:
            self.name += "_{}".format(index)
        self.bounds = bounds
        if self.config is None:
            raise("Please input a config file with bounds for chirpmass/massration/m1/m2")
        self.pars = pars
        if self.pars[0] == "mass_2":
            self.order_flipped = True
        elif self.pars[0] == "mass_1":
            self.order_flipped = False

        self.num_pars = len(self.pars)
        if self.num_pars != 2:
            raise Exception("Please only use two variables for JointM1M2")
        self.num_outputs = [self.num_pars,self.num_pars]
        self.get_cost = self.cost_setup()
        self.sample = self.sample_setup()

    def Mconstrainm1(self, q_norm, m):
        """Contraint for the chirp mass based on maxmium mass of m1                                                                         
        """
        # convert q from normalised q back into real space
        q = q_norm*(self.bounds["mass_ratio_max"] - self.bounds["mass_ratio_min"]) + self.bounds["mass_ratio_min"]
        # calculate the chirp mass from the q and given mass
        num = (q*m*m)**(3./5.)
        den = (m*(1 + q))**(1./5.)
        chirpmass = num/den
        # if the chirp mass is greater than the maximum allowed chirp mass then set to max chirp mass 
        inds = tf.where(chirpmass > self.bounds["chirp_mass_max"])
        if inds.shape[0] is not None:
            updates = tf.ones(inds.shape[0], dtype=tf.float32)*np.float32(self.bounds["chirp_mass_max"])
            chirpmass = tf.tensor_scatter_nd_update(chirpmass, inds, updates)
        # renormalise the chirpmass between ranges
        return (chirpmass - self.bounds["chirp_mass_min"])/(self.bounds["chirp_mass_max"] - self.bounds["chirp_mass_min"])

    def Mconstrainm2(self, q_norm, m):
        """Contraint for the chirp mass based on minimum mass of m2"""
        q = q_norm*(self.bounds["mass_ratio_max"] - self.bounds["mass_ratio_min"]) + self.bounds["mass_ratio_min"]
        num = ((1/q)*m*m)**(3./5.)
        den = (m*(1 + 1/q))**(1./5.)
        chirpmass = num/den
        inds = tf.where(chirpmass < self.bounds["chirp_mass_min"])
        if inds.shape[0] is not None:
            updates = tf.ones(inds.shape[0], dtype=tf.float32)*np.float32(self.bounds["chirp_mass_min"])
            chirpmass = tf.tensor_scatter_nd_update(chirpmass, inds, updates)
        return (chirpmass  - self.bounds["chirp_mass_min"])/(self.bounds["chirp_mass_max"] - self.bounds["chirp_mass_min"])
    
    def get_distribution(self, mean, logvar, ramp = 1.0):
        mean_q, mean_cm = tf.split(mean, num_or_size_splits=2, axis=1)
        logvarq, logvarcm = tf.split(logvar, num_or_size_splits=2, axis=1)

        # sample from q, then set chirpmass limits based on about constraint equations
        joint = tfp.distributions.JointDistributionSequential([
            tfp.distributions.TruncatedNormal(
                loc=tf.cast(mean_q,dtype=tf.float32),
                scale=tf.cast(tf.sqrt(tf.exp(logvarq)),dtype=tf.float32),
                low=0, high=1),
            lambda b0: tfp.distributions.TruncatedNormal(
                loc=tf.cast(mean_cm,dtype=tf.float32),
                scale=tf.cast(tf.sqrt(tf.exp(logvarcm)),dtype=tf.float32),
                low=self.Mconstrainm2(b0, self.bounds["mass_2_min"]),
                high=self.Mconstrainm1(b0, self.bounds["mass_1_max"])),
        ])
        return joint

    def get_networks(self,logvar_activation="none"):
        # setup network for joint sitribution                                                                                               
        mean =  tf.keras.layers.Dense(2,activation='sigmoid', use_bias = True, name="{}_mean".format(self.name))
        logvar = tf.keras.layers.Dense(2,use_bias=True, name="{}_logvar".format(self.name),activation = logvar_activation)
        return mean, logvar
    
    def component_masses_to_chirpmass_massratio(self, normed_mass_1, normed_mass_2):
        """ Convert component masses to normalised chirp mass and mass ratio"""
        mass_1 = normed_mass_1*(self.bounds["mass_1_max"] - self.bounds["mass_1_min"]) + self.bounds["mass_1_min"]
        mass_2 = normed_mass_2*(self.bounds["mass_2_max"] - self.bounds["mass_2_min"]) + self.bounds["mass_2_min"]
        chirpmass, massratio = ((mass_1*mass_2)**(0.6))/((mass_1 + mass_2)**(0.2)), mass_2/mass_1
        normed_chirpmass = (chirpmass - self.bounds["chirp_mass_min"])/(self.bounds["chirp_mass_max"] - self.bounds["chirp_mass_min"])
        normed_massratio = (massratio - self.bounds["mass_ratio_min"])/(self.bounds["mass_ratio_max"] - self.bounds["mass_ratio_min"])
        return normed_massratio, normed_chirpmass

    def chirpmass_massratio_to_component_masses(self, chirpmass, massratio):
        """ Convert component masses to normalised chirp mass and mass ratio"""
        chirpmass = chirpmass*(self.bounds["chirp_mass_max"] - self.bounds["chirp_mass_min"]) + self.bounds["chirp_mass_min"]
        massratio = massratio*(self.bounds["mass_ratio_max"] - self.bounds["mass_ratio_min"]) + self.bounds["mass_ratio_min"]
        massratio = massratio
        total_mass = chirpmass*(1+massratio)**1.2/massratio**0.6
        mass_1 = total_mass/(1+massratio)
        mass_2 = mass_1*massratio
        normed_mass_1 = (mass_1 - self.bounds["mass_1_min"])/(self.bounds["mass_1_max"] - self.bounds["mass_1_min"])
        normed_mass_2 = (mass_2 - self.bounds["mass_2_min"])/(self.bounds["mass_2_max"] - self.bounds["mass_2_min"])
        return normed_mass_1, normed_mass_2
    
    def cost_setup(self):
        if self.order_flipped:
            def get_cost(dist, x):
                # reverse order of true parmaeters to estimate logprob                                                                      
                x2, x1 = tf.split(x, num_or_size_splits=2, axis=1)
                normed_massratio, normed_chirpmass = self.component_masses_to_chirpmass_massratio(x1,x2)
                return -1.0*tf.reduce_mean(tf.reduce_sum(dist.log_prob(normed_massratio, normed_chirpmass),axis=1),axis=0)
        else:
            def get_cost(dist, x):
                x1, x2 = tf.split(x, num_or_size_splits=2, axis=1)
                normed_massratio, normed_chirpmass = self.component_masses_to_chirpmass_massratio(x1,x2)
                return -1.0*tf.reduce_mean(tf.reduce_sum(dist.log_prob(normed_massratio, normed_chirpmass),axis=1),axis=0)

        return get_cost

    def sample_setup(self):
        if self.order_flipped:
            # reverse order of samples based on inputs                                                                                      
            def sample(dist, max_samples):
                # transpose amd squeeze to get [samples, parameters]                                                                       
                chirpmass, massratio =  tf.split(tf.squeeze(tf.transpose(dist.sample(), [1, 0, 2]), 2), num_or_size_splits=2, axis=1)
                mass_1, mass_2 = self.chirpmass_massratio_to_component_masses(chirpmass, massratio)
                return tf.concat([mass_1,mass_2], axis = 1)
        else:
            def sample(dist, max_samples):
                massratio, chirpmass =  tf.split(tf.squeeze(tf.transpose(dist.sample(), [1, 0, 2]), 2), num_or_size_splits=2, axis=1)
                mass_1, mass_2 = self.chirpmass_massratio_to_component_masses(chirpmass, massratio)
                return tf.concat([mass_1,mass_2], axis = 1)
        return sample

class DiscardChirpmassMRM1M2:

    def __init__(self, pars, bounds, index = None):
        """
        Calculate the distribution on m1 and m2 converting to chirp mass and mass ration internally, no joint distrinution but discards point outside of prior bounds

        args
        ---------------
        pars: list
            list of the parameters input into this distribution (in order)
        config: dict
            config file to be used with this setup (not used for this distribution)
        index : str or int (default None)
            index of the distribution, if multiple groups with same distribution index with this value

        """
        self.name = "DiscardChirpmassMRM1M2"
        if index is not None:
            self.name += "_{}".format(index)

        self.pars = pars
        self.bounds = bounds

        if self.pars[0] == "mass_2":
            self.order_flipped = True
        elif self.pars[0] == "mass_1":
            self.order_flipped = False

        self.num_pars = len(self.pars)
        if self.num_pars != 2:
            raise Exception("Please only use two variables for JointM1M2")
        self.num_outputs = [self.num_pars,self.num_pars]
        self.get_cost = self.cost_setup()
        self.sample = self.sample_setup()

    def get_distribution(self, mean, logvar, ramp = 1.0):

        mean_q, mean_cm = tf.split(mean, num_or_size_splits=2, axis=1)
        logvarq, logvarcm = tf.split(logvar, num_or_size_splits=2, axis=1)

        tmvnq = tfp.distributions.TruncatedNormal(
            loc=tf.cast(mean_q, dtype=tf.float32),
            scale=tf.cast(tf.sqrt(tf.exp(logvarq)),dtype=tf.float32),
            low=-10.0 + ramp*10.0, high=1.0 + 10.0 - ramp*10.0)

        tmvncm = tfp.distributions.TruncatedNormal(
            loc=tf.cast(mean_cm, dtype=tf.float32),
            scale=tf.cast(tf.sqrt(tf.exp(logvarcm)),dtype=tf.float32),
            low=-10.0 + ramp*10.0, high=1.0 + 10.0 - ramp*10.0)

        return tmvnq, tmvncm

    def get_networks(self,logvar_activation="none"):
        mean =  tf.keras.layers.Dense(self.num_pars, activation='sigmoid', use_bias = True, name="{}_mean".format(self.name))
        logvar = tf.keras.layers.Dense(self.num_pars, use_bias=True, name="{}_logvar".format(self.name), activation=logvar_activation)
        return mean, logvar

    def cost_setup(self):
        if self.order_flipped:
            def get_cost(dist, x):
                # reverse order of true parmaeters to estimate logprob                                                                      
                x1, x2 = tf.split(x, num_or_size_splits=2, axis=1)
                normed_massratio, normed_chirpmass = self.component_masses_to_chirpmass_massratio(x1,x2)
                qcost = -1.0*tf.reduce_mean(tf.reduce_sum(dist[1].log_prob(normed_massratio),axis=1),axis=0)
                mrcost = -1.0*tf.reduce_mean(tf.reduce_sum(dist[0].log_prob(normed_chirpmass),axis=1),axis=0)
                return qcost + mrcost 
        else:
            def get_cost(dist, x):
                x1, x2 = tf.split(x, num_or_size_splits=2, axis=1)
                normed_massratio, normed_chirpmass = self.component_masses_to_chirpmass_massratio(x1,x2)
                qcost = -1.0*tf.reduce_mean(tf.reduce_sum(dist[0].log_prob(normed_massratio),axis=1),axis=0)
                mrcost = -1.0*tf.reduce_mean(tf.reduce_sum(dist[1].log_prob(normed_chirpmass),axis=1),axis=0)
                return qcost + mrcost 

        return get_cost

    def sample_setup(self):
        if self.order_flipped:
            # reverse order of samples based on inputs                                                                                      
            def sample(dist, max_samples):
                # transpose amd squeeze to get [samples, parameters]                                                                       
                massratio =  dist[1].sample()
                chirpmass =  dist[0].sample()
                mass_1, mass_2 = self.chirpmass_massratio_to_component_masses(chirpmass, massratio)
                return tf.concat([mass_1,mass_2], axis = 1)
        else:
            def sample(dist, max_samples):
                massratio =  dist[0].sample()
                chirpmass =  dist[1].sample()
                mass_1, mass_2 = self.chirpmass_massratio_to_component_masses(chirpmass, massratio)
                return tf.concat([mass_1,mass_2], axis = 1)
        return sample

    def component_masses_to_chirpmass_massratio(self, normed_mass_1, normed_mass_2):
        """ Convert component masses to normalised chirp mass and mass ratio"""
        mass_1 = normed_mass_1*(self.bounds["mass_1_max"] - self.bounds["mass_1_min"]) + self.bounds["mass_1_min"]
        mass_2 = normed_mass_2*(self.bounds["mass_2_max"] - self.bounds["mass_2_min"]) + self.bounds["mass_2_min"]
        chirpmass, massratio = ((mass_1*mass_2)**(0.6))/((mass_1 + mass_2)**(0.2)), mass_2/mass_1
        normed_chirpmass = (chirpmass - self.bounds["chirp_mass_min"])/(self.bounds["chirp_mass_max"] - self.bounds["chirp_mass_min"])
        normed_massratio = (massratio - self.bounds["mass_ratio_min"])/(self.bounds["mass_ratio_max"] - self.bounds["mass_ratio_min"])
        return normed_massratio, normed_chirpmass

    def chirpmass_massratio_to_component_masses(self, chirpmass, massratio):
        """ Convert component masses to normalised chirp mass and mass ratio"""
        chirpmass = chirpmass*(self.bounds["chirp_mass_max"] - self.bounds["chirp_mass_min"]) + self.bounds["chirp_mass_min"]
        massratio = massratio*(self.bounds["mass_ratio_max"] - self.bounds["mass_ratio_min"]) + self.bounds["mass_ratio_min"]
        massratio = massratio
        total_mass = chirpmass*(1+massratio)**1.2/massratio**0.6
        mass_1 = total_mass/(1+massratio)
        mass_2 = mass_1*massratio
        normed_mass_1 = (mass_1 - self.bounds["mass_1_min"])/(self.bounds["mass_1_max"] - self.bounds["mass_1_min"])
        normed_mass_2 = (mass_2 - self.bounds["mass_2_min"])/(self.bounds["mass_2_max"] - self.bounds["mass_2_min"])
        return normed_mass_1, normed_mass_2



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
        tmvn = tfp.distributions.TruncatedNormal(
            loc=tf.cast(mean, dtype=tf.float32),
            scale=tf.cast(tf.sqrt(tf.exp(logvar)),dtype=tf.float32),
            low=-10.0 + ramp*10.0, high=1.0 + 10.0 - ramp*10.0)

        return tmvn

    def get_networks(self,logvar_activation="none"):
        mean =  tf.keras.layers.Dense(self.num_pars, activation='sigmoid', use_bias = True, name="{}_mean".format(self.name))
        logvar = tf.keras.layers.Dense(self.num_pars, use_bias=True, name="{}_logvar".format(self.name), activation=logvar_activation)
        return mean, logvar

    def get_cost(self, dist, x):
        return -1.0*tf.reduce_mean(tf.reduce_sum(dist.log_prob(x),axis=1),axis=0)

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
        tmvn = tfp.distributions.Normal(
            loc=tf.cast(mean, dtype=tf.float32),
            scale=tf.cast(tf.sqrt(tf.exp(logvar)),dtype=tf.float32))

        return tmvn

    def get_networks(self,logvar_activation="none"):
        mean =  tf.keras.layers.Dense(self.num_pars, activation='sigmoid', use_bias = True, name="{}_mean".format(self.name))
        logvar = tf.keras.layers.Dense(self.num_pars, use_bias=True, name="{}_logvar".format(self.name), activation=logvar_activation)
        return mean, logvar

    def get_cost(self, dist, x):
        return -1.0*tf.reduce_mean(tf.reduce_sum(dist.log_prob(x),axis=1),axis=0)

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
        self.lntwopi = tf.math.log(2.0*np.pi)
        self.lnfourpi = tf.math.log(4.0*np.pi)

    def get_distribution(self, mean, logvar, ramp = 1):
        mean_x, mean_y = tf.split(mean, num_or_size_splits=2, axis=1)
        # the mean angle (0,2pi) [size (batch,p)]
        mean_angle = tf.math.floormod(tf.math.atan2(tf.cast(mean_y,dtype=tf.float32),tf.cast(mean_x,dtype=tf.float32)),2.0*np.pi)

        # define the 2D Gaussian scale [size (batch,2*p)]
        vm = tfp.distributions.VonMises(
            loc=tf.cast(mean_angle,dtype=tf.float32),
            concentration=tf.cast(tf.math.reciprocal(tf.exp(logvar)),dtype=tf.float32))
        
        return vm

    def get_networks(self, logvar_activation="none"):
        mean =  tf.keras.layers.Dense(2*self.num_pars, use_bias = True, name="{}_mean".format(self.name))
        logvar = tf.keras.layers.Dense(self.num_pars, use_bias=True, name="{}_logvar".format(self.name),activation=logvar_activation)
        return mean, logvar


    def get_cost(self, dist, x):
        return -1.0*tf.reduce_mean(tf.reduce_sum(self.lntwopi + dist.log_prob(2.0*np.pi*x),axis=1),axis=0)

    def sample(self, dist, max_samples):
        return tf.math.floormod(dist.sample(),(2.0*np.pi))/(2.0*np.pi)

class JointVonMisesFisher:

    def __init__(self, pars, config, index = None):
        """
        Joint von mises fisher distribution on the sky        
        args
        ---------------
        pars: list
            list of the sky parameters (alpha, delta) [radians] input into this distribution (in order)
        config: dict
            config file to be used with this setup (not used for this distribution)
        index : str or int (default None)
            index of the distribution, if multiple groups with same distribution index with this valu
e
        """
        self.name = "JointVonMisesFisher"
        if index is not None:
            self.name += "_{}".format(index)

        self.pars = pars
        self.num_pars = len(self.pars)
        if self.num_pars != 2:
            raise Exception("Please use two inputs for the joint sky distribution JointVonMisesFisher")
        self.num_outputs = [3,1]
        self.fourpisq = 4.0*np.pi*np.pi
        self.lntwopi = tf.math.log(2.0*np.pi)
        self.lnfourpi = tf.math.log(4.0*np.pi)

    def get_distribution(self, mean, logvar, ramp = 1):
        # define the von mises fisher for the sky
        fvm_loc = tf.reshape(tf.math.l2_normalize(mean, axis=1),[-1,3])  # mean in (x,y,z)
        fvm_con = tf.reshape(tf.math.reciprocal(tf.exp(logvar)),[-1])
        fvm_r2 = tfp.distributions.VonMisesFisher(
            mean_direction = tf.cast(fvm_loc,dtype=tf.float32),
            concentration = tf.cast(fvm_con,dtype=tf.float32)
        )
        
        return fvm_r2

    def get_cost(self, dist, x):

        x_ra, x_dec = tf.split(x, num_or_size_splits=2, axis=1)
        ra_sky = tf.reshape(2*np.pi*x_ra,(-1,1))       # convert the scaled 0->1 true RA value back to radians
        dec_sky = tf.reshape(np.pi*(x_dec - 0.5),(-1,1)) # convert the scaled 0>1 true dec value back to radians
        xyz_unit = tf.concat([tf.cos(ra_sky)*tf.cos(dec_sky),tf.sin(ra_sky)*tf.cos(dec_sky),tf.sin(dec_sky)],axis=1)   # construct the true parameter unit vector
        normed_xyz = tf.math.l2_normalize(xyz_unit,axis=1) # normalise x,y,z coords to r
        # get log prob
        fvm_r2_cost_recon = -1.0*tf.reduce_mean(self.lnfourpi + dist.log_prob(normed_xyz),axis=0)

        return fvm_r2_cost_recon

    def get_networks(self, logvar_activation="none"):
        mean = tf.keras.layers.Dense(3,use_bias=True,name="{}_mean".format(self.name))
        logvar = tf.keras.layers.Dense(1,use_bias=True,activation = logvar_activation,name="{}_logvar".format(self.name))
        return mean, logvar

    def sample(self, dist, max_samples):
        xyz = tf.reshape(dist.sample(),[max_samples,3])
        # convert to rescaled 0-1 ra from the unit vector
        samp_ra = tf.math.floormod(tf.math.atan2(tf.slice(xyz,[0,1],[max_samples,1]),tf.slice(xyz,[0,0],[max_samples,1])),2.0*np.pi)/(2.0*np.pi)
        # convert resaled 0-1 dec from unit vector
        samp_dec = (tf.asin(tf.slice(xyz,[0,2],[max_samples,1])) + 0.5*np.pi)/np.pi
        # group the sky samples 
        samples = tf.reshape(tf.concat([samp_ra,samp_dec],axis=1),[max_samples,2])

        return samples

class JointPowerSpherical:

    def __init__(self, pars, config, index = None):
        self.name = "JointPowerSpherical"
        if index is not None:
            self.name += "_{}".format(index)

        self.pars = pars
        self.num_pars = len(self.pars)
        if self.num_pars != 2:
            raise Exception("Please use two inputs for the joint sky distribution JointVonMisesFisher")
        self.num_outputs = [3,1]
        self.fourpisq = 4.0*np.pi*np.pi
        self.lntwopi = tf.math.log(2.0*np.pi)
        self.lnfourpi = tf.math.log(4.0*np.pi)

    def get_distribution(self, mean, logvar, ramp = 1):
        # define the von mises fisher for the sky
        fvm_loc = tf.reshape(tf.math.l2_normalize(mean, axis=1),[-1,3])  # mean in (x,y,z)
        fvm_con = tf.reshape(tf.math.reciprocal(tf.exp(logvar)),[-1])
        fvm_r2 = tfp.distributions.PowerSpherical(
            mean_direction = tf.cast(fvm_loc,dtype=tf.float32),
            concentration = tf.cast(fvm_con,dtype=tf.float32)
        )
        
        return fvm_r2

    def get_cost(self, dist, x):

        x_ra, x_dec = tf.split(x, num_or_size_splits=2, axis=1)
        ra_sky = tf.reshape(2*np.pi*x_ra,(-1,1))       # convert the scaled 0->1 true RA value back to radians
        dec_sky = tf.reshape(np.pi*(x_dec - 0.5),(-1,1)) # convert the scaled 0>1 true dec value back to radians
        xyz_unit = tf.concat([tf.cos(ra_sky)*tf.cos(dec_sky),tf.sin(ra_sky)*tf.cos(dec_sky),tf.sin(dec_sky)],axis=1)   # construct the true parameter unit vector
        normed_xyz = tf.math.l2_normalize(xyz_unit,axis=1) # normalise x,y,z coords to r
        # get log prob
        fvm_r2_cost_recon = -1.0*tf.reduce_mean(self.lnfourpi + dist.log_prob(normed_xyz),axis=0)

        return fvm_r2_cost_recon

    def get_networks(self,logvar_activation="none"):
        mean = tf.keras.layers.Dense(3,use_bias=True, name="{}_mean".format(self.name))
        logvar = tf.keras.layers.Dense(1,use_bias=True, activation=logvar_activation,name="{}_logvar".format(self.name))
        return mean, logvar

    def sample(self, dist, max_samples):
        xyz = tf.reshape(dist.sample(),[max_samples,3])
        # convert to rescaled 0-1 ra from the unit vector
        samp_ra = tf.math.floormod(tf.math.atan2(tf.slice(xyz,[0,1],[max_samples,1]),tf.slice(xyz,[0,0],[max_samples,1])),2.0*np.pi)/(2.0*np.pi)
        # convert resaled 0-1 dec from unit vector
        samp_dec = (tf.asin(tf.slice(xyz,[0,2],[max_samples,1])) + 0.5*np.pi)/np.pi
        # group the sky samples 
        samples = tf.reshape(tf.concat([samp_ra,samp_dec],axis=1),[max_samples,2])

        return samples


class JointXYZ:

    def __init__(self, pars, config):
        """                                                                                                                                  
        !!!! Not working do not use!!!!!!                                                                                                    
        Joint distribution for sjy and distance, i.e. sample in postion in space and convery back to distance and sky                        
        """
        self.name = "JointXYZ"
        self.config = config
        self.pars = pars

        order = ["ra", "dec", "luminosity_distance"]
        self.order_adl = []
        self.inorder = []
        for i,p in enumerate(order):
            for j,p1 in enumerate(pars):
                if p == p1:
                    self.order_adl.append(j)
        for i,p in enumerate(pars):
            for j,p1 in enumerate(order):
                if p == p1:
                    self.inorder.append(j)

        self.num_pars = len(self.pars)
        if self.num_pars != 3:
            raise Exception("Please only use three variables for JointXYZ")
        self.num_outputs = [self.num_pars,self.num_pars]
        self.fourpisq = 4.0*np.pi*np.pi
        self.lntwopi = tf.math.log(2.0*np.pi)
        self.lnfourpi = tf.math.log(4.0*np.pi)
        
    def get_distribution(self, mean, logvar, ramp = 1.0):
        mean_x, mean_y, mean_z = tf.split(mean, num_or_size_splits=3, axis=1)
        logvarx, logvary, logvarz = tf.split(logvar, num_or_size_splits=3, axis=1)

        joint = tfp.distributions.JointDistributionNamed(dict(
            x = tfp.distributions.TruncatedNormal(
                loc=tf.cast(mean_x,dtype=tf.float32),
                scale=tf.cast(tf.sqrt(tf.exp(logvarx)),dtype=tf.float32),
                low=-1, high=1),
            y = lambda x: tfp.distributions.TruncatedNormal(
                loc=tf.cast(mean_y,dtype=tf.float32),
                scale=tf.cast(tf.sqrt(tf.exp(logvary)),dtype=tf.float32),
                low=-tf.sqrt(1 - x*x),
                high=tf.sqrt(1 - x*x)),
            z = lambda x, y: tfp.distributions.TruncatedNormal(
                loc=tf.cast(mean_z,dtype=tf.float32),
                scale=tf.cast(tf.sqrt(tf.exp(logvarz)),dtype=tf.float32),
                low=-tf.sqrt(1 - y*y - x*x),
                high=tf.sqrt(1 - y*y - x*x)),

        ))
        return joint
    
    def get_networks(self,logvar_activation="none"):
        # setup network for joint sitribution                                                                                               
        mean =  tf.keras.layers.Dense(3,activation='sigmoid', use_bias = True, name="{}_mean".format(self.name))
        logvar = tf.keras.layers.Dense(3,use_bias=True, name="{}_logvar".format(self.name),activation = logvar_activation)
        return mean, logvar
    
    def get_cost(self, dist, x):
        # reverse order of true parmaeters to estimate logprob                                                                               
        xs = tf.split(x, num_or_size_splits=3, axis=1)
        ra_sky = tf.reshape(2*np.pi*xs[self.order_adl[0]],(-1,1))       # convert the scaled 0->1 true RA value back to radians                
        dec_sky = tf.reshape(np.pi*(xs[self.order_adl[1]] - 0.5),(-1,1)) # convert the scaled 0>1 true dec value back to radians
        xyz = {}
        xyz["x"] = xs[self.order_adl[2]]*tf.cos(ra_sky)*tf.cos(dec_sky)
        xyz["y"] = xs[self.order_adl[2]]*tf.sin(ra_sky)*tf.cos(dec_sky)
        xyz["z"] = xs[self.order_adl[2]]*tf.sin(dec_sky)   # construct the true parameter unit vector                                                             
        #normed_xyz = tf.math.l2_normalize(xyz_unit,axis=1) # normalise x,y,z coords to r                                                    
        # get log prob        
        
        cost_recon = -1.0*tf.reduce_mean(self.lnfourpi + dist.log_prob(xyz),axis=0)
        return cost_recon

    def sample(self, dist, max_samples):
        # transpose amd squeeze to get [samples, parameters]       
        smp = dist.sample()
        xyz_samples = tf.concat([smp["x"], smp["y"], smp["z"]], axis = 1)
        #xyz_samples = tf.gather(smp, indices=self.reorder, axis = 1)
        x,y,z = tf.split(xyz_samples, num_or_size_splits=3, axis=1)
        dl = tf.sqrt(x*x + y*y + z*z)
        dec_sky = tf.acos(z/tf.sqrt(dl))
        ra_sky = tf.atan(y/x)
        # convert ra back depending on x and y
        add_pi_ind = tf.squeeze(tf.where(tf.logical_and(tf.squeeze(x) < 0,tf.squeeze(y) >= 0)))
        minus_pi_ind = tf.squeeze(tf.where(tf.logical_and(tf.squeeze(x) < 0,tf.squeeze(y) < 0)))
        ra_sky = ra_sky.numpy()
        ra_sky[add_pi_ind.numpy()] += np.pi
        ra_sky[minus_pi_ind.numpy()] -= np.pi
        # shift ra to between 0 and 2pi
        ra_sky = tf.convert_to_tensor(ra_sky) + np.pi
        adl = tf.concat([ra_sky/(2*np.pi), dec_sky/np.pi, dl], axis = 1)
        return tf.gather(adl, indices=self.inorder, axis = 1)



