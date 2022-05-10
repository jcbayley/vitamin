import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class JointM1M2:

    def __init__(self, pars, config):
        self.name = "JointM1M2"
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
        mean1, mean2 = tf.split(mean, num_or_size_splits=2, axis=1)
        logvar1, logvar2 = tf.split(logvar, num_or_size_splits=2, axis=1)

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
        mean =  tf.keras.layers.Dense(2,activation='sigmoid', use_bias = True, name="{}_mean".format(self.name))
        logvar = tf.keras.layers.Dense(2,use_bias=True, name="{}_logvar".format(self.name), activation=logvar_activation)
        return mean, logvar

    def cost_setup(self):
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

    def __init__(self, pars, config):
        """
        Joint distribution for Chirpmass and symetric mas ratio
        """
        self.name = "JointChirpmassMR"
        self.config = config
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
        q = q_norm*(self.config["bounds"]["mass_ratio_max"] - self.config["bounds"]["mass_ratio_min"]) + self.config["bounds"]["mass_ratio_min"]
        num = (q*m*m)**(3./5.)
        den = (m*(1 + q))**(1./5.)
        return (num/den - self.config["bounds"]["chirp_mass_min"])/(self.config["bounds"]["chirp_mass_max"] - self.config["bounds"]["chirp_mass_min"])

    def Mconstrainm2(self, q_norm, m):
        """Contraint for the chirp mass based on minimum mass of m2"""
        q = q_norm*(self.config["bounds"]["mass_ratio_max"] - self.config["bounds"]["mass_ratio_min"]) + self.config["bounds"]["mass_ratio_min"]
        num = ((1/q)*m*m)**(3./5.)
        den = (m*(1 + 1/q))**(1./5.)
        return (num/den - self.config["bounds"]["chirp_mass_min"])/(self.config["bounds"]["chirp_mass_max"] - self.config["bounds"]["chirp_mass_min"])

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
                low=self.Mconstrainm2(b0, self.config["bounds"]["mass_1_min"]), 
                high=self.Mconstrainm1(b0, self.config["bounds"]["mass_1_max"])),
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


class TruncatedNormal:

    def __init__(self, pars, config):
        self.name = "TruncatedNormal"
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

class VonMises:

    def __init__(self, pars, config):
        self.name = "VonMises"
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
            concentration=tf.cast(tf.math.reciprocal(self.fourpisq*tf.exp(logvar)),dtype=tf.float32))
        
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

    def __init__(self, pars, config):
        self.name = "JointVonMisesFisher"
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

    def __init__(self, pars, config):
        self.name = "JointPowerSpherical"
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
        if self.pars[0] == "alpha":
            self.order_flipped = True
        elif self.pars[0] == "luminosity_distance":
            self.order_flipped = False

        self.num_pars = len(self.pars)
        if self.num_pars != 3:
            raise Exception("Please only use three variables for JointXYZ")
        self.num_outputs = [self.num_pars,self.num_pars]
        self.get_cost = self.cost_setup()
        self.sample = self.sample_setup()

    def Mconstrainm1(self, q_norm, m):
        """Contraint for the chirp mass based on maxmium mass of m1
        """
        q = q_norm*(self.config["bounds"]["mass_ratio_max"] - self.config["bounds"]["mass_ratio_min"]) + self.config["bounds"]["mass_ratio_min"]
        num = (q*m*m)**(3./5.)
        den = (m*(1 + q))**(1./5.)
        return (num/den - self.config["bounds"]["chirp_mass_min"])/(self.config["bounds"]["chirp_mass_max"] - self.config["bounds"]["chirp_mass_min"])

    def Mconstrainm2(self, q_norm, m):
        """Contraint for the chirp mass based on minimum mass of m2"""
        q = q_norm*(self.config["bounds"]["mass_ratio_max"] - self.config["bounds"]["mass_ratio_min"]) + self.config["bounds"]["mass_ratio_min"]
        num = ((1/q)*m*m)**(3./5.)
        den = (m*(1 + 1/q))**(1./5.)
        return (num/den - self.config["bounds"]["chirp_mass_min"])/(self.config["bounds"]["chirp_mass_max"] - self.config["bounds"]["chirp_mass_min"])

    def get_distribution(self, mean, logvar, ramp = 1.0):
        # this is not working yet
        mean_q, mean_cm = tf.split(mean, num_or_size_splits=2, axis=1)
        logvarq, logvarcm = tf.split(logvar, num_or_size_splits=2, axis=1)

        joint = tfp.distributions.JointDistributionSequential([
            tfp.distributions.TruncatedNormal(
                loc=tf.cast(mean_q,dtype=tf.float32),
                scale=tf.cast(tf.sqrt(tf.exp(logvarq)),dtype=tf.float32),
                low=0, high=1),
            lambda y: tfp.distributions.TruncatedNormal(
                loc=tf.cast(mean_cm,dtype=tf.float32),
                scale=tf.cast(tf.sqrt(tf.exp(logvarcm)),dtype=tf.float32),
                low=self.Mconstrainm2(b0, self.config["bounds"]["mass_1_min"]), 
                high=self.Mconstrainm1(b0, self.config["bounds"]["mass_1_max"])),
            lambda z: tfp.distributions.TruncatedNormal(
                loc=tf.cast(mean_cm,dtype=tf.float32),
                scale=tf.cast(tf.sqrt(tf.exp(logvarcm)),dtype=tf.float32),
                low=self.Mconstrainm2(b0, self.config["bounds"]["mass_1_min"]), 
                high=self.Mconstrainm1(b0, self.config["bounds"]["mass_1_max"])),

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
                x_ra, x_dec, x_dist = tf.split(x, num_or_size_splits=3, axis=1)
                ra_sky = tf.reshape(2*np.pi*x_ra,(-1,1))       # convert the scaled 0->1 true RA value back to radians
                dec_sky = tf.reshape(np.pi*(x_dec - 0.5),(-1,1)) # convert the scaled 0>1 true dec value back to radians
                xyz_unit = tf.concat([x_dist*tf.cos(ra_sky)*tf.cos(dec_sky),x_dist*tf.sin(ra_sky)*tf.cos(dec_sky),x_dist*tf.sin(dec_sky)],axis=1)   # construct the true parameter unit vector
                normed_xyz = tf.math.l2_normalize(xyz_unit,axis=1) # normalise x,y,z coords to r
                # get log prob
                fvm_r2_cost_recon = -1.0*tf.reduce_mean(self.lnfourpi + dist.log_prob(normed_xyz),axis=0)

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
