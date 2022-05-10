import tensorflow as tf
from tensorflow.keras import regularizers
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import time 
import matplotlib.pyplot as plt
import os
import sys

class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, config, verbose = False):
        super(CVAE, self).__init__()
        self.config = config
        self.z_dim = self.config["model"]["z_dimension"]
        self.n_modes = self.config["model"]["n_modes"]
        self.x_modes = 1   # hardcoded for testing
        self.x_dim_periodic = int(self.config["masks"]["periodic_len"])
        self.x_dim_nonperiodic = int(self.config["masks"]["nonperiodic_len"])
        if "ra" in config["model"]["inf_pars_list"]:
            self.x_dim_sky = int(2)
        else:
            self.x_dim_sky = 0
        self.x_dim = self.x_dim_periodic + self.x_dim_nonperiodic + self.x_dim_sky
        self.y_dim = self.config["data"]["sampling_frequency"]*self.config["data"]["duration"]
        self.n_channels = len(self.config["data"]["detectors"])
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.activation_relu = tf.keras.layers.ReLU()
        self.kernel_initializer = "glorot_uniform"#tf.keras.initializers.HeNormal() 
        self.bias_initializer = "zeros"#tf.keras.initializers.HeNormal() 

        self.verbose = verbose
        # consts
        self.EPS = 1e-3
        self.fourpisq = 4.0*np.pi*np.pi
        self.lntwopi = tf.math.log(2.0*np.pi)
        self.lnfourpi = tf.math.log(4.0*np.pi)
        # variables
        self.ramp = tf.Variable(0.0, trainable=False)
        self.initial_learning_rate = self.config["training"]["initial_learning_rate"]

        self.total_loss_metric = tf.keras.metrics.Mean('total_loss', dtype=tf.float32)
        self.recon_loss_metric = tf.keras.metrics.Mean('recon_loss', dtype=tf.float32)
        self.kl_loss_metric = tf.keras.metrics.Mean('KL_loss', dtype=tf.float32)
        self.gauss_loss_metric = tf.keras.metrics.Mean('gauss_loss', dtype=tf.float32)
        self.vm_loss_metric = tf.keras.metrics.Mean('vm_loss', dtype=tf.float32)

        self.val_total_loss_metric = tf.keras.metrics.Mean('val_total_loss', dtype=tf.float32)
        self.val_recon_loss_metric = tf.keras.metrics.Mean('val_recon_loss', dtype=tf.float32)
        self.val_kl_loss_metric = tf.keras.metrics.Mean('val_KL_loss', dtype=tf.float32)

        # the shared convolutional network
        all_input_y = tf.keras.Input(shape=(self.y_dim, self.n_channels))
        conv = self.get_network(all_input_y, self.config["model"]["shared_network"])
        conv = tf.keras.layers.Flatten()(conv)

        # r1 encoder network
        r1 = self.get_network(conv, self.config["model"]["output_network"])
        r1 = tf.keras.layers.Dense(2*self.z_dim*self.n_modes + self.n_modes)(r1)
        self.encoder_r1 = tf.keras.Model(inputs=all_input_y, outputs=r1)

        # the q encoder network
        q_input_x = tf.keras.Input(shape=(self.x_dim))
        q_inx = tf.keras.layers.Flatten()(q_input_x)
        q = tf.keras.layers.concatenate([conv,q_inx])
        q = self.get_network(q, self.config["model"]["output_network"])
        q_mean = tf.keras.layers.Dense(self.z_dim)(q)
        q_logvar = tf.keras.layers.Dense(self.z_dim, activation = self.activation)(q)
        q = tf.keras.layers.concatenate([q_mean, q_logvar])
        self.encoder_q = tf.keras.Model(inputs=[all_input_y, q_input_x], outputs=q)

        # the r2 decoder network
        r2_input_z = tf.keras.Input(shape=(self.z_dim))
        r2_inz = tf.keras.layers.Flatten()(r2_input_z)
        r2 = tf.keras.layers.concatenate([conv,r2_inz])
        r2 = self.get_network(r2, self.config["model"]["output_network"])

        # non periodic outputs
        r2mu_nonperiodic = tf.keras.layers.Dense(self.x_dim_nonperiodic,activation='sigmoid')(r2)
        r2logvar_nonperiodic = tf.keras.layers.Dense(self.x_dim_nonperiodic,use_bias=True)(r2)

        # periodic outputs
        r2mu_periodic = tf.keras.layers.Dense(2*self.x_dim_periodic,use_bias=True)(r2)
        r2logvar_periodic = tf.keras.layers.Dense(self.x_dim_periodic,use_bias=True)(r2)

        # sky outputs (x,y,z)
        r2mu_sky = tf.keras.layers.Dense(3,use_bias=True)(r2)
        r2logvar_sky = tf.keras.layers.Dense(1,use_bias=True)(r2)
        # all outputs
        r2 = tf.keras.layers.concatenate([r2mu_nonperiodic,r2mu_periodic,r2mu_sky,r2logvar_nonperiodic,r2logvar_periodic,r2logvar_sky])

        self.decoder_r2 = tf.keras.Model(inputs=[all_input_y, r2_input_z], outputs=r2)



    def encode_r1(self, y=None):
        mean, logvar, weight = tf.split(self.encoder_r1(y), num_or_size_splits=[self.z_dim*self.n_modes, self.z_dim*self.n_modes,self.n_modes], axis=1)
        return tf.reshape(mean,[-1,self.n_modes,self.z_dim]), tf.reshape(logvar,[-1,self.n_modes,self.z_dim]), tf.reshape(weight,[-1,self.n_modes])

    def encode_q(self, x=None, y=None):
        return tf.split(self.encoder_q([y,x]), num_or_size_splits=[self.z_dim,self.z_dim], axis=1)

    def decode_r2(self, y=None, z=None, apply_sigmoid=False):
        return tf.split(self.decoder_r2([y,z]), num_or_size_splits=[self.x_dim_nonperiodic, 2*self.x_dim_periodic, self.x_dim_sky + 1, self.x_dim_nonperiodic, self.x_dim_periodic, 1], axis=1)
        #mean, logvar, weight = tf.split(self.decoder_r2([y,z]), num_or_size_splits=[self.x_dim*self.x_modes, self.x_dim*self.x_modes,self.x_modes], axis=1)
        #return tf.reshape(mean,[-1,self.x_modes,self.x_dim]), tf.reshape(logvar,[-1,self.x_modes,self.x_dim]), tf.reshape(weight,[-1,self.x_modes])


    @property
    def metrics(self):
        return [
            self.total_loss_metric,
            self.recon_loss_metric,
            self.kl_loss_metric,
            self.gauss_loss_metric,
            self.vm_loss_metric,
            self.val_total_loss_metric,
            self.val_recon_loss_metric,
            self.val_kl_loss_metric,
        ]

    #@tf.function
    def train_step(self, data):
        """Executes one training step and returns the loss.
        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        
        with tf.GradientTape() as tape:
            r_loss, kl_loss = self.compute_loss(data[1], data[0], self.ramp)
            loss = r_loss + self.ramp*kl_loss
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.total_loss_metric.update_state(loss)
        self.recon_loss_metric.update_state(r_loss)
        self.kl_loss_metric.update_state(kl_loss)
        #self.gauss_loss_metric.update_state(g_loss)
        #self.vm_loss_metric.update_state(vm_loss)

        #return r_loss, kl_loss
        return {"total_loss":self.total_loss_metric.result(),
                "recon_loss":self.recon_loss_metric.result(),
                "kl_loss":self.kl_loss_metric.result()}
                #"gauss_loss":self.gauss_loss_metric.result(),
                #"vm_loss":self.vm_loss_metric.result()}

        return r_loss, kl_loss, g_loss, vm_loss, mean_r2, scale_r2, truths, gcost

    def test_step(self, data):
        """Executes one test step and returns the loss.                                                        
        This function computes the loss and gradients (used for validation data)
        """
        
        #r_loss, kl_loss,g_loss, vm_loss, mean_r2, scale_r2, truths, gcost = self.compute_loss(data[1], data[0])
        r_loss, kl_loss  = self.compute_loss(data[1], data[0])
        loss = r_loss + self.ramp*kl_loss

        self.val_total_loss_metric.update_state(loss)
        self.val_recon_loss_metric.update_state(r_loss)
        self.val_kl_loss_metric.update_state(kl_loss)

        #return r_loss, kl_loss
        return {"total_loss":self.val_total_loss_metric.result(),
                "recon_loss":self.val_recon_loss_metric.result(),
                "kl_loss":self.val_kl_loss_metric.result()}

    def compute_loss(self, x, y, noiseamp=1.0, ramp = 1.0):
        
        # Recasting some things to float32
        noiseamp = tf.cast(noiseamp, dtype=tf.float32)

        y = tf.cast(y, dtype=tf.float32)
        y = tf.keras.activations.tanh(y)
        x = tf.cast(x, dtype=tf.float32)
        
        mean_r1, logvar_r1, logweight_r1 = self.encode_r1(y=y)
        scale_r1 = self.EPS + tf.sqrt(tf.exp(logvar_r1))
        gm_r1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logweight_r1),
                                      components_distribution=tfd.MultivariateNormalDiag(
                                          loc=mean_r1,
                                          scale_diag=scale_r1))

        mean_q, logvar_q = self.encode_q(x=x,y=y)
        scale_q = self.EPS + tf.sqrt(tf.exp(-logvar_q))
        mvn_q = tfp.distributions.MultivariateNormalDiag(
            loc=mean_q,
            scale_diag=scale_q)
        #mvn_q = tfd.Normal(loc=mean_q,scale=scale_q)
        z_samp = mvn_q.sample()


        mean_nonperiodic_r2, mean_periodic_r2, mean_sky_r2, logvar_nonperiodic_r2, logvar_periodic_r2, logvar_sky_r2 = self.decode_r2(z = z_samp, y=y)

        
        Root = tfp.distributions.JointDistributionCoroutine.Root 
        def model():
            md1 = yield Root(tfp.distributions.TruncatedNormal(
                loc=tf.cast(tf.boolean_mask(mean_nonperiodic_r2,self.config["masks"]["nonperiodic_m1_mask"],axis=1),dtype=tf.float32),
                scale=tf.cast(tf.boolean_mask(self.EPS + tf.sqrt(tf.exp(logvar_nonperiodic_r2)),self.config["masks"]["nonperiodic_m1_mask"],axis=1),dtype=tf.float32),
                low=-10.0 + self.ramp*10.0, high=1.0 + 10.0 - self.ramp*10.0))
            
            md2 = yield tfp.distributions.TruncatedNormal(
                loc=tf.cast(tf.boolean_mask(mean_nonperiodic_r2,self.config["masks"]["nonperiodic_m2_mask"],axis=1),dtype=tf.float32),
                scale=tf.cast(tf.boolean_mask(self.EPS + tf.sqrt(tf.exp(logvar_nonperiodic_r2)),self.config["masks"]["nonperiodic_m2_mask"],axis=1),dtype=tf.float32),
                low=-10.0 + self.ramp*10.0, high=md1)
            
        joint_trunc_m1m2 = tfp.distributions.JointDistributionCoroutine(model)
        
        # truncated normal for non-periodic params                      
        tmvn_np_r2 = tfp.distributions.TruncatedNormal(
            loc=tf.cast(tf.boolean_mask(mean_nonperiodic_r2,self.config["masks"]["nonperiodicpars_nonm1m2_mask"],axis=1),dtype=tf.float32),
            scale=tf.cast(tf.boolean_mask(self.EPS + tf.sqrt(tf.exp(logvar_nonperiodic_r2)),self.config["masks"]["nonperiodicpars_nonm1m2_mask"],axis=1),dtype=tf.float32),
            low=-10.0 + self.ramp*10.0, high=1.0 + 10.0 - self.ramp*10.0)

        
        tmvn_r2_cost_recon = -1.0*tf.reduce_mean(tf.reduce_sum(tmvn_np_r2.log_prob(tf.boolean_mask(x,self.config["masks"]["nonperiodic_nonm1m2_mask"],axis=1)),axis=1),axis=0)
        m1m2_r2_cost_recon = -1.0*tf.reduce_mean(tf.reduce_sum(joint_trunc_m1m2.log_prob(tf.boolean_mask(x,self.config["masks"]["mass_1_mask"],axis=1),tf.boolean_mask(x,self.config["masks"]["mass_2_mask"],axis=1)),axis=1),axis=0)

        # 2D representations of periodic params
        # split the means into x and y coords [size (batch,p) each]
        mean_x_r2, mean_y_r2 = tf.split(mean_periodic_r2, num_or_size_splits=2, axis=1)
        # the mean angle (0,2pi) [size (batch,p)]
        mean_angle_r2 = tf.math.floormod(tf.math.atan2(tf.cast(mean_y_r2,dtype=tf.float32),tf.cast(mean_x_r2,dtype=tf.float32)),2.0*np.pi)

        # define the 2D Gaussian scale [size (batch,2*p)]
        vm_r2 = tfp.distributions.VonMises(
            loc=tf.cast(mean_angle_r2,dtype=tf.float32),
            concentration=tf.cast(tf.math.reciprocal(self.EPS + self.fourpisq*tf.exp(logvar_periodic_r2)),dtype=tf.float32)
        )
        # get log prob of periodic params
        vm_r2_cost_recon = -1.0*tf.reduce_mean(tf.reduce_sum(self.lntwopi + vm_r2.log_prob(2.0*np.pi*tf.boolean_mask(x,self.config["masks"]["periodic_mask"],axis=1)),axis=1),axis=0)


        # define the von mises fisher for the sky
        fvm_loc = tf.reshape(tf.math.l2_normalize(mean_sky_r2, axis=1),[-1,3])  # mean in (x,y,z)
        fvm_con = tf.reshape(tf.math.reciprocal(self.EPS + tf.exp(logvar_sky_r2)),[-1])
        fvm_r2 = tfp.distributions.VonMisesFisher(
            mean_direction = tf.cast(fvm_loc,dtype=tf.float32),
            concentration = tf.cast(fvm_con,dtype=tf.float32)
        )

        ra_sky = tf.reshape(2*np.pi*tf.boolean_mask(x,self.config["masks"]["ra_mask"],axis=1),(-1,1))       # convert the scaled 0->1 true RA value back to radians
        dec_sky = tf.reshape(np.pi*(tf.boolean_mask(x,self.config["masks"]["dec_mask"],axis=1) - 0.5),(-1,1)) # convert the scaled 0>1 true dec value back to radians
        xyz_unit = tf.concat([tf.cos(ra_sky)*tf.cos(dec_sky),tf.sin(ra_sky)*tf.cos(dec_sky),tf.sin(dec_sky)],axis=1)   # construct the true parameter unit vector
        normed_xyz = tf.math.l2_normalize(xyz_unit,axis=1) # normalise x,y,z coords to r
        # get log prob
        fvm_r2_cost_recon = -1.0*tf.reduce_mean(self.lnfourpi + fvm_r2.log_prob(normed_xyz),axis=0)


        simple_cost_recon = tmvn_r2_cost_recon + m1m2_r2_cost_recon + vm_r2_cost_recon + fvm_r2_cost_recon

        selfent_q = -1.0*tf.reduce_mean(mvn_q.entropy())
        log_r1_q = gm_r1.log_prob(tf.cast(z_samp,dtype=tf.float32))   # evaluate the log prob of r1 at the q samples
        cost_KL = tf.cast(selfent_q,dtype=tf.float32) - tf.reduce_mean(log_r1_q)

        return simple_cost_recon, cost_KL


    def gen_samples(self, y, ramp=1.0, nsamples=1000, max_samples=1000):
        
        #y = y/self.params['y_normscale']
        y = tf.keras.activations.tanh(y)
        y = tf.tile(y,(max_samples,1,1))
        samp_iterations = int(nsamples/max_samples)
        for i in range(samp_iterations):
            mean_r1, logvar_r1, logweight_r1 = self.encode_r1(y=y)
            scale_r1 = self.EPS + tf.sqrt(tf.exp(logvar_r1))
            gm_r1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logweight_r1),
                                          components_distribution=tfd.MultivariateNormalDiag(
                                              loc=mean_r1,
                                              scale_diag=scale_r1))
            z_samp = gm_r1.sample()

            mean_nonperiodic_r2_temp, mean_periodic_r2_temp, mean_sky_r2_temp, logvar_nonperiodic_r2_temp, logvar_periodic_r2_temp, logvar_sky_r2_temp = self.decode_r2(z=z_samp,y=y)

            Root = tfp.distributions.JointDistributionCoroutine.Root 
            def model():
                md1 = yield Root(tfp.distributions.TruncatedNormal(
                    loc=tf.cast(tf.boolean_mask(mean_nonperiodic_r2_temp,self.config["masks"]["nonperiodic_m1_mask"],axis=1),dtype=tf.float32),
                    scale=tf.cast(tf.boolean_mask(self.EPS + tf.sqrt(tf.exp(logvar_nonperiodic_r2_temp)),self.config["masks"]["nonperiodic_m1_mask"],axis=1),dtype=tf.float32),
                    low=-10.0 + self.ramp*10.0, high=1.0 + 10.0 - self.ramp*10.0))
                
                md2 = yield tfp.distributions.TruncatedNormal(
                    loc=tf.cast(tf.boolean_mask(mean_nonperiodic_r2_temp,self.config["masks"]["nonperiodic_m2_mask"],axis=1),dtype=tf.float32),
                    scale=tf.cast(tf.boolean_mask(self.EPS + tf.sqrt(tf.exp(logvar_nonperiodic_r2_temp)),self.config["masks"]["nonperiodic_m2_mask"],axis=1),dtype=tf.float32),
                    low=-10.0 + self.ramp*10.0, high=md1)
                
            m1m2_nonperiodic_r2 = tfp.distributions.JointDistributionCoroutine(model)

            tmvn_nonperiodic_r2 = tfp.distributions.TruncatedNormal(
                loc=tf.cast(tf.boolean_mask(mean_nonperiodic_r2_temp,self.config["masks"]["nonperiodicpars_nonm1m2_mask"],axis=1),dtype=tf.float32),
                scale=tf.cast(tf.boolean_mask(self.EPS + tf.sqrt(tf.exp(logvar_nonperiodic_r2_temp)),self.config["masks"]["nonperiodicpars_nonm1m2_mask"],axis=1),dtype=tf.float32),
                low=-10.0 + self.ramp*10.0, high=1.0 + 10.0 - self.ramp*10.0)
            
            # periodic z,y, params for vonmises
            mean_x_r2, mean_y_r2 = tf.split(mean_periodic_r2_temp, num_or_size_splits=2, axis=1)
            mean_angle_r2 = tf.math.floormod(tf.math.atan2(tf.cast(mean_y_r2,dtype=tf.float32),tf.cast(mean_x_r2,dtype=tf.float32)),2.0*np.pi)

            # Von Mises
            vm_r2 = tfp.distributions.VonMises(
                loc=tf.cast(mean_angle_r2,dtype=tf.float32),
                concentration=tf.cast(tf.math.reciprocal(self.EPS + self.fourpisq*tf.exp(logvar_periodic_r2_temp)),dtype=tf.float32)
            )

            # sky with Fisher Von Mises
            fvm_loc = tf.reshape(tf.math.l2_normalize(mean_sky_r2_temp, axis=1),[max_samples,3])
            fvm_con = tf.reshape(tf.math.reciprocal(self.EPS + tf.exp(logvar_sky_r2_temp)),[max_samples])
            fvm_r2 = tfp.distributions.VonMisesFisher(
                mean_direction = tf.cast(fvm_loc,dtype=tf.float32),
                concentration = tf.cast(fvm_con,dtype=tf.float32)
            )

            tmvn_x_sample = tmvn_nonperiodic_r2.sample()
            m1m2_x_sample = tf.concat(m1m2_nonperiodic_r2.sample(), axis = 1)

            # periodic sample, samples are output on range [-pi,pi] so rescale to [0,1]
            vm_x_sample = tf.math.floormod(vm_r2.sample(),(2.0*np.pi))/(2.0*np.pi)
            # sky sample
            xyz = tf.reshape(fvm_r2.sample(),[max_samples,3])
            # convert to rescaled 0-1 ra from the unit vector
            samp_ra = tf.math.floormod(tf.math.atan2(tf.slice(xyz,[0,1],[max_samples,1]),tf.slice(xyz,[0,0],[max_samples,1])),2.0*np.pi)/(2.0*np.pi)
            # convert resaled 0-1 dec from unit vector
            samp_dec = (tf.asin(tf.slice(xyz,[0,2],[max_samples,1])) + 0.5*np.pi)/np.pi
            # group the sky samples 
            fvm_x_sample = tf.reshape(tf.concat([samp_ra,samp_dec],axis=1),[max_samples,2])

            
            if i==0:
                x_sample = tf.gather(tf.concat([m1m2_x_sample,tmvn_x_sample,vm_x_sample,fvm_x_sample],axis=1),self.config["masks"]["idx_periodic_mask"],axis=1)
            else:
                x_sample = tf.concat([x_sample,tf.gather(tf.concat([m1m2_x_sample,tmvn_x_sample,vm_x_sample,fvm_x_sample],axis=1),self.config["masks"]["idx_periodic_mask"],axis=1)], axis = 0)

        return x_sample


    def gen_z_samples(self, x, y, nsamples=1000):

        #y = y/self.params['y_normscale']
        y = tf.keras.activations.tanh(y)
        y = tf.tile(y,(nsamples,1,1))
        x = tf.tile(x,(nsamples,1))
        mean_r1, logvar_r1, logweight_r1 = self.encode_r1(y=y)
        scale_r1 = self.EPS + tf.sqrt(tf.exp(logvar_r1))
        gm_r1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logweight_r1),
                                      components_distribution=tfd.MultivariateNormalDiag(
                                          loc=mean_r1,
                                          scale_diag=scale_r1))
        z_samp_r1 = gm_r1.sample()
        mean_q, logvar_q = self.encode_q(x=x,y=y)
        scale_q = self.EPS + tf.sqrt(tf.exp(-logvar_q))
        mvn_q = tfp.distributions.MultivariateNormalDiag(
            loc=mean_q,
            scale_diag=scale_q)
        z_samp_q = mvn_q.sample()

        return mean_r1, z_samp_r1, mean_q, z_samp_q, scale_r1, scale_q, logvar_q

    def call(self, inputs):
        '''
        call the function generates one sample of output (only here for the build section)
        '''
        
        # encode through r1 network                          
        mean_r1, logvar_r1, logweight_r1 = self.encode_r1(y=inputs)
        scale_r1 = self.EPS + tf.sqrt(tf.exp(logvar_r1))

        gm_r1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logweight_r1),
                                      components_distribution=tfd.MultivariateNormalDiag(
                                          loc=mean_r1,
                                          scale_diag=scale_r1))

        z_samp = gm_r1.sample()

        mean_nonperiodic_r2, mean_periodic_r2, mean_sky_r2, logvar_nonperiodic_r2, logvar_periodic_r2, logvar_sky_r2 = self.decode_r2(z=z_samp,y=inputs)

        means = tf.concat([mean_nonperiodic_r2, mean_periodic_r2, mean_sky_r2], axis = 1)
        return means

    def get_network(self, in_tensor, layers):
        
        conv = in_tensor
        for layer in layers:
            if layer.split("(")[0] == "Conv1D":
                nfilters, filter_size, stride = layer.split("(")[1].strip(")").split(",")
                conv = self.ConvBlock(conv, int(nfilters), int(filter_size), int(stride))
            if layer.split("(")[0] == "ResBlock":
                nfilters, filter_size, stride = layer.split("(")[1].strip(")").split(",")
                # add a bottleneck block
                conv = self.ResBlock(conv, [int(nfilters), int(nfilters), int(nfilters)], [1, int(filter_size)], int(stride))
            elif layer.split("(")[0] == "Linear":
                num_neurons = layer.split("(")[1].strip(")")
                conv = self.LinearBlock(conv, int(num_neurons))
            else:
                print("No layers saved")

        return conv

    def ConvBlock(self, input_data, filters, kernel_size, strides):
        conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, kernel_regularizer=regularizers.l2(0.001), padding="same", kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)(input_data)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = self.activation(conv)

        return conv

    def LinearBlock(self,input_data, num_neurons):

        out = tf.keras.layers.Dense(num_neurons, kernel_regularizer=regularizers.l2(0.001), activation=self.activation, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)(input_data)
        out = tf.keras.layers.BatchNormalization()(out)
        return out

    def ResBlock(self,input_data, filters, kernel_size, strides):
        filters1, filters2, filters3 = filters
        kernel_size1,kernel_size2 = kernel_size
        
        conv_short = input_data
        
        conv = tf.keras.layers.Conv1D(filters=filters1, kernel_size=kernel_size1, strides=strides, kernel_regularizer=regularizers.l2(0.001), padding="same", kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)(input_data)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = self.activation(conv)
        
        conv = tf.keras.layers.Conv1D(filters=filters2, kernel_size=kernel_size2, kernel_regularizer=regularizers.l2(0.001), padding="same", kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = self.activation(conv)
        
        conv = tf.keras.layers.Conv1D(filters=filters3, kernel_size=kernel_size1, kernel_regularizer=regularizers.l2(0.001), padding="same", kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        #conv = self.act(conv)
        
        if strides > 1:
            conv_short = tf.keras.layers.Conv1D(filters=filters3, kernel_size=kernel_size1, strides=strides, kernel_regularizer=regularizers.l2(0.001), padding="same", kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)(conv_short)
            conv_short = tf.keras.layers.BatchNormalization()(conv_short)
            #conv_short = self.act(conv_short)                                                                     

        conv = tf.keras.layers.Add()([conv, conv_short])
        conv = self.activation(conv)
        return conv



    def ResBlock2d(self,input_y, filters, kernel_size, stride=2, activation=None):
        
        F1, F2, F3 = filters
        
        y_shortcut = input_y
        
        y = tf.keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(stride, stride), padding='valid')(input_y)
        y =tf.keras.layers.BatchNormalization(axis=3)(y)
        y = self.activation(y)
        
        y = tf.keras.layers.Conv2D(filters=F2, kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same')(y)
        y = tf.keras.layers.BatchNormalization(axis=3)(y)
        y = self.activation(y)
        
        y = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(y)
        y = tf.keras.layers.BatchNormalization(axis=3)(y)
        
        if stride > 1:
            y_shortcut = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(stride, stride), padding='valid')(y_shortcut)
            y_shortcut = tf.keras.layers.BatchNormalization(axis=3)(y_shortcut)

        y = tf.keras.layers.Add()([y, y_shortcut])
        y = self.activation(y)
        
        return y
