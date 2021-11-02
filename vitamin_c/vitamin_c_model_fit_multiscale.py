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

    def __init__(self, params, bounds, masks):
        super(CVAE, self).__init__()
        self.z_dim = params["z_dimension"]
        self.n_modes = params["n_modes"]
        self.x_modes = 1   # hardcoded for testing
        self.x_dim_periodic = int(masks["periodic_len"])
        self.x_dim_nonperiodic = int(masks["nonperiodic_len"])
        self.x_dim_sky = int(2)
        self.x_dim = self.x_dim_periodic + self.x_dim_nonperiodic + self.x_dim_sky
        self.y_dim = params["ndata"]
        self.n_channels = len(params["det"])
        self.act = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.act_relu = tf.keras.layers.ReLU()
        self.params = params
        self.bounds = bounds
        self.masks = masks
        # consts
        self.EPS = 1e-3
        self.fourpisq = 4.0*np.pi*np.pi
        self.lntwopi = tf.math.log(2.0*np.pi)
        self.lnfourpi = tf.math.log(4.0*np.pi)
        # variables
        self.ramp = tf.Variable(0.0, trainable=False)
        self.initial_learning_rate = params["initial_training_rate"]

        self.total_loss_metric = tf.keras.metrics.Mean('total_loss', dtype=tf.float32)
        self.recon_loss_metric = tf.keras.metrics.Mean('recon_loss', dtype=tf.float32)
        self.kl_loss_metric = tf.keras.metrics.Mean('KL_loss', dtype=tf.float32)
        self.gauss_loss_metric = tf.keras.metrics.Mean('gauss_loss', dtype=tf.float32)
        self.vm_loss_metric = tf.keras.metrics.Mean('vm_loss', dtype=tf.float32)

        self.val_total_loss_metric = tf.keras.metrics.Mean('val_total_loss', dtype=tf.float32)
        self.val_recon_loss_metric = tf.keras.metrics.Mean('val_recon_loss', dtype=tf.float32)
        self.val_kl_loss_metric = tf.keras.metrics.Mean('val_KL_loss', dtype=tf.float32)

        # the convolutional network
        all_input_y = tf.keras.Input(shape=(self.y_dim, self.n_channels))
        #all_input_y = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(all_input_y)
        
        # scale 1

        conv = tf.keras.layers.Conv1D(filters=64, kernel_size=32, strides=1, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(all_input_y)
        conv = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(conv)
        #conv = tf.keras.layers.BatchNormalization()(conv)
        
        conv = tf.keras.layers.Conv1D(filters=32, kernel_size=16, strides=1, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(conv)
        conv = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(conv)
        #conv = tf.keras.layers.BatchNormalization()(conv)

        conv = tf.keras.layers.Conv1D(filters=32, kernel_size=8, strides=2, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(conv)
        conv = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(conv)
        #conv = tf.keras.layers.BatchNormalization()(conv)

        conv = tf.keras.layers.Conv1D(filters=16, kernel_size=8, strides=2, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(conv)
        conv = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(conv)
        #conv = tf.keras.layers.BatchNormalization()(conv)

        # scale 2

        conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=32, strides=1, kernel_regularizer=regularizers.l2(0.001), activation=self.act, dilation_rate = 8)(all_input_y)
        conv2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(conv2)
        #conv2 = tf.keras.layers.BatchNormalization()(conv2)
        
        conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=8, strides=1, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(conv2)
        conv2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(conv2)
        #conv2 = tf.keras.layers.BatchNormalization()(conv2)

        conv2 = tf.keras.layers.Conv1D(filters=16, kernel_size=8, strides=1, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(conv2)
        conv2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(conv2)
        #conv2 = tf.keras.layers.BatchNormalization()(conv2)

        # 3rd scale

        conv3 = tf.keras.layers.Conv1D(filters=64, kernel_size=32, strides=1, kernel_regularizer=regularizers.l2(0.001), activation=self.act, dilation_rate = 32)(all_input_y)
        conv3 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(conv3)
        #conv3 = tf.keras.layers.BatchNormalization()(conv3)
        
        #conv3 = tf.keras.layers.Conv1D(filters=32, kernel_size=16, strides=1, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(conv3)
        #conv3 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(conv3)
        #conv3 = tf.keras.layers.BatchNormalization()(conv3)

        conv3 = tf.keras.layers.Conv1D(filters=16, kernel_size=8, strides=1, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(conv3)
        conv3 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(conv3)
        #conv3 = tf.keras.layers.BatchNormalization()(conv3)


        """
        conv = tf.keras.layers.Conv1D(filters=16, kernel_size=1, strides=1, activation=self.act)(conv)
        #conv = tf.keras.layers.BatchNormalization()(conv)
        """

        conv = tf.keras.layers.Flatten()(conv)
        conv2 = tf.keras.layers.Flatten()(conv2)
        conv3 = tf.keras.layers.Flatten()(conv3)
        conv = tf.keras.layers.concatenate([conv,conv2, conv3])        
        #conv = tf.keras.layers.Flatten()(all_input_y)

        lin_layers = 4096, 2048, 1024
        #lin_layers = 2048, 1024, 512, 256
        #lin_layers = 1024, 512, 256, 256
        #lin1, lin2, lin3 = 1024, 512, 256
        #lin1, lin2, lin3 = 512, 256, 128
        # r1 encoder
        r1 = conv
        for lin_l in lin_layers:
            r1 = tf.keras.layers.Dense(lin_l, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(r1)            
            r1 = tf.keras.layers.BatchNormalization()(r1)
            #r1 = tf.keras.layers.Dropout(.5)(r1)

        r1 = tf.keras.layers.Dense(2*self.z_dim*self.n_modes + self.n_modes)(r1)
        self.encoder_r1 = tf.keras.Model(inputs=all_input_y, outputs=r1)
        print(self.encoder_r1.summary())

        # the q encoder network
        q_input_x = tf.keras.Input(shape=(self.x_dim))
        q_inx = tf.keras.layers.Flatten()(q_input_x)
        q = tf.keras.layers.concatenate([conv,q_inx])
        for lin_l in lin_layers:
            q = tf.keras.layers.Dense(lin_l, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(q)            
            q = tf.keras.layers.BatchNormalization()(q)
            #q = tf.keras.layers.Dropout(.5)(q)

        q = tf.keras.layers.Dense(2*self.z_dim)(q)
        self.encoder_q = tf.keras.Model(inputs=[all_input_y, q_input_x], outputs=q)
        print(self.encoder_q.summary())

        # the r2 decoder network
        r2_input_z = tf.keras.Input(shape=(self.z_dim))
        r2_inz = tf.keras.layers.Flatten()(r2_input_z)
        r2 = tf.keras.layers.concatenate([conv,r2_inz])
        for lin_l in lin_layers:
            r2 = tf.keras.layers.Dense(lin_l, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(r2)     
            r2 = tf.keras.layers.BatchNormalization()(r2)
            #r2 = tf.keras.layers.Dropout(.5)(r2)

        extra_layer = False
        if extra_layer:
            exdim = 1024
            # non periodic outputs
            r2_nonp = tf.keras.layers.Dense(exdim, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(r2)
        else:
            r2_nonp = r2
        r2mu_nonperiodic = tf.keras.layers.Dense(self.x_dim_nonperiodic,activation='sigmoid')(r2_nonp)
        r2logvar_nonperiodic = tf.keras.layers.Dense(self.x_dim_nonperiodic,use_bias=True)(r2_nonp)

        # periodic outputs
        if extra_layer:
            r2_p = tf.keras.layers.Dense(exdim, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(r2)     
        else:
            r2_p = r2
        r2mu_periodic = tf.keras.layers.Dense(2*self.x_dim_periodic,use_bias=True)(r2_p)
        r2logvar_periodic = tf.keras.layers.Dense(self.x_dim_periodic,use_bias=True)(r2_p)

        # sky outputs (x,y,z)
        if extra_layer:
            r2_sky = tf.keras.layers.Dense(exdim, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(r2)     
        else:
            r2_sky = r2
        r2mu_sky = tf.keras.layers.Dense(3,use_bias=True)(r2_sky)
        r2logvar_sky = tf.keras.layers.Dense(1,use_bias=True)(r2_sky)
        # all outputs
        r2 = tf.keras.layers.concatenate([r2mu_nonperiodic,r2mu_periodic,r2mu_sky,r2logvar_nonperiodic,r2logvar_periodic,r2logvar_sky])

        #r2_weights = tf.keras.layers.Dense(self.x_modes,use_bias=True,bias_initializer='Zeros')(r2)
        #r2_mu = tf.keras.layers.Dense(self.x_dim*self.x_modes,activation='sigmoid')(r2)
        #r2_logvar = -1.0*tf.keras.layers.Dense(self.x_dim*self.x_modes,activation=self.act_relu,use_bias=True,bias_initializer=tf.keras.initializers.Constant(value=2.0))(r2)
        #r2 = tf.keras.layers.concatenate([r2mu,r2logvar,r2w])
        #r2 = tf.keras.layers.Dense(2*self.x_dim*self.x_modes + self.x_modes)(r2)
        self.decoder_r2 = tf.keras.Model(inputs=[all_input_y, r2_input_z], outputs=r2)
        print(self.decoder_r2.summary())

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
            #r_loss, kl_loss, g_loss, vm_loss, mean_r2, scale_r2, truths, gcost = self.compute_loss(data[1], data[0], self.ramp)
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

        """
        if np.isfinite(loss):
        else:
            print("Inf train loss: rerunning train_step")
        """
        return r_loss, kl_loss, g_loss, vm_loss, mean_r2, scale_r2, truths, gcost

    def test_step(self, data):
        """Executes one test step and returns the loss.                                                        
        This function computes the loss and gradients (used for validation data)
        """
        
        #self.ramp = 0.0
        #r_loss, kl_loss,g_loss, vm_loss, mean_r2, scale_r2, truths, gcost = self.compute_loss(data[1], data[0])
        r_loss, kl_loss  = self.compute_loss(data[1], data[0])
        #print("test_loss", self.ramp, r_loss, kl_loss)
        loss = r_loss + self.ramp*kl_loss

        self.val_total_loss_metric.update_state(loss)
        self.val_recon_loss_metric.update_state(r_loss)
        self.val_kl_loss_metric.update_state(kl_loss)

        #return r_loss, kl_loss
        return {"total_loss":self.val_total_loss_metric.result(),
                "recon_loss":self.val_recon_loss_metric.result(),
                "kl_loss":self.val_kl_loss_metric.result()}

    def mass_sample(self,num_samples, bounds = [10,80]):
        """
        sample from a mass distribution (uniform in lower triangle within bounds)
        """
        scale = bounds[1] - bounds[0]
        unif_1 = tfp.distributions.Uniform(0,1)
        m1 = tf.sqrt(unif_1.sample(num_samples))
        
        unif_2 = tfp.distributions.Uniform(0,m1)
        m2 = unif_2.sample()
        return tf.concat([[m1*scale + bounds[0]],[m2*scale + bounds[0]]], axis = 0)

    def mass_post(self,m1,m2, bounds):
        """
        get posterior for mass
        """
        output = tf.ones(len(m1))/((bounds[1] - bounds[0])**2)
        mass_mask = tf.greater(m2, m1) | tf.less(m1, bounds[0]) | tf.less(m2, bounds[0]) | tf.greater(m1, bounds[1]) | tf.greater(m2, bounds[1])
        infs =  tf.multiply(tf.ones_like(output), tf.constant(0, dtype=tf.float32))
        output = tf.where(mass_mask, infs, output)
        return tf.log(output)

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
        scale_q = self.EPS + tf.sqrt(tf.exp(logvar_q))
        mvn_q = tfp.distributions.MultivariateNormalDiag(
            loc=mean_q,
            scale_diag=scale_q)
        #mvn_q = tfd.Normal(loc=mean_q,scale=scale_q)
        z_samp = mvn_q.sample()


        mean_nonperiodic_r2, mean_periodic_r2, mean_sky_r2, logvar_nonperiodic_r2, logvar_periodic_r2, logvar_sky_r2 = self.decode_r2(z = z_samp, y=y)


        Root = tfp.distributions.JointDistributionCoroutine.Root 
        def model():
            md1 = yield Root(tfp.distributions.TruncatedNormal(
                loc=tf.cast(tf.boolean_mask(mean_nonperiodic_r2,[True, False, False, False, False, False, False, False, False],axis=1),dtype=tf.float32),
                scale=tf.cast(tf.boolean_mask(self.EPS + tf.sqrt(tf.exp(logvar_nonperiodic_r2)),[True, False, False, False, False, False, False, False, False],axis=1),dtype=tf.float32),
                low=-10.0 + self.ramp*10.0, high=1.0 + 10.0 - self.ramp*10.0))
            
            md2 = yield tfp.distributions.TruncatedNormal(
                loc=tf.cast(tf.boolean_mask(mean_nonperiodic_r2,[False, True, False, False, False, False, False, False, False],axis=1),dtype=tf.float32),
                scale=tf.cast(tf.boolean_mask(self.EPS + tf.sqrt(tf.exp(logvar_nonperiodic_r2)),[False, True, False, False, False, False, False, False, False],axis=1),dtype=tf.float32),
                low=-10.0 + self.ramp*10.0, high=md1)
            
        joint_trunc_m1m2 = tfp.distributions.JointDistributionCoroutine(model)
        
        # truncated normal for non-periodic params                      
        tmvn_np_r2 = tfp.distributions.TruncatedNormal(
            loc=tf.cast(tf.boolean_mask(mean_nonperiodic_r2,[False, False, True, True, True, True, True, True, True],axis=1),dtype=tf.float32),
            scale=tf.cast(tf.boolean_mask(self.EPS + tf.sqrt(tf.exp(logvar_nonperiodic_r2)),[False, False, True, True, True, True, True, True, True],axis=1),dtype=tf.float32),
            low=-10.0 + self.ramp*10.0, high=1.0 + 10.0 - self.ramp*10.0)

        
        tmvn_r2_cost_recon = -1.0*tf.reduce_mean(tf.reduce_sum(tmvn_np_r2.log_prob(tf.boolean_mask(x,self.masks["nonperiodic_nonm1m2_mask"],axis=1)),axis=1),axis=0)
        m1m2_r2_cost_recon = -1.0*tf.reduce_mean(tf.reduce_sum(joint_trunc_m1m2.log_prob(tf.boolean_mask(x,self.masks["m1_mask"],axis=1),tf.boolean_mask(x,self.masks["m2_mask"],axis=1)),axis=1),axis=0)
        

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
        vm_r2_cost_recon = -1.0*tf.reduce_mean(tf.reduce_sum(self.lntwopi + vm_r2.log_prob(2.0*np.pi*tf.boolean_mask(x,self.masks["periodic_mask"],axis=1)),axis=1),axis=0)


        # define the von mises fisher for the sky
        fvm_loc = tf.reshape(tf.math.l2_normalize(mean_sky_r2, axis=1),[-1,3])  # mean in (x,y,z)
        fvm_con = tf.reshape(tf.math.reciprocal(self.EPS + tf.exp(logvar_sky_r2)),[-1])
        fvm_r2 = tfp.distributions.VonMisesFisher(
            mean_direction = tf.cast(fvm_loc,dtype=tf.float32),
            concentration = tf.cast(fvm_con,dtype=tf.float32)
        )

        ra_sky = tf.reshape(2*np.pi*tf.boolean_mask(x,self.masks["ra_mask"],axis=1),(-1,1))       # convert the scaled 0->1 true RA value back to radians
        dec_sky = tf.reshape(np.pi*(tf.boolean_mask(x,self.masks["dec_mask"],axis=1) - 0.5),(-1,1)) # convert the scaled 0>1 true dec value back to radians
        xyz_unit = tf.concat([tf.cos(ra_sky)*tf.cos(dec_sky),tf.sin(ra_sky)*tf.cos(dec_sky),tf.sin(dec_sky)],axis=1)   # construct the true parameter unit vector
        normed_xyz = tf.math.l2_normalize(xyz_unit,axis=1) # normalise x,y,z coords to r
        # get log prob
        fvm_r2_cost_recon = -1.0*tf.reduce_mean(self.lnfourpi + fvm_r2.log_prob(normed_xyz),axis=0)


        simple_cost_recon = tmvn_r2_cost_recon + m1m2_r2_cost_recon + vm_r2_cost_recon + fvm_r2_cost_recon


        selfent_q = -1.0*tf.reduce_mean(mvn_q.entropy())
        log_r1_q = gm_r1.log_prob(tf.cast(z_samp,dtype=tf.float32))   # evaluate the log prob of r1 at the q samples
        cost_KL = tf.cast(selfent_q,dtype=tf.float32) - tf.reduce_mean(log_r1_q)

        return simple_cost_recon, cost_KL


    def old_compute_loss(self, x, y, noiseamp=1.0, ramp = 1.0):
        
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
        scale_q = self.EPS + tf.sqrt(tf.exp(logvar_q))
        mvn_q = tfp.distributions.MultivariateNormalDiag(
            loc=mean_q,
            scale_diag=scale_q)
        #mvn_q = tfd.Normal(loc=mean_q,scale=scale_q)
        z_samp = mvn_q.sample()


        mean_r2, logvar_r2, logweight_r2 = self.decode_r2(z=z_samp,y=y)
        scale_r2 = self.EPS + tf.sqrt(tf.exp(logvar_r2))
        
        extra_width = 10.0
        tmvn_r2 = tfp.distributions.TruncatedNormal(
            loc=tf.boolean_mask(tf.squeeze(mean_r2),self.masks["nonperiodic_mask"],axis=1),
            scale=tf.boolean_mask(tf.squeeze(scale_r2),self.masks["nonperiodic_mask"],axis=1),
            low=-extra_width + ramp*extra_width, high=1.0 + extra_width - ramp*extra_width)
        tmvn_r2_cost = tmvn_r2.log_prob(tf.boolean_mask(tf.squeeze(x),self.masks["nonperiodic_mask"],axis=1))
        tmvn_r2_cost_recon = -1.0*tf.reduce_mean(tf.reduce_sum(tmvn_r2_cost,axis=1),axis=0)
        """
        tmvn_r2 = tfd.MultivariateNormalDiag(
            loc=tf.boolean_mask(tf.squeeze(mean_r2),self.masks["nonperiodic_mask"],axis=1),
            scale_diag=tf.boolean_mask(tf.squeeze(scale_r2),self.masks["nonperiodic_mask"],axis=1))
        tmvn_r2_cost = tmvn_r2.log_prob(tf.boolean_mask(tf.squeeze(x),self.masks["nonperiodic_mask"],axis=1))
        tmvn_r2_cost_recon = -1.0*tf.reduce_mean(tmvn_r2_cost)
        """
        
        vm_r2 = tfp.distributions.VonMises(
            loc=2.0*np.pi*tf.boolean_mask(tf.squeeze(mean_r2),self.masks["periodic_mask"],axis=1),
            concentration=tf.math.reciprocal(tf.math.square(tf.boolean_mask(tf.squeeze(2.0*np.pi*scale_r2),self.masks["periodic_mask"],axis=1)))
        )
        vm_r2_cost_recon = -1.0*tf.reduce_mean(tf.reduce_sum(tf.math.log(2.0*np.pi) + vm_r2.log_prob(2.0*np.pi*tf.boolean_mask(x,self.masks["periodic_mask"],axis=1)),axis=1),axis=0)
        simple_cost_recon = tmvn_r2_cost_recon + vm_r2_cost_recon

        # all Gaussian
        """
        gm_r2 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logweight_r2),
                                      components_distribution=tfd.MultivariateNormalDiag(
                                          loc=mean_r2,
                                          scale_diag=scale_r2))
        simple_cost_recon = -1.0*tf.reduce_mean(gm_r2.log_prob(x))
        """


        
        selfent_q = -1.0*tf.reduce_mean(mvn_q.entropy())
        log_r1_q = gm_r1.log_prob(z_samp)   # evaluate the log prob of r1 at the q samples
        cost_KL = selfent_q - tf.reduce_mean(log_r1_q)

        return simple_cost_recon, cost_KL, tmvn_r2_cost_recon, vm_r2_cost_recon, tf.boolean_mask(tf.squeeze(mean_r2),self.masks["nonperiodic_mask"],axis=1), tf.boolean_mask(tf.squeeze(scale_r2),self.masks["nonperiodic_mask"],axis=1), tf.boolean_mask(tf.squeeze(x),self.masks["nonperiodic_mask"],axis=1), tmvn_r2_cost
        


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
                    loc=tf.cast(tf.boolean_mask(mean_nonperiodic_r2_temp,[True, False, False, False, False, False, False, False, False],axis=1),dtype=tf.float32),
                    scale=tf.cast(tf.boolean_mask(self.EPS + tf.sqrt(tf.exp(logvar_nonperiodic_r2_temp)),[True, False, False, False, False, False, False, False, False],axis=1),dtype=tf.float32),
                    low=-10.0 + self.ramp*10.0, high=1.0 + 10.0 - self.ramp*10.0))
                
                md2 = yield tfp.distributions.TruncatedNormal(
                    loc=tf.cast(tf.boolean_mask(mean_nonperiodic_r2_temp,[False, True, False, False, False, False, False, False, False],axis=1),dtype=tf.float32),
                    scale=tf.cast(tf.boolean_mask(self.EPS + tf.sqrt(tf.exp(logvar_nonperiodic_r2_temp)),[False, True, False, False, False, False, False, False, False],axis=1),dtype=tf.float32),
                    low=-10.0 + self.ramp*10.0, high=md1)
                
            m1m2_nonperiodic_r2 = tfp.distributions.JointDistributionCoroutine(model)


            tmvn_nonperiodic_r2 = tfp.distributions.TruncatedNormal(
                loc=tf.cast(tf.boolean_mask(mean_nonperiodic_r2_temp,[False, False, True, True, True, True, True, True, True],axis=1),dtype=tf.float32),
                scale=tf.cast(tf.boolean_mask(self.EPS + tf.sqrt(tf.exp(logvar_nonperiodic_r2_temp)),[False, False, True, True, True, True, True, True, True],axis=1),dtype=tf.float32),
                low=-10.0 + self.ramp*10.0, high=1.0 + 10.0 - self.ramp*10.0)
            
            """
            # non periodic params
            tmvn_nonperiodic_r2 = tfp.distributions.TruncatedNormal(
                loc=tf.cast(mean_nonperiodic_r2_temp,dtype=tf.float32),
                scale=tf.cast(self.EPS + tf.sqrt(tf.exp(logvar_nonperiodic_r2_temp)),dtype=tf.float32),
                low=-10.0 + self.ramp*10.0, high=1.0 + 10.0 - self.ramp*10.0)
            """
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
                x_sample = tf.gather(tf.concat([m1m2_x_sample,tmvn_x_sample,vm_x_sample,fvm_x_sample],axis=1),self.masks["idx_periodic_mask"],axis=1)

                #mean_r2 = tf.concat([mean_np_r2_temp,mean_p_r2_temp,mean_s_r2_temp],axis=1)
                #logvar_r2 = tf.concat([logvar_np_r2_temp,logvar_p_r2_temp,logvar_s_r2_temp],axis=1)
            else:
                x_sample = tf.concat([x_sample,tf.gather(tf.concat([m1m2_x_sample,tmvn_x_sample,vm_x_sample,fvm_x_sample],axis=1),self.masks["idx_periodic_mask"],axis=1)], axis = 0)

                #mean_r2 = tf.concat([mean_r2,tf.concat([mean_np_r2_temp,mean_p_r2_temp,mean_s_r2_temp],axis=1)],axis=0)
                #logvar_r2 = tf.concat([logvar_r2,tf.concat([logvar_np_r2_temp,logvar_p_r2_temp,logvar_s_r2_temp],axis=1)],axis=0)

        return x_sample



    def old_gen_samples(self, y, ramp=1.0, nsamples=1000, max_samples=1000):
        
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

            mean_r2, logvar_r2, logweight_r2 = self.decode_r2(z=z_samp,y=y)
            scale_r2 = self.EPS + tf.sqrt(tf.exp(logvar_r2))
            """
            tmvn_r2 = tfd.MultivariateNormalDiag(
                loc=tf.boolean_mask(tf.squeeze(mean_r2),self.masks["nonperiodic_mask"],axis=1),
                scale_diag=tf.boolean_mask(tf.squeeze(scale_r2),self.masks["nonperiodic_mask"],axis=1))
            """

            extra_width = 10.0
            tmvn_r2 = tfp.distributions.TruncatedNormal(
                loc=tf.boolean_mask(tf.squeeze(mean_r2),self.masks["nonperiodic_mask"],axis=1),
                scale=tf.boolean_mask(tf.squeeze(scale_r2),self.masks["nonperiodic_mask"],axis=1),
                low=-extra_width + ramp*extra_width, high=1.0 + extra_width - ramp*extra_width)
            
            vm_r2 = tfp.distributions.VonMises(
                loc=2.0*np.pi*tf.boolean_mask(tf.squeeze(mean_r2),self.masks["periodic_mask"],axis=1),
                concentration=tf.math.reciprocal(tf.math.square(tf.boolean_mask(tf.squeeze(2.0*np.pi*scale_r2),self.masks["periodic_mask"],axis=1)))
            )
            """
            gm_r2 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logweight_r2),
                                          components_distribution=tfd.MultivariateNormalDiag(
                                              loc=mean_r2,
                                             scale_diag=scale_r2))
            """

            if i==0:
                #x_sample = gm_r2.sample()
                tmvn_x_sample = tmvn_r2.sample()
                vm_x_sample = tf.math.floormod(vm_r2.sample(),(2.0*np.pi))/(2.0*np.pi)
                x_sample = tf.gather(tf.concat([tmvn_x_sample,vm_x_sample],axis=1),self.masks["idx_periodic_mask"],axis=1)
            else:
                #x_sample = tf.concat([x_sample,gm_r2.sample()],axis=0)
                tmvn_x_sample = tmvn_r2.sample()
                vm_x_sample = tf.math.floormod(vm_r2.sample(),(2.0*np.pi))/(2.0*np.pi)
                x_sample = tf.concat([x_sample,tf.gather(tf.concat([tmvn_x_sample,vm_x_sample],axis=1),self.masks["idx_periodic_mask"],axis=1)],axis=0)

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
        scale_q = self.EPS + tf.sqrt(tf.exp(logvar_q))
        mvn_q = tfp.distributions.MultivariateNormalDiag(
            loc=mean_q,
            scale_diag=scale_q)
        z_samp_q = mvn_q.sample()    
        return mean_r1, z_samp_r1, mean_q, z_samp_q

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



