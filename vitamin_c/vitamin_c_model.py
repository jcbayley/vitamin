import tensorflow as tf
from tensorflow.keras import regularizers
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np

class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, x_dim, y_dim, n_channels, z_dim, n_modes, params, bounds, masks):
        super(CVAE, self).__init__()
        self.z_dim = z_dim
        self.n_modes = n_modes
        self.x_modes = 1   # hardcoded for testing
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_channels = n_channels
        self.act = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.params = params
        self.bounds = bounds
        self.masks = masks
        self.EPS = 1e-3
        self.train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

        """
        # Add this to get rid of regularizer
        a2 = tf.keras.layers.Dense(2*self.z_dim*self.n_modes + self.n_modes)(a2)
        self.encoder_r1 = tf.keras.Model(inputs=r1_input_y, outputs=a2)
        print(self.encoder_r1.summary())
        """
        # the r1 encoder network
        r1_input_y = tf.keras.Input(shape=(self.y_dim, self.n_channels))
        a = tf.keras.layers.Conv1D(filters=32, kernel_size=11, strides=1, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(r1_input_y)
        a = tf.keras.layers.Conv1D(filters=32, kernel_size=8, strides=2, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(a)
        a = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(a)
        #a = tf.keras.layers.Conv1D(filters=96, kernel_size=16, strides=2, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(a)
        #a = tf.keras.layers.Conv1D(filters=96, kernel_size=16, strides=1, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(a)
        #a = tf.keras.layers.Conv1D(filters=96, kernel_size=16, strides=2, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(a)
        a = tf.keras.layers.Flatten()(a)
        a2 = tf.keras.layers.Dense(4096, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(a)
        a2 = tf.keras.layers.Dropout(.5)(a2)
        a2 = tf.keras.layers.Dense(2048, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(a2)
        a2 = tf.keras.layers.Dropout(.5)(a2)
        a2 = tf.keras.layers.Dense(1024, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(a2)
        a2 = tf.keras.layers.Dense(2*self.z_dim*self.n_modes + self.n_modes)(a2)
        self.encoder_r1 = tf.keras.Model(inputs=r1_input_y, outputs=a2)
        print(self.encoder_r1.summary())

        # the q encoder network
        q_input_x = tf.keras.Input(shape=(self.x_dim))
        c = tf.keras.layers.Flatten()(q_input_x)
        d = tf.keras.layers.concatenate([a,c])        
        e = tf.keras.layers.Dense(4096, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(d)
        e = tf.keras.layers.Dropout(.5)(e)
        e = tf.keras.layers.Dense(2048, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(e)
        e = tf.keras.layers.Dropout(.5)(e)
        e = tf.keras.layers.Dense(1024, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(e)
        e = tf.keras.layers.Dense(2*self.z_dim)(e)
        self.encoder_q = tf.keras.Model(inputs=[r1_input_y, q_input_x], outputs=e)
        print(self.encoder_q.summary())

        # the r2 decoder network
        r2_input_z = tf.keras.Input(shape=(self.z_dim))
        g = tf.keras.layers.Flatten()(r2_input_z)
        h = tf.keras.layers.concatenate([a,g])
        i = tf.keras.layers.Dense(4096, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(h)
        i = tf.keras.layers.Dropout(.5)(i)
        i = tf.keras.layers.Dense(2048, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(i)
        i = tf.keras.layers.Dropout(.5)(i)
        i = tf.keras.layers.Dense(1024, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(i)
        j = tf.keras.layers.Dense(2*self.x_dim*self.x_modes + self.x_modes)(i)
        self.decoder_r2 = tf.keras.Model(inputs=[r1_input_y, r2_input_z], outputs=j)
        print(self.decoder_r2.summary())

    def encode_r1(self, y=None):
        mean, logvar, weight = tf.split(self.encoder_r1(y), num_or_size_splits=[self.z_dim*self.n_modes, self.z_dim*self.n_modes,self.n_modes], axis=1)
        return tf.reshape(mean,[-1,self.n_modes,self.z_dim]), tf.reshape(logvar,[-1,self.n_modes,self.z_dim]), tf.reshape(weight,[-1,self.n_modes])

    def encode_q(self, x=None, y=None):
        return tf.split(self.encoder_q([y,x]), num_or_size_splits=[self.z_dim,self.z_dim], axis=1)

    def decode_r2(self, y=None, z=None, apply_sigmoid=False):
        mean, logvar, weight = tf.split(self.decoder_r2([y,z]), num_or_size_splits=[self.x_dim*self.x_modes, self.x_dim*self.x_modes,self.x_modes], axis=1)
        return tf.reshape(mean,[-1,self.x_modes,self.x_dim]), tf.reshape(logvar,[-1,self.x_modes,self.x_dim]), tf.reshape(weight,[-1,self.x_modes])

    @tf.function
    def train_step(self, x, y, optimizer, ramp=1.0):
        """Executes one training step and returns the loss.
        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            r_loss, kl_loss = self.compute_loss(x, y, ramp)
            loss = r_loss + ramp*kl_loss

        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss_metric(loss)
        return r_loss, kl_loss


    def compute_loss(self, x, y, noiseamp=1.0, ramp = 1.0):
        
        # Recasting some things to float32
        noiseamp = tf.cast(noiseamp, dtype=tf.float32)

        y = tf.cast(y, dtype=tf.float32)
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
        

        #print("means",tf.reduce_mean(mean_r2))
            
        """
        tmvn_r2 = tfp.distributions.TruncatedNormal(
            loc=tf.boolean_mask(tf.squeeze(mean_r2),self.masks["nonperiodic_mask"],axis=1),
            scale=tf.boolean_mask(tf.squeeze(scale_r2),self.masks["nonperiodic_mask"],axis=1),
            low=-10.0 + ramp*10.0, high=1.0 + 10.0 - ramp*10.0)

        tmvn_r2_cost_recon = -1.0*tf.reduce_mean(tf.reduce_sum(tmvn_r2.log_prob(tf.boolean_mask(x,self.masks["nonperiodic_mask"],axis=1)),axis=1),axis=0)
        """
        tmvn_r2 = tfd.MultivariateNormalDiag(
            loc=tf.boolean_mask(tf.squeeze(mean_r2),self.masks["nonperiodic_mask"],axis=1),
            scale_diag=tf.boolean_mask(tf.squeeze(scale_r2),self.masks["nonperiodic_mask"],axis=1))
        tmvn_r2_cost_recon = -1.0*tf.reduce_mean(tmvn_r2.log_prob(tf.boolean_mask(tf.squeeze(x),self.masks["nonperiodic_mask"],axis=1)))

        vm_r2 = tfp.distributions.VonMises(
            loc=2.0*np.pi*tf.boolean_mask(tf.squeeze(mean_r2),self.masks["periodic_mask"],axis=1),
            concentration=tf.math.reciprocal(tf.math.square(tf.boolean_mask(tf.squeeze(2.0*np.pi*scale_r2),self.masks["periodic_mask"],axis=1)))
        )
        vm_r2_cost_recon = -1.0*tf.reduce_mean(tf.reduce_sum(tf.math.log(2.0*np.pi) + vm_r2.log_prob(2.0*np.pi*tf.boolean_mask(x,self.masks["periodic_mask"],axis=1)),axis=1),axis=0)
        simple_cost_recon = tmvn_r2_cost_recon + vm_r2_cost_recon
        #print("cost", tmvn_r2_cost_recon , vm_r2_cost_recon)
        #if np.isnan(simple_cost_recon):
        #    print(tmvn_r2_cost_recon, vm_r2_cost_recon)
        #    print(mean_r2)
        #    sys.exit()

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
        return simple_cost_recon, cost_KL
        
    def gen_samples(self, y, ramp=1.0, nsamples=1000, max_samples=1000):
        
        y = y/self.params['y_normscale']
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

            tmvn_r2 = tfd.MultivariateNormalDiag(
                loc=tf.boolean_mask(tf.squeeze(mean_r2),self.masks["nonperiodic_mask"],axis=1),
                scale_diag=tf.boolean_mask(tf.squeeze(scale_r2),self.masks["nonperiodic_mask"],axis=1))
            
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
                x_sample = tf.gather(tf.concat([tmvn_r2.sample(),vm_x_sample],axis=1),self.masks["idx_periodic_mask"],axis=1)
            else:
                #x_sample = tf.concat([x_sample,gm_r2.sample()],axis=0)
                tmvn_x_sample = tmvn_r2.sample()
                vm_x_sample = tf.math.floormod(vm_r2.sample(),(2.0*np.pi))/(2.0*np.pi)
                x_sample = tf.concat([x_sample,tf.gather(tf.concat([tmvn_r2.sample(),vm_x_sample],axis=1),self.masks["idx_periodic_mask"],axis=1)],axis=0)

        return x_sample


    def gen_z_samples(self, x, y, nsamples=1000):
        y = y/self.params['y_normscale']
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
