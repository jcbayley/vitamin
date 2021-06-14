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

    def __init__(self, x_dim, y_dim, n_channels, z_dim, n_modes, params, bounds, masks):
        super(CVAE, self).__init__()
        self.z_dim = z_dim
        self.n_modes = n_modes
        self.x_modes = 1   # hardcoded for testing
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_channels = n_channels
        self.act = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.act_relu = tf.keras.layers.ReLU()
        self.params = params
        self.bounds = bounds
        self.masks = masks
        self.EPS = 1e-3
        self.ramp = tf.Variable(0.0, trainable=False)

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

        conv = tf.keras.layers.Conv1D(filters=64, kernel_size=32, strides=1, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(all_input_y)
        #conv = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(conv)
        #conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv1D(filters=32, kernel_size=16, strides=1, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(conv)
        #conv = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(conv)
        #conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv1D(filters=32, kernel_size=8, strides=1, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(conv)
        #conv = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(conv)
        #conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv1D(filters=16, kernel_size=8, strides=1, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(conv)
        #conv = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(conv)

        conv = tf.keras.layers.Flatten()(conv)

        lin1, lin2, lin3 = 4096, 2048, 1024
        # r1 encoder
        r1_in = tf.keras.layers.Dense(lin1, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(conv)
        #r1 = tf.keras.layers.BatchNormalization()(r1_in)
        #r1 = tf.keras.layers.Dropout(.5)(r1_in)
        r1 = tf.keras.layers.Dense(lin2, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(r1_in)
        #r1 = tf.keras.layers.BatchNormalization()(r1)
        #r1 = tf.keras.layers.Dropout(.5)(r1)
        r1 = tf.keras.layers.Dense(lin3, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(r1)
        r1 = tf.keras.layers.Dense(2*self.z_dim*self.n_modes + self.n_modes)(r1)
        self.encoder_r1 = tf.keras.Model(inputs=all_input_y, outputs=r1)
        print(self.encoder_r1.summary())

        # the q encoder network
        q_input_x = tf.keras.Input(shape=(self.x_dim))
        q_inx = tf.keras.layers.Flatten()(q_input_x)
        q_in = tf.keras.layers.concatenate([conv,q_inx])        
        q = tf.keras.layers.Dense(lin1, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(q_in)
        #q = tf.keras.layers.BatchNormalization()(q)
        #q = tf.keras.layers.Dropout(.5)(q)
        q = tf.keras.layers.Dense(lin2, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(q)
        #q = tf.keras.layers.BatchNormalization()(q)
        #q = tf.keras.layers.Dropout(.5)(q)
        q = tf.keras.layers.Dense(lin3, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(q)
        q = tf.keras.layers.Dense(2*self.z_dim)(q)
        self.encoder_q = tf.keras.Model(inputs=[all_input_y, q_input_x], outputs=q)
        print(self.encoder_q.summary())

        # the r2 decoder network
        r2_input_z = tf.keras.Input(shape=(self.z_dim))
        r2_inz = tf.keras.layers.Flatten()(r2_input_z)
        r2_in = tf.keras.layers.concatenate([conv,r2_inz])
        r2 = tf.keras.layers.Dense(lin1, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(r2_in)
        #r2 = tf.keras.layers.BatchNormalization()(r2)
        #r2 = tf.keras.layers.Dropout(.5)(r2)
        r2 = tf.keras.layers.Dense(lin2, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(r2)
        #r2 = tf.keras.layers.BatchNormalization()(r2)
        #r2 = tf.keras.layers.Dropout(.5)(r2)
        r2 = tf.keras.layers.Dense(lin3, kernel_regularizer=regularizers.l2(0.001), activation=self.act)(r2)

        r2w = tf.keras.layers.Dense(self.x_modes,use_bias=True,bias_initializer='Zeros')(r2)
        r2mu = tf.keras.layers.Dense(self.x_dim*self.x_modes,activation='sigmoid')(r2)
        r2logvar = -1.0*tf.keras.layers.Dense(self.x_dim*self.x_modes,activation=self.act_relu,use_bias=True,bias_initializer=tf.keras.initializers.Constant(value=2.0))(r2)
        r2 = tf.keras.layers.concatenate([r2mu,r2logvar,r2w])
        #r2 = tf.keras.layers.Dense(2*self.x_dim*self.x_modes + self.x_modes)(r2)
        self.decoder_r2 = tf.keras.Model(inputs=[all_input_y, r2_input_z], outputs=r2)
        print(self.decoder_r2.summary())

    def encode_r1(self, y=None):
        mean, logvar, weight = tf.split(self.encoder_r1(y), num_or_size_splits=[self.z_dim*self.n_modes, self.z_dim*self.n_modes,self.n_modes], axis=1)
        return tf.reshape(mean,[-1,self.n_modes,self.z_dim]), tf.reshape(logvar,[-1,self.n_modes,self.z_dim]), tf.reshape(weight,[-1,self.n_modes])

    def encode_q(self, x=None, y=None):
        return tf.split(self.encoder_q([y,x]), num_or_size_splits=[self.z_dim,self.z_dim], axis=1)

    def decode_r2(self, y=None, z=None, apply_sigmoid=False):
        mean, logvar, weight = tf.split(self.decoder_r2([y,z]), num_or_size_splits=[self.x_dim*self.x_modes, self.x_dim*self.x_modes,self.x_modes], axis=1)
        return tf.reshape(mean,[-1,self.x_modes,self.x_dim]), tf.reshape(logvar,[-1,self.x_modes,self.x_dim]), tf.reshape(weight,[-1,self.x_modes])

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
            r_loss, kl_loss, g_loss, vm_loss, mean_r2, scale_r2, truths, gcost = self.compute_loss(data[1], data[0], self.ramp)
            loss = r_loss + self.ramp*kl_loss
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.total_loss_metric.update_state(loss)
        self.recon_loss_metric.update_state(r_loss)
        self.kl_loss_metric.update_state(kl_loss)
        self.gauss_loss_metric.update_state(g_loss)
        self.vm_loss_metric.update_state(vm_loss)

        #return r_loss, kl_loss
        return {"total_loss":self.total_loss_metric.result(),
                "recon_loss":self.recon_loss_metric.result(),
                "kl_loss":self.kl_loss_metric.result(),
                "gauss_loss":self.gauss_loss_metric.result(),
                "vm_loss":self.vm_loss_metric.result()}

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
        r_loss, kl_loss,g_loss, vm_loss, mean_r2, scale_r2, truths, gcost = self.compute_loss(data[1], data[0])
        #print("test_loss", self.ramp, r_loss, kl_loss)
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
        mean_r2, logvar_r2, logweight_r2 = self.decode_r2(z=z_samp,y=inputs)
        scale_r2 = self.EPS + tf.sqrt(tf.exp(logvar_r2))
        gm_r2 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logweight_r2),
                                      components_distribution=tfd.MultivariateNormalDiag(
                                          loc=mean_r2,
                                          scale_diag=scale_r2))

        x_sample = gm_r2.sample()

        return x_sample





class PlotCallback(tf.keras.callbacks.Callback):

    def __init__(self, plot_dir, epoch_plot = 2):
        from vitamin_c_fit import plot_losses, plot_losses_zoom
        self.plot_losses = plot_losses
        self.plot_losses_zoom = plot_losses_zoom
        self.plot_dir = plot_dir
        self.epoch_plot = epoch_plot
        self.train_losses = [[],[],[]]
        self.val_losses = [[],[],[]]
    def on_epoch_end(self, epoch, logs = None):
        self.train_losses[2].append(logs["total_loss"])
        self.train_losses[0].append(logs["recon_loss"])
        self.train_losses[1].append(logs["kl_loss"])

        self.val_losses[2].append(logs["val_total_loss"])
        self.val_losses[0].append(logs["val_recon_loss"])
        self.val_losses[1].append(logs["val_kl_loss"])

        if epoch % self.epoch_plot == 0 and epoch > 3:
            self.plot_losses(np.array(self.train_losses).T, np.array(self.val_losses).T, epoch, run = self.plot_dir)
            self.plot_losses_zoom(np.array(self.train_losses).T, np.array(self.val_losses).T, epoch, run = self.plot_dir, ind_start=500)


class TrainCallback(tf.keras.callbacks.Callback):

    def __init__(self, checkpoint_path, optimizer, plot_dir):
        super(TrainCallback, self).__init__()
        self.ramp_start = 400
        self.ramp_length = 600
        self.n_cycles = 1
        self.checkpoint_path = checkpoint_path
        self.plot_dir = plot_dir
        self.optimizer = optimizer
        self.recon_losses = []
        self.kl_losses = []
        self.gauss_losses = []
        self.vm_losses = []
        self.old_recon_losses = []
        self.old_kl_losses = []
        self.old_gauss_losses = []
        self.old_vm_losses = []

    def nan_plots(self):

        total_r_losses = np.append(self.old_recon_losses, self.recon_losses)
        total_kl_losses = np.append(self.old_kl_losses, self.kl_losses)
        total_gauss_losses = np.append(self.old_gauss_losses, self.gauss_losses)
        total_vm_losses = np.append(self.old_vm_losses, self.vm_losses)
        
        fig, ax = plt.subplots(nrows = 2, figsize = (16,10))
        ax[0].plot(total_r_losses, label = "recon")
        ax[0].plot(total_kl_losses, label = "kl")
        ax[0].set_yscale("symlog")
        ax[0].legend()
        ax[1].plot(total_gauss_losses, label = "gauss")
        ax[1].plot(total_vm_losses, label = "von mises")
        ax[1].set_yscale("symlog")
        ax[1].legend()
        
        fig.savefig(os.path.join(self.plot_dir, "batch_inf_plot.png"))
        sys.exit()
        
    def ramp_func(self,epoch):
        ramp = (epoch-self.ramp_start)/(2.0*self.ramp_length)
        #print(epoch,ramp)
        if ramp<0:
            return 0.0
        if ramp>=self.n_cycles:
            return 1.0
        tf.keras.backend.set_value(self.model.ramp, min(1.0,2.0*np.remainder(ramp,1.0)))
        #self.model.compile(run_eagerly = False, optimizer = self.optimizer)

    def on_epoch_begin(self, epoch, logs = None):

        if epoch > self.ramp_start:
            self.ramp_func(epoch)

    def on_batch_end(self, batch, logs=None):
        self.recon_losses.append(self.model.recon_loss_metric.result())
        self.kl_losses.append(self.model.kl_loss_metric.result())
        self.gauss_losses.append(self.model.gauss_loss_metric.result())
        self.vm_losses.append(self.model.vm_loss_metric.result())

        if not np.isfinite(self.recon_losses[-1]):
            print("recon loss inf \n")
            if not np.isfinite(self.gauss_losses[-1]):
                print("gauss inf \n")
            self.nan_plots()
        if not np.isfinite(self.kl_losses[-1]):
            print("kl loss inf")
            self.nan_plots()

        #print("mr",self.model.ramp)
        
    def on_epoch_end(self,epoch, logs = None):
        self.old_recon_losses = self.recon_losses
        self.old_kl_losses = self.kl_losses
        self.old_gauss_losses = self.gauss_losses
        self.old_vm_losses = self.vm_losses

        self.recon_losses = []
        self.kl_losses = []
        self.gauss_losses = []
        self.vm_losses = []

        if epoch % self.model.params['save_interval'] == 0:
            # Save the weights using the `checkpoint_path` format
            self.model.save_weights(self.checkpoint_path)
            print('... Saved model %s ' % self.checkpoint_path)

class TestCallback(tf.keras.callbacks.Callback):


    def __init__(self,test_dataset, plot_dir, bilby_samples):
        from vitamin_c_fit import plot_latent, plot_posterior
        self.plot_latent = plot_latent
        self.plot_posterior = plot_posterior
        self.test_dataset = test_dataset
        self.plot_dir = plot_dir
        self.bilby_samples = bilby_samples

    def on_epoch_end(self, epoch, logs = None):
        
        if epoch % 500 == 0:
            for step in range(len(self.test_dataset)):             
                mu_r1, z_r1, mu_q, z_q = self.model.gen_z_samples(tf.expand_dims(self.test_dataset.X[step],0), tf.expand_dims(self.test_dataset.Y_noisy[step],0), nsamples=1000)
                self.plot_latent(mu_r1,z_r1,mu_q,z_q,epoch,step,run=self.plot_dir)
                start_time_test = time.time()
                samples = self.model.gen_samples(tf.expand_dims(self.test_dataset.Y_noisy[step],0), nsamples=self.model.params['n_samples'])
                end_time_test = time.time()
                if np.any(np.isnan(samples)):
                    print('Epoch: {}, found nans in samples. Not making plots'.format(epoch))
                    for k,s in enumerate(samples):
                        if np.any(np.isnan(s)):
                            print(k,s)
                    KL_est = [-1,-1,-1]
                else:
                    print('Epoch: {}, Testing time elapsed for {} samples: {}'.format(epoch,self.model.params['n_samples'],end_time_test - start_time_test))
                    KL_est = self.plot_posterior(samples,self.test_dataset.X[step],epoch,step,all_other_samples=self.bilby_samples[:,step,:],run=self.plot_dir, params = self.model.params, bounds = self.model.bounds, masks = self.model.masks)
                    _ = self.plot_posterior(samples,self.test_dataset.X[step],epoch,step,run=self.plot_dir, params = self.model.params, bounds = self.model.bounds, masks = self.model.masks)
                #KL_samples.append(KL_est)


class TimeCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        temp_time = time.time() - self.epoch_time_start
        self.times.append(temp_time)
        print(temp_time)











