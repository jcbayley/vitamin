import tensorflow as tf
from tensorflow.keras import regularizers
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import time 
import matplotlib.pyplot as plt
import os
import sys
from .initialise import group_outputs

class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, config, verbose = False):
        super(CVAE, self).__init__()
        self.config = config
        self.z_dim = self.config["model"]["z_dimension"]
        self.n_modes = self.config["model"]["n_modes"]
        self.x_modes = 1   # hardcoded for testing
        self.x_dim = len(self.config["inf_pars"])
        self.y_dim = self.config["data"]["sampling_frequency"]*self.config["data"]["duration"]
        self.n_channels = len(self.config["data"]["detectors"])
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.activation_relu = tf.keras.layers.ReLU()
        self.kernel_initializer = "glorot_uniform"#tf.keras.initializers.HeNormal() 
        self.bias_initializer = "zeros"#tf.keras.initializers.HeNormal() 
        self.bias_initializer_2 = tf.keras.initializers.HeNormal() 

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

        self.init_network()

    def init_network(self):

        # the shared convolutional network
        all_input_y = tf.keras.Input(shape=(self.y_dim, self.n_channels))
        conv = self.get_network(all_input_y, self.config["model"]["shared_network"])
        conv = tf.keras.layers.Flatten()(conv)

        # r1 encoder network
        r1 = self.get_network(conv, self.config["model"]["output_network"])
        r1mu = tf.keras.layers.Dense(self.z_dim*self.n_modes, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer_2)(r1)
        r1logvar = tf.keras.layers.Dense(self.z_dim*self.n_modes, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)(r1)
        r1modes = tf.keras.layers.Dense(self.n_modes, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)(r1)
        r1 = tf.keras.layers.concatenate([r1mu,r1logvar,r1modes])
        self.encoder_r1 = tf.keras.Model(inputs=all_input_y, outputs=r1)

        # the q encoder network
        q_input_x = tf.keras.Input(shape=(self.x_dim))
        q_inx = tf.keras.layers.Flatten()(q_input_x)
        q = tf.keras.layers.concatenate([conv,q_inx])
        q = self.get_network(q, self.config["model"]["output_network"])
        qmu = tf.keras.layers.Dense(self.z_dim, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer_2)(q)
        qlogvar = tf.keras.layers.Dense(self.z_dim, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)(q)
        q = tf.keras.layers.concatenate([qmu,qlogvar])
        self.encoder_q = tf.keras.Model(inputs=[all_input_y, q_input_x], outputs=q)

        # the r2 decoder network
        r2_input_z = tf.keras.Input(shape=(self.z_dim))
        r2_inz = tf.keras.layers.Flatten()(r2_input_z)
        r2 = tf.keras.layers.concatenate([conv,r2_inz])
        r2 = self.get_network(r2, self.config["model"]["output_network"])
        
        self.grouped_params = group_outputs(self.config)
        
        outputs = []
        self.group_par_sizes = []
        self.group_output_sizes = []
        for name, group in self.grouped_params.items():
            means, logvars = group.get_networks()
            outputs.append(means(r2))
            outputs.append(logvars(r2))
            self.group_par_sizes.append(group.num_pars)
            self.group_output_sizes.extend(group.num_outputs)
            
        # all outputs
        r2 = tf.keras.layers.concatenate(outputs)

        self.decoder_r2 = tf.keras.Model(inputs=[all_input_y, r2_input_z], outputs=r2)

        self.compute_loss = self.create_loss_func()

        self.gen_samples = self.create_sample_func()


    def encode_r1(self, y=None):
        mean, logvar, weight = tf.split(self.encoder_r1(y), num_or_size_splits=[self.z_dim*self.n_modes, self.z_dim*self.n_modes,self.n_modes], axis=1)
        return tf.reshape(mean,[-1,self.n_modes,self.z_dim]), tf.reshape(logvar,[-1,self.n_modes,self.z_dim]), tf.reshape(weight,[-1,self.n_modes])

    def encode_q(self, x=None, y=None):
        return tf.split(self.encoder_q([y,x]), num_or_size_splits=[self.z_dim,self.z_dim], axis=1)

    def decode_r2(self, y=None, z=None, apply_sigmoid=False):
        return tf.split(self.decoder_r2([y,z]), num_or_size_splits=self.group_output_sizes, axis=1)

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

    def create_loss_func(self):
        @tf.function
        def loss_func(x, y, noiseamp=1.0, ramp = 1.0):
        
            # Recasting some things to float32
            noiseamp = tf.cast(noiseamp, tf.float32)
            
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

            decoded_outputs = self.decode_r2(z = z_samp, y=y)
            x_grouped = tf.split(x, num_or_size_splits=self.group_par_sizes, axis=1)
            cost_recon = 0 
            ind = 0
            indx = 0
            for group in self.grouped_params.values():
                dist = group.get_distribution(decoded_outputs[ind], decoded_outputs[ind + 1], EPS = self.EPS, ramp = self.ramp)
                cr = group.get_cost(dist, x_grouped[indx])
                #print(group.pars, cr)
                cost_recon += cr
                ind += 2
                indx += 1


            selfent_q = -1.0*tf.reduce_mean(mvn_q.entropy())
            log_r1_q = gm_r1.log_prob(tf.cast(z_samp,dtype=tf.float32))   # evaluate the log prob of r1 at the q samples
            cost_KL = tf.cast(selfent_q,dtype=tf.float32) - tf.reduce_mean(log_r1_q)
            return cost_recon, cost_KL
            
        return loss_func

    def create_sample_func(self):

        def sample_func(y, nsamples = 1000, max_samples = 1000):
        
            # Recasting some things to float32
            #noiseamp = tf.cast(noiseamp, dtype=tf.float32)
            
            y = tf.cast(y, dtype=tf.float32)
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
                
                decoded_outputs = self.decode_r2(z = z_samp, y=y)
                ind = 0
                indx = 0
                dist_samples = []
                for group in self.grouped_params.values():
                    dist = group.get_distribution(decoded_outputs[ind], decoded_outputs[ind + 1], EPS = self.EPS, ramp = self.ramp)
                    dist_samples.append(group.sample(dist, max_samples))
                    ind += 2
                    indx += 1

                if i==0:
                    x_sample = tf.concat(dist_samples,axis=1)
                else:
                    x_sample = tf.concat([x_sample,tf.concat(dist_samples,axis=1)], axis = 0)
                    
            return tf.gather(x_sample, self.config["masks"]["ungroup_order_idx"],axis=1)
        

        return sample_func

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

        decoded_pars = self.decode_r2(z=z_samp,y=inputs)
        
        means = tf.concat(decoded_pars, axis = 1)
        return means

    def get_network(self, in_tensor, layers):
        
        conv = in_tensor
        for layer in layers:
            if layer.split("(")[0] == "Conv1D":
                nfilters, filter_size, stride = layer.split("(")[1].strip(")").split(",")
                conv = self.ConvBlock(conv, int(nfilters), int(filter_size), int(stride))
            elif layer.split("(")[0] == "ResBlock":
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
        #, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer
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
