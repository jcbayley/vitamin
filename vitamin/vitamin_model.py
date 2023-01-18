import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import time 
import matplotlib.pyplot as plt
import os
import sys
from .group_inference_parameters import group_outputs

class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, config = None, verbose = False,**kwargs):
        """
        Input to the convolutional variational autoencoder
        kwargs
        --------------
        z_dim: int (default 4)
            size of the latent space
        n_modes: int (default 2)
            number of modes to allow in latent space
        x_dim: int 
            size of the x space (parameters to be inferred)
        inf_pars: dict
            the parameters to be inferred and the output distributions associated with them {"p0":"TruncatedNormal", "p1":"TruncatedNormal"}
        bounds: dict
            the upper and lower bounds to be used for normalisation for each of the parameters {"p0_min":0,"p0_max":1, "p1_min":0,"p1_max":1} 
        y_dim: int 
            the size of the input data
        n_channels: int
            the number of channels to use for the input data
        shared_network: list
            the structure of the shared network
        r1_network: list
            the structure of the shared network
        q_network: list
            the structure of the shared network
        r2_network: list
            the structure of the shared network

        """
        super(CVAE, self).__init__()
        default_kwargs = dict(
            z_dim = 4,
            n_modes = 2,
            x_modes = 1,
            hidden_activation = "leakyrelu",
            include_psd = False,
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

        for key, val in kwargs.items():
            if key in default_kwargs.keys():
                setattr(self, key, val)
            else:
                raise Exception("Key {} not valid, please choose from {}".format(key, list(default_kwargs.keys())))

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
            self.shared_network = self.config["model"]["shared_network"]
            self.r1_network = self.config["model"]["r1_network"]
            self.q_network = self.config["model"]["q_network"]
            self.r2_network = self.config["model"]["r2_network"]
            self.bounds = self.config["bounds"]
            self.inf_pars = self.config["inf_pars"]
            self.hidden_activation = self.config["model"]["hidden_activation"]
            if self.logvarmin:
                self.logvarmin_start = self.config["training"]["logvarmin_start"]

        #self.activation = tf.keras.layers.LeakyReLU(alpha=0.3)
        if self.hidden_activation == "leakyrelu":
            self.activation = tf.keras.layers.LeakyReLU(alpha=0.3)
        elif self.hidden_activation == "tanh":
            self.activation = tf.keras.activations.tanh
        elif self.hidden_activation == "swish":
            self.activation = tf.keras.activations.swish
        elif self.hidden_activation == "relu":
            self.activation = tf.keras.activations.relu
        elif self.hidden_activation == "gelu":
            self.activation = tf.keras.activations.gelu
        else:
            raise NotImplementedError(f"{self.hidden_activation} is not implemented, use: [leakyrelu, relu, tanh, swish]")

        self.activation_relu = tf.keras.activations.relu
        self.kernel_initializer = "glorot_uniform"#tf.keras.initializers.HeNormal() 
        self.bias_initializer = "zeros"#tf.keras.initializers.HeNormal() 
        self.bias_initializer_2 = tf.keras.initializers.HeNormal() 

        self.verbose = verbose
        # consts
        self.EPS = 1e-3
        self.fourpisq = 4.0*np.pi*np.pi
        self.lntwopi = tf.math.log(2.0*np.pi)
        self.lnfourpi = tf.math.log(4.0*np.pi)
        #self.maxlogvar = np.log(np.nan_to_num(np.float32(np.inf))) - 1
        #self.inv_maxlogvar = 1./self.maxlogvar
        if self.logvarmin:
            self.minlogvar = tf.Variable(self.logvarmin_start, trainable=False, dtype=tf.float32)
            self.maxlogvar = 4

        self.logvar_act = self.setup_logvaract()

        # variables
        self.ramp = tf.Variable(1.0, trainable=False)
        self.initial_learning_rate = self.initial_learning_rate

        self.grouped_params, self.new_params_order, self.reverse_params_order = group_outputs(self.inf_pars, self.bounds)

        self.total_loss_metric = tf.keras.metrics.Mean('total_loss', dtype=tf.float32)
        self.recon_loss_metric = tf.keras.metrics.Mean('recon_loss', dtype=tf.float32)
        self.kl_loss_metric = tf.keras.metrics.Mean('KL_loss', dtype=tf.float32)

        self.val_total_loss_metric = tf.keras.metrics.Mean('val_total_loss', dtype=tf.float32)
        self.val_recon_loss_metric = tf.keras.metrics.Mean('val_recon_loss', dtype=tf.float32)
        self.val_kl_loss_metric = tf.keras.metrics.Mean('val_KL_loss', dtype=tf.float32)
        
        self.init_network()

    def setup_logvaract(self):
        if self.logvarmin == False:
            def logvar_act(x):
                return x
        else:
            def logvar_act(x):
                return (self.maxlogvar - self.minlogvar)*tf.keras.activations.sigmoid(x) + self.minlogvar
        return logvar_act

    def init_network(self):

        # the shared convolutional network
        if self.include_psd:
            all_input_y = tf.keras.Input(shape=(self.y_dim, 2*self.n_channels))
        else:
            all_input_y = tf.keras.Input(shape=(self.y_dim, self.n_channels))
        #if self.include_psd:
        #    all_input_psd = tf.keras.Input(shape=(self.y_dim, self.n_channels))
        if "keras" not in str(type(self.shared_network)):
            # if network a list then create the network
            if self.split_channels:
                conv = []
                for i in range(self.n_channels):
                    inch = tf.keras.layers.Lambda(lambda x: x[:, :, i:i+1])(all_input_y)
                    conv.append(self.get_network(inch, self.shared_network, label=f"shared_{i}"))

                conv = tf.keras.layers.Concatenate()(conv)
            else:
                conv = self.get_network(all_input_y, self.shared_network, label="shared")
            #if self.include_psd:
            #    convpsd = self.get_network(all_input_psd, self.shared_network, label="sharedpsd")
        else:
            #otherwise use the input network
            conv = self.shared_network(all_input_y)
            #if self.include_psd:
            #    convpsd = self.shared_network(all_input_psd)

        #if self.include_psd:
        #    conv = tf.keras.layers.concatenate([conv,convpsd])

        # r1 encoder network
        if "keras" not in str(type(self.shared_network)):
            r1 = self.get_network(conv, self.r1_network, label = "r1")
        else:
            r1 = self.r1_network(conv)
        r1mu = tf.keras.layers.Dense(self.z_dim*self.n_modes, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer_2, name = "r1_mean_dense")(r1)
        r1logvar = tf.keras.layers.Dense(self.z_dim*self.n_modes, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer,name="r1_logvar_dense",activation="relu")(r1)
        #r1logvar = tf.keras.layers.Dense(self.z_dim*self.n_modes, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer,name="r1_logvar_dense",activation = self.logvar_act)(r1)
        r1modes = tf.keras.layers.Dense(self.n_modes, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer,name="r1_modes_dense")(r1)
        r1 = tf.keras.layers.concatenate([r1mu,r1logvar,r1modes])
        #if self.include_psd:
        #    self.encoder_r1 = tf.keras.Model(inputs=[all_input_y,all_input_psd], outputs=r1,name="encoder_r1")
        #else:
        self.encoder_r1 = tf.keras.Model(inputs=all_input_y, outputs=r1,name="encoder_r1")

        # the q encoder network
        q_input_x = tf.keras.Input(shape=(self.x_dim))
        q_inx = tf.keras.layers.Flatten()(q_input_x)
        q = tf.keras.layers.concatenate([conv,q_inx])
        if "keras" not in str(type(self.shared_network)):
            q = self.get_network(q, self.q_network, label = "q")
        else:
            q = self.q_network(conv)
        qmu = tf.keras.layers.Dense(self.z_dim, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer_2, name="q_mean_dense")(q)
        qlogvar = tf.keras.layers.Dense(self.z_dim, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer,name="q_logvar_dense",activation="relu")(q)
        #qlogvar = tf.keras.layers.Dense(self.z_dim, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer,name="q_logvar_dense",activation=self.logvar_act)(q)
        q = tf.keras.layers.concatenate([qmu,qlogvar])
        #if self.include_psd:
        #    self.encoder_q = tf.keras.Model(inputs=[all_input_y, q_input_x, all_input_psd], outputs=q,name = "encoder_q")
        #else:
        self.encoder_q = tf.keras.Model(inputs=[all_input_y, q_input_x], outputs=q,name = "encoder_q")

        # the r2 decoder network
        r2_input_z = tf.keras.Input(shape=(self.z_dim))
        r2_inz = tf.keras.layers.Flatten()(r2_input_z)
        r2 = tf.keras.layers.concatenate([conv,r2_inz])
        if "keras" not in str(type(self.shared_network)):
            r2 = self.get_network(r2, self.r2_network, label = "r2")
        else:
            r2 = self.r2_network(conv)
        
        
        outputs = []
        self.group_par_sizes = []
        self.group_output_sizes = []
        for name, group in self.grouped_params.items():
            means, logvars = group.get_networks(logvar_activation = self.logvar_act)
            outputs.append(means(r2))
            outputs.append(logvars(r2))
            self.group_par_sizes.append(group.num_pars)
            self.group_output_sizes.extend(group.num_outputs)
            setattr(self, "{}_loss_metric".format(name), tf.keras.metrics.Mean('{}_loss'.format(name), dtype=tf.float32))

        # all outputs
        r2 = tf.keras.layers.concatenate(outputs)

        #if self.include_psd:
        #    self.decoder_r2 = tf.keras.Model(inputs=[all_input_y, r2_input_z, all_input_psd], outputs=r2,name = "decoder_r2")
        #else:
        self.decoder_r2 = tf.keras.Model(inputs=[all_input_y, r2_input_z], outputs=r2,name = "decoder_r2")
        self.compute_loss = self.create_loss_func()

        self.gen_samples = self.create_sample_func()


    def encode_r1(self, y=None):
        if self.include_psd:
            mean, logvar, weight = tf.split(self.encoder_r1(y), num_or_size_splits=[self.z_dim*self.n_modes, self.z_dim*self.n_modes,self.n_modes], axis=1)
        else:
            mean, logvar, weight = tf.split(self.encoder_r1(y), num_or_size_splits=[self.z_dim*self.n_modes, self.z_dim*self.n_modes,self.n_modes], axis=1)
        #mean, logvar, weight = tf.split(self.encoder_r1(y), num_or_size_splits=[self.z_dim*self.n_modes, self.z_dim*self.z_dim*self.n_modes,self.n_modes], axis=1)
        return tf.reshape(mean,[-1,self.n_modes,self.z_dim]), tf.reshape(logvar,[-1,self.n_modes,self.z_dim]), tf.reshape(weight,[-1,self.n_modes])

    def encode_q(self, x=None, y=None):
        return tf.split(self.encoder_q([y,x]), num_or_size_splits=[self.z_dim,self.z_dim], axis=1)
        #mean, logvar =  tf.split(self.encoder_q([y,x]), num_or_size_splits=[self.z_dim,self.z_dim*self.z_dim], axis=1)
        #return mean, tf.reshape(logvar, [-1, self.z_dim, self.z_dim])

    def decode_r2(self, y=None, z=None, apply_sigmoid=False):
        return tf.split(self.decoder_r2([y,z]), num_or_size_splits=self.group_output_sizes, axis=1)

    @property
    def metrics(self):
        base_metrics = [self.total_loss_metric,
                        self.val_total_loss_metric,
                        self.recon_loss_metric,
                        self.val_recon_loss_metric,
                        self.kl_loss_metric,
                        self.val_kl_loss_metric]

        for name, group in self.grouped_params.items():
            base_metrics.append(getattr(self, "{}_loss_metric".format(name)))

        return base_metrics

    @tf.function
    def train_step(self, data):
        """Executes one training step and returns the loss.
        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        
        with tf.GradientTape() as tape:
            r_loss, kl_loss, reconlosses = self.compute_loss(data[1], data[0], ramp=self.ramp)
            loss = r_loss + self.ramp*kl_loss
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        outputs = {}
        self.total_loss_metric.update_state(loss)
        outputs["total_loss"] = self.total_loss_metric.result()
        self.recon_loss_metric.update_state(r_loss)
        outputs["recon_loss"] = self.recon_loss_metric.result()
        self.kl_loss_metric.update_state(kl_loss)
        outputs["kl_loss"] = self.kl_loss_metric.result()
        
        for name, group in self.grouped_params.items():
            getattr(self, "{}_loss_metric".format(name)).update_state(reconlosses[name])
            outputs["{}_loss".format(name)] = getattr(self, "{}_loss_metric".format(name)).result()

        return outputs

    @tf.function
    def test_step(self, data):
        """Executes one test step and returns the loss.                                                        
        This function computes the loss and gradients (used for validation data)
        """
        
        #r_loss, kl_loss,g_loss, vm_loss, mean_r2, scale_r2, truths, gcost = self.compute_loss(data[1], data[0])
        r_loss, kl_loss, reconloss  = self.compute_loss(data[1], data[0])
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
            scale_r1 = tf.sqrt(tf.exp(logvar_r1))
            gm_r1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logweight_r1),
                                          components_distribution=tfd.MultivariateNormalDiag(
                                              loc=mean_r1,
                                              scale_diag=scale_r1))
            #gm_r1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logweight_r1),
            #                              components_distribution = tfp.distributions.MultivariateNormalFullCovariance(loc=mean_r1, covariance_matrix=scale_r1))


            mean_q, logvar_q = self.encode_q(x=x,y=y)
            scale_q = tf.sqrt(tf.exp(logvar_q))
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
            outs = {}
            for name, group in self.grouped_params.items():
                dist = group.get_distribution(decoded_outputs[ind], decoded_outputs[ind + 1], ramp = self.ramp)
                cr = group.get_cost(dist, x_grouped[indx])
                outs[name] = cr
                #(group.pars, cr)
                cost_recon += cr
                ind += 2
                indx += 1


            selfent_q = -1.0*tf.reduce_mean(mvn_q.entropy())
            log_r1_q = gm_r1.log_prob(tf.cast(z_samp,dtype=tf.float32))   # evaluate the log prob of r1 at the q samples
            cost_KL = tf.cast(selfent_q,dtype=tf.float32) - tf.reduce_mean(log_r1_q)
            return cost_recon, cost_KL, outs 
            
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
                scale_r1 = tf.sqrt(tf.exp(logvar_r1))
                gm_r1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logweight_r1),
                                              components_distribution=tfd.MultivariateNormalDiag(
                                                  loc=mean_r1,
                                                  scale_diag=scale_r1))
                #gm_r1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logweight_r1),
                #                              components_distribution = tfp.distributions.MultivariateNormalFullCovariance(loc=mean_r1, covariance_matrix=scale_r1))


                z_samp = gm_r1.sample()
                
                decoded_outputs = self.decode_r2(z = z_samp, y=y)
                ind = 0
                indx = 0
                dist_samples = []
                for group in self.grouped_params.values():
                    dist = group.get_distribution(decoded_outputs[ind], decoded_outputs[ind + 1],  ramp = self.ramp)
                    dist_samples.append(group.sample(dist, max_samples))
                    ind += 2
                    indx += 1

                if i==0:
                    x_sample = tf.concat(dist_samples,axis=1)
                else:
                    x_sample = tf.concat([x_sample,tf.concat(dist_samples,axis=1)], axis = 0)
                    
            return tf.gather(x_sample, self.reverse_params_order,axis=1)
        

        return sample_func

    def gen_z_samples(self, x, y, nsamples=1000):

        #y = y/self.params['y_normscale']
        y = tf.keras.activations.tanh(y)
        y = tf.tile(y,(nsamples,1,1))
        x = tf.tile(x,(nsamples,1))
        mean_r1, logvar_r1, logweight_r1 = self.encode_r1(y=y)
        scale_r1 = tf.sqrt(tf.exp(logvar_r1))
        gm_r1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logweight_r1),
                                      components_distribution=tfd.MultivariateNormalDiag(
                                          loc=mean_r1,
                                          scale_diag=scale_r1))
        #gm_r1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logweight_r1),
        #                              components_distribution = tfp.distributions.MultivariateNormalFullCovariance(loc=mean_r1, covariance_matrix=scale_r1))

        z_samp_r1 = gm_r1.sample()

        mean_q, logvar_q = self.encode_q(x=x,y=y)
        scale_q = tf.sqrt(tf.exp(logvar_q))
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
        scale_r1 = tf.sqrt(tf.exp(logvar_r1))
        #gm_r1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logweight_r1),
        #                              components_distribution = tfp.distributions.MultivariateNormalFullCovariance(loc=mean_r1, covariance_matrix=scale_r1))

        gm_r1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logweight_r1),
                                      components_distribution=tfd.MultivariateNormalDiag(
                                          loc=mean_r1,
                                          scale_diag=scale_r1))

        z_samp = gm_r1.sample()

        decoded_pars = self.decode_r2(z=z_samp,y=inputs)
        
        means = tf.concat(decoded_pars, axis = 1)
        return means

    def get_network(self, in_tensor, layers, label = "share"):
        """ create the layers of the network from the list of layers in config file or list"""
        conv = in_tensor
        for i, layer in enumerate(layers):
            if layer.split("(")[0] == "Conv1D":
                nfilters, filter_size, stride = layer.split("(")[1].strip(")").split(",")
                conv = self.ConvBlock(conv, int(nfilters), int(filter_size), int(stride), name = "{}_conv_{}".format(label,i))
            elif layer.split("(")[0] == "SepConv1D":
                nfilters, filter_size, stride = layer.split("(")[1].strip(")").split(",")
                conv = self.SepConvBlock(conv, int(nfilters), int(filter_size), int(stride), name = "{}_conv_{}".format(label,i))
            elif layer.split("(")[0] == "ResBlock":
                nfilters, filter_size, stride = layer.split("(")[1].strip(")").split(",")
                # add a bottleneck block
                conv = self.ResBlock(conv, [int(nfilters), int(nfilters), int(nfilters)], [1, int(filter_size)], int(stride), name = "{}_res_{}".format(label, i))
            elif layer.split("(")[0] == "Linear":
                num_neurons = layer.split("(")[1].strip(")")
                conv = self.LinearBlock(conv, int(num_neurons), name = "{}_dense_{}".format(label, i))
            elif layer.split("(")[0] == "Reshape":
                s1,s2 = layer.split("(")[1].strip(")").split(",")
                conv = tf.keras.layers.Reshape((int(s1),int(s2)))(conv)
            elif layer.split("(")[0] == "Flatten":
                conv = tf.keras.layers.Flatten()(conv)
            elif layer.split("(")[0] == "Transformer":
                head_size, num_heads, ff_dim = layer.split("(")[1].strip(")").split(",")
                conv = self.TransformerEncoder(conv, int(head_size), int(num_heads), int(ff_dim), dropout=0)
            elif layer.split("(")[0] == "MaxPool":
                pool_size = layer.split("(")[1].strip(")")
                conv = tf.keras.layers.MaxPooling1D(pool_size=int(pool_size), strides=None, padding="valid", data_format="channels_last")(conv)
            else:
                raise Exception(f"Error: No layer with name {layer.split('(')[0]}, please use one of Conv1D, SepConv1D, ResBlock, Linear, Reshape, Flatten, Transformer, MaxPool")

        return conv

    def Reshape(self, input_data, shape):
        return 

    def ConvBlock(self, input_data, filters, kernel_size, strides, name = ""):
        #, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer
        conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, kernel_regularizer=tf.keras.regularizers.l2(0.001), padding="same", kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer, name = name)(input_data)
        conv = tf.keras.layers.BatchNormalization(name = "{}_batchnorm".format(name))(conv)
        conv = self.activation(conv)

        return conv

    def SepConvBlock(self, input_data, filters, kernel_size, strides, name = ""):
        #, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer
        conv = tf.keras.layers.SeparableConv1D(filters=filters, kernel_size=kernel_size, strides=strides, depthwise_regularizer=tf.keras.regularizers.l2(0.001), padding="same", depthwise_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer, name = name)(input_data)
        conv = tf.keras.layers.BatchNormalization(name = "{}_batchnorm".format(name))(conv)
        conv = self.activation(conv)

        return conv

    def LinearBlock(self,input_data, num_neurons, name = ""):

        out = tf.keras.layers.Dense(num_neurons, kernel_regularizer=tf.keras.regularizers.l2(0.001), kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer, name = name)(input_data)
        out = tf.keras.layers.BatchNormalization(name = "{}_batchnorm".format(name))(out)
        out = self.activation_relu(out)
        return out

    def ResBlock(self,input_data, filters, kernel_size, strides, name = ""):
        filters1, filters2, filters3 = filters
        kernel_size1,kernel_size2 = kernel_size
        
        conv_short = input_data
        
        conv = tf.keras.layers.Conv1D(filters=filters1, kernel_size=kernel_size1, strides=strides, kernel_regularizer=tf.keras.regularizers.l2(0.001), padding="same", kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer, name = "{}_1st".format(name))(input_data)
        conv = tf.keras.layers.BatchNormalization(name="{}_1st_batchnorm".format(name))(conv)
        conv = self.activation(conv)
        
        conv = tf.keras.layers.Conv1D(filters=filters2, kernel_size=kernel_size2, kernel_regularizer=tf.keras.regularizers.l2(0.001), padding="same", kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer, name = "{}_2nd".format(name))(conv)
        conv = tf.keras.layers.BatchNormalization(name="{}_2nd_batchnorm".format(name))(conv)
        conv = self.activation(conv)
        
        conv = tf.keras.layers.Conv1D(filters=filters3, kernel_size=kernel_size1, kernel_regularizer=tf.keras.regularizers.l2(0.001), padding="same", kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer, name = "{}_3rd".format(name))(conv)
        conv = tf.keras.layers.BatchNormalization(name="{}_3rd_batchnorm".format(name))(conv)
        #conv = self.act(conv)
        
        if strides > 1:
            conv_short = tf.keras.layers.Conv1D(filters=filters3, kernel_size=kernel_size1, strides=strides, kernel_regularizer=tf.keras.regularizers.l2(0.001), padding="same", kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer,name = "{}_shortcut".format(name))(conv_short)
            conv_short = tf.keras.layers.BatchNormalization(name="{}_shortcut_batchnorm".format(name))(conv_short)
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

    def TransformerEncoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = tf.keras.layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

