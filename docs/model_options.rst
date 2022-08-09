=========
Model customising
=========

You can customise your own model using the "shared" network and "output" networks

The shared network is shared between the three r1, q and r2 networks within the CVAE. This is primamrily used to exract important information for the supplied data.

The output network then combines this distilled data with parameters or the latent space to output the latent space or the real space.

The avaible layers are:

Conv1D 
Resnet
Linear
Flatten
Reshape

One can also input each of the r1,r1 and q networks in indivudually by writing the r1_network option input to the model. see the examples folder for examples of how to do this 
