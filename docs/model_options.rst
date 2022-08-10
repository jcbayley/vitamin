=========
Model customisation
=========

You can customise your own model using the "shared" network and "r1", "q" and "r2" networks

The shared network is shared between the three r1, q and r2 networks within the CVAE. This is primamrily used to exract important information for the supplied data.

The "r1" and "q" outputs parameters describing distributions in the latent space and the "r2" network outputs parameters describing deistributions in the real parameter space.

There are two main ways to input the design of each of these networks. The layers can be chosen from a set of commonly used predefined layers which include batch normalisation etc.
The available layers are:

Conv1D 
------
This is input as a string within a list.
1d convolutional layer
.. code-block:: python
    Conv1D(nfilters, filtersize, stride)
Resnet
------
A resnet bottleneck block with three 1d convolutional layers
.. code-block:: python
    ResBlock(nfilters, filtersize, stride)
Linear
------
A dense or linear fully connected layer
.. code-block:: python
    Linear(n_neurons)
Flatten
-------
Used to flatten the outputs of convolutional layers to put into linear layer
.. code-block:: python
    Flatten()
Reshape

One can then define the shared, r1, q and r2 network:
.. code-block:: python 
    shared_network = ['Conv1D(16,16,2)','Conv1D(16,8,2)','Flatten()']
    r1_network = ['Linear(128)','Linear(32)']
    r2_network = ['Linear(128)','Linear(32)']
    q_network = ['Linear(128)','Linear(32)']


The second method to define each of the networks is to define them as custom models, an example of this is below

.. code-block:: python 

    shared_network = tf.keras.Sequential([tf.keras.layers.Conv1D(8,3, name="conv1", activation="relu"), 
                                        tf.keras.layers.Conv1D(8,3, name="conv2", activation="relu"),
                                        tf.keras.layers.Flatten()])

    r1_network = tf.keras.Sequential([tf.keras.layers.Dense(32, name="r1lin1",activation="relu"),
                                    tf.keras.layers.Dense(16, name="r1lin2",activation="relu")])

    q_network = tf.keras.Sequential([tf.keras.layers.Dense(32, name="qlin1",activation="relu"),
                                    tf.keras.layers.Dense(16, name="qlin2",activation="relu")])

    r2_network = tf.keras.Sequential([tf.keras.layers.Dense(32, name="r2lin1",activation="relu"),
                                    tf.keras.layers.Dense(16, name="r2lin2",activation="relu")])