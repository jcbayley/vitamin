==========
Output distributions
==========

There are a number of available output distributions in this package some of which are specific to certain binary black hole parameters and some are for general use.

The available distributions are:

Normal

TruncatedNormal

VonMises

JointVonMisesFisher

====
GW specific distributions
====

JointM1M2

JointChirpmassMR

JointChirpmassMRM1M2

DiscardChirpmassMRM1M2

Each of these can be input into the networks design when defining the CVAE model.
For example one can define the parameters to fit and the output ditributions that they would like to use for each of those parameters.
In this example for a straight line with parameter m and c, we can use a trucated normal distribution where the truncation limits are defined by the parameters bounds defined in the bounds dictionary.

.. code-block:: python

    inf_pars = {"m":"TruncatedNormal","c":"TruncatedNormal"}
    bounds = {"m_min":0,"m_max":1,"c_min":0,"c_max":1}