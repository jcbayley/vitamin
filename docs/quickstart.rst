==========
Quickstart
==========

This section will detail how you can produce samples from a gravitational wave (GW) postrior within 
minutes of downloading VItamin given your own GW time series binary black hole data. You may also 
optionally use the pregenerated sets of waveforms provided here

Some things to note about the provided black box:

* We currently set all priors to be uniformly distributed and produce posteriors on 15 parameters (chirp_mass, mass_ratio,luminosity distance,time of coalescence,inclination angle, spin_magnitude1, spin_magnitude2, spin_angle1, spin_angle2, phi_jl, phi_12, psi, right ascension and declination). Both phase and psi are internally marginalized out.

* Only works on binary black hole signals (more boxes trained on other signals to be released in the future).

* Does not require a GPU to generate samples, but will be ~1s slower to generate all samples per time series.  

==========
Quickstart from console
==========

To create a default config file and directory to run code from run:

.. code-block:: console

   $ python -m vitamin.initialise_directory --out-dir your_run_directory
   $ cd your_run_directory

This will create a config file with various parameters that you can modify. Here you should write the output and data directores as complete paths not relative paths.
Also a prior file is created which is the default prior distributions that we use (also can be modified).

To create a set of files to generate training, validation and test data run.

.. code-block:: console

   $ python -m vitamin.gw.make_condor_files --out-dir your_run_directory

This will create a condor directory which contains a set of training/validation/testing submit and dag files and a set of bash scripts which will generate some data.
Running the dag files will create all of the training/validation/test data and run any standard PE code to compare to (currently only runs dynesty and nessai).

Once the data has began training one can train a model using

.. code-block:: console

   $ python -m vitamin.gw.train --ini-file config.ini

which will output plots and information to your_run_directory

==========
Quickstart from notebook
==========

!! in progress - Not working yet!!

* Start an ipython notebook (or Google Colab Notebook)

.. code-block:: console

   $ ipython3

* import vitamin_b and run_vitamin module

.. code-block:: console

   $ import vitamin

.. note:: Test samples should be of the format 'data_<test sample number>.h5py'. Where the h5py file 
   should have a directory containing the noisy time series labeled 'y_data_noisy'. 
   'y_data' should be a numpy array of shape (<number of detectors>,<sample rate X duration>) 

* To produce test sample posteriors using VItamin, simply point vitamin to the directory containing your test waveforms (examples provided `here <https://drive.google.com/file/d/1yWZOzvN8yf9rB_boRbXg70nEqhmb5Tfc/view?usp=sharing>`_), the pre-trained model (`model download <https://drive.google.com/file/d/1GSdGX2t2SoF3rencUnQ1mZAyoxO5F-zl/view?usp=sharing>`_) and specify the number of samples per posterior requested.

.. code-block:: console

   $ model = vitamin.load_model("path_to_checkpoint.ckpt")
   $ samples = model.gen_samples(test_data)

   $ vitamin.generate_posterior("path_to_checkpoint.ckpt", test_data)

* The function will now return a set of samples from the posterior per timeseries(default is 10000). 

* Since we set the option plot_corner=True, you will also find a corner plot in the same directory as we ran the code under the title 'vitamin_example_corner.png'.

